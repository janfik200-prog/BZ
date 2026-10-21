"""Подтверждает ли фрагмент утверждение, которое на него ссылается.

Ссылка на фрагмент проверялась только по номеру: есть ли такой фрагмент. А что он
говорит, никто не сверял — и в ответ попадало чужое: в AgentKB (20.10.2026) место из
одной статьи приписано другой, ограничение решёточной КХД — «расчётам eLSM». Здесь
каждое утверждение со ссылкой сверяется с текстом своих фрагментов:

1. числа — кодом, без модели: числа утверждения нет в его фрагментах, а все они — в
   одном другом, ссылка переставляется на тот;
2. смысл — одним вызовом модели на все утверждения: «да» (сказано во фрагменте, хотя бы
   пересказом или переводом), «частично», «нет» с названным «чего нет» (во фрагменте
   этого нет или сказано другое: другое место, объект, число). Опровергнутое
   перепроверяется по одному, только со своими фрагментами; опровергнуто оба раза —
   утверждение убирается из ответа и показывается отдельно.

Модель молчит или ответила не то — ответ остаётся как был: сверка не должна ломать ответ.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .checks import CITE_MARK, about_sources, cited_numbers, is_claim, numbers, rewrite
from .messages import fragment_block
from .settings import Settings

MAX_CLAIMS = 24  # утверждений на проверку самое большее — остальные не трогаются

SUPPORT_SYSTEM = (
    "Ты проверяешь ответ по научным статьям: подтверждает ли фрагмент статьи утверждение, "
    "которое на него ссылается. Отвечай только JSON."
)
# Пример в «чего_нет» — нейтральный: название из базы модель переносит в свой ответ.
SUPPORT_USER = """Фрагменты статей:

{fragments}

Утверждения ответа, у каждого — номера фрагментов, на которые оно ссылается:

{claims}

Для каждого утверждения реши, подтверждают ли его ЕГО фрагменты:
- «да» — сказанное есть во фрагменте: дословно, пересказом, обобщением или в переводе с \
другого языка;
- «частично» — главное есть, но есть и добавка, которой во фрагменте нет (её — в «чего_нет»);
- «нет» — главного во фрагменте нет или сказано другое: другое место или объект, другое \
число, другая работа или метод, чем в утверждении.
Не придирайся к словам, пересказу и переводу. «нет» — только когда можешь назвать, чего \
именно во фрагменте нет: тогда напиши это в «чего_нет» (например, «во фрагменте Франция, \
а не Испания»). Не можешь назвать — значит, не «нет».

Верни JSON: {{{keys}}}"""


def seen(source: dict[str, Any]) -> str:
    """Фрагмент так, как его видела модель при ответе (fragment_block): с заголовком —
    название, авторы, год — и с той же обрезкой. По первым 1800 знакам сверка в AgentKB
    убирала верное, что стояло во фрагменте дальше."""
    return fragment_block(source)


_LITERATURE = re.compile(r"\(лит\. [^)]*\)")


def seen_numbers(source: dict[str, Any]) -> str:
    """Фрагмент, каким его видела модель, без ссылок на литературу «(лит. 10-12)»: их
    номера — не числа статьи. На живой проверке (21.10.2026) «разрешение 10 м» получало
    ссылку на фрагмент, где 10 было только номером в «(лит. 10-12)»."""
    return _LITERATURE.sub(" ", seen(source))


def _distinctive(values: set[str]) -> set[str]:
    """Числа, по которым можно узнать фрагмент: с дробью или от трёх цифр, кроме годов.
    Двузначное целое («30 м») стоит почти в каждом втором фрагменте о снимках, год —
    в каждой ссылке на литературу."""
    return {v for v in values if "." in v or (len(v) >= 3 and not _YEAR.fullmatch(v))}


_YEAR = re.compile(r"1[89]\d\d|20\d\d")


def _claims(answer: str) -> list[str]:
    found: list[str] = []

    def collect(part: str) -> str:
        # Фраза о самих фрагментах («во фрагментах нет точных значений [1]») — не факт
        # статьи: сверять её не с чем, а убиралась бы как неподтверждённая.
        if is_claim(part) and cited_numbers(part) and not about_sources(part):
            found.append(part)
        return part

    rewrite(answer, collect)
    return found


def _recite(claim: str, by_n: dict[int, dict[str, Any]]) -> str | None:
    """Числа утверждения не в его фрагментах, а все — в одном другом: ссылка на тот.
    Среди них должно быть отличительное (_distinctive): на живой проверке 21.10.2026
    «SRTM… до 30 метров [7]» уходило с верного фрагмента на чужой, где тоже было «30 м»."""
    need = numbers(claim)
    if not _distinctive(need):
        return None
    cited = [n for n in cited_numbers(claim) if n in by_n]
    have = set().union(*(numbers(seen_numbers(by_n[n])) for n in cited)) if cited else set()
    if need <= have:
        return None
    others = [n for n, s in by_n.items() if n not in cited and need <= numbers(seen_numbers(s))]
    if len(others) != 1:  # ни одного или несколько — не угадывать
        return None
    other = others[0]
    end = re.search(r"[.!?…]$", claim.rstrip())
    return CITE_MARK.sub("", claim).rstrip(" .!?…") + f" [{other}]" + (end.group() if end else "")


# Имя собственное: слово с заглавной от четырёх знаков, не после точки и не внутри слова.
_PROPER = re.compile(r"(?<![\w.])[A-ZА-ЯЁ][\w-]{3,}")


def _stems(text: str, skip_first: bool = False) -> set[str]:
    words = _PROPER.findall(text)
    if skip_first and words and text.lstrip().startswith(words[0]):
        words = words[1:]  # первое слово предложения — с заглавной и не будучи именем
    return {w.lower()[:5] for w in words}


def _foreign(reason: str, claim: str, fragments: str) -> bool:
    """В «чего нет» — имя или число из самого утверждения, и в его фрагментах их правда
    нет: место, объект или величина, приписанные фрагменту. Это не добавка к верному.
    Имя, которое во фрагментах есть, — не чужое: в AgentKB модель называла имя «нет во
    фрагментах 1, 2…, кроме 3», а ссылка на 3 и стояла."""
    if {n for n in numbers(reason) & numbers(claim) if n not in numbers(fragments)}:
        return True
    have = fragments.lower()
    named = _stems(reason) & _stems(CITE_MARK.sub("", claim), skip_first=True)
    return any(stem not in have for stem in named)


def _refuted(verdict: Any, claim: str = "", fragments: str = "") -> bool:
    """Опровергнуто ли утверждение: «нет» с названным «чего нет» — или «частично», где
    «чего нет» — имя или число из утверждения, которых нет в его фрагментах. Без причины
    — не опровергнуто: голое «нет» стояло и у пересказа, который фрагмент прямо
    подтверждает, а место, приписанное статье о другом месте, модель называла
    «частично, чего нет: <место>» (AgentKB, 20.10.2026)."""
    if not isinstance(verdict, dict):
        return False
    answer = str(verdict.get("ответ") or "").strip().lower()
    reason = str(verdict.get("чего_нет") or "").strip()
    if len(reason) < 4:
        return False
    return answer == "нет" or (answer == "частично" and _foreign(reason, claim, fragments))


def _ask(claims: list[str], by_n: dict[int, dict[str, Any]], settings: Settings) -> list[bool]:
    """Модель о каждом утверждении: опровергнуто ли оно его фрагментами. Молчит — нет."""
    cited = sorted({n for c in claims for n in cited_numbers(c) if n in by_n})
    blocks = "\n\n".join(seen(by_n[n]) for n in cited)
    lines = [
        f"{i}) «{CITE_MARK.sub('', c).strip()}» — фрагменты {', '.join(map(str, cited_numbers(c)))}"
        for i, c in enumerate(claims, start=1)
    ]
    # Ключи — в значение подстановки: фигурные скобки в нём одинарные, .format их не трогает.
    keys = ", ".join(
        f'"{i}": {{"ответ": "да|частично|нет", "чего_нет": ""}}' for i in range(1, len(claims) + 1)
    )
    try:
        data = settings.client().chat_json(
            SUPPORT_SYSTEM,
            SUPPORT_USER.format(fragments=blocks, claims="\n".join(lines), keys=keys),
            temperature=0,
            max_tokens=40 * len(claims) + 60,
        )
    except Exception:  # noqa: BLE001 — не вышло сверить: ничего не опровергнуто
        return [False] * len(claims)
    data = data if isinstance(data, dict) else {}
    return [
        _refuted(
            data.get(str(i)),
            c,
            " ".join(seen(by_n[n]) for n in cited_numbers(c) if n in by_n),
        )
        for i, c in enumerate(claims, start=1)
    ]


def check_support(
    question: str, answer: str, sources: list[dict[str, Any]], settings: Settings
) -> tuple[str, list[str], int]:
    """Ответ без неподтверждённых утверждений: (текст, убранные, сколько ссылок переставлено)."""
    by_n = {s["n"]: s for s in sources}
    claims = _claims(answer)[:MAX_CLAIMS]
    if not claims:
        return answer, [], 0

    fixed = {c: new for c in claims if (new := _recite(c, by_n)) is not None}
    checked = [fixed.get(c, c) for c in claims]
    first = _ask(checked, by_n, settings)
    # Опровергнутое — ещё раз, по одному, только со своими фрагментами: в AgentKB общий
    # проход опровергал и верное (около трети из «нет»), а узкая перепроверка ошибается
    # реже. Убирается то, что опровергнуто оба раза.
    doubtful = [i for i, no in enumerate(first) if no]
    again: list[bool] = []
    if doubtful:
        with ThreadPoolExecutor(min(4, len(doubtful))) as pool:
            again = [
                r[0] if r else False
                for r in pool.map(lambda i: _ask([checked[i]], by_n, settings), doubtful)
            ]
    refuted = {claims[i] for i, no in zip(doubtful, again, strict=True) if no}
    removed: list[str] = []

    def change(part: str) -> str | None:
        if part in refuted:
            removed.append(fixed.get(part, part))
            return None
        return fixed.get(part, part)

    return rewrite(answer, change), removed, len(fixed)
