"""Типы фактов: связь из закрытого списка и вид каждого конца — по цитате.

Модель выписывает связь словами статьи («associated with», «вмещает», «отображено
в»), и на 7350 подтверждённых фактов вышло 1693 разных связи (1155 — по одному
разу); среди «предметов» — «Figure 2», «аншлифы», «эталон». Закрыть список прямо
при выписывании не вышло (проба на 40 фрагментах, .local/logs/graph-v2-20261014):
модель перефразирует статью, и проверка вопросами по цитате (confirm.py), которая
держится слов предложения, отбрасывает почти всё — 12 фактов вместо 79.

Поэтому тип ставится после подтверждения, отдельным шагом: модель видит факт с
цитатой и выбирает связь из RELATIONS (или «нет» — факт не о геологии: рисунок,
цвет на карте, расчёт, свойство) и вид каждого конца из KINDS; код проверяет, что
виды подходят к связи (перепутанное направление разворачивает) и что объект назван.
Граф (api.Graph) берёт тип вместо слов статьи; факт с «нет» в граф не идёт, но в
базе остаётся. Слова статьи и цитата — при факте, как были. Тип ставится один раз
(typed_at); --redo — заново, когда поменялся список.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import store

BATCH = 10  # фактов в одном запросе к модели
QUOTE_CHARS = 400  # цитата в запросе — не длиннее
ANSWER_TOKENS = 1200

KINDS = (
    "объект", "структура", "порода", "минерал", "элемент", "оруденение", "процесс", "признак",
    "метод",
)  # fmt: skip
_ALL = frozenset(KINDS)


def _kinds(*names: str) -> frozenset[str]:
    return frozenset(names)


# Связи и виды их концов (от, к). Другой связи в графе не бывает.
RELATIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # Только в объект или структуру: по «входит в» датасет собирает части целого, и
    # «граниты» с «порода → порода» собирали в себя мигматиты и сланцы.
    "входит в": (_kinds("объект", "структура", "порода"), _kinds("объект", "структура")),
    "сложено": (_kinds("объект", "структура", "порода"), _kinds("порода", "минерал")),
    "локализовано в": (_ALL - {"метод"}, _kinds("порода", "структура", "объект")),
    "контролируется": (
        _kinds("оруденение", "объект", "порода", "минерал", "элемент", "процесс"),
        _kinds("структура", "порода", "процесс"),
    ),
    "содержит": (
        _kinds("порода", "оруденение", "объект", "минерал", "процесс", "признак"),
        _kinds("минерал", "элемент"),
    ),
    "ассоциирует с": (_kinds("минерал", "элемент", "оруденение"), _kinds("минерал", "элемент")),
    "сопровождается": (
        _kinds("оруденение", "объект", "порода", "структура", "элемент", "процесс", "минерал"),
        _kinds("процесс", "минерал", "признак"),
    ),
    "генетически связано с": (
        _kinds("оруденение", "объект", "минерал", "порода", "элемент", "процесс"),
        _kinds("процесс", "порода"),
    ),
    "образовано при": (_ALL - {"метод", "признак"}, _kinds("процесс")),
    "указывает на": (
        _kinds("признак", "минерал", "процесс", "структура", "элемент", "порода"),
        _kinds("оруденение", "элемент", "объект"),
    ),
    "выявлено методом": (_ALL - {"метод"}, _kinds("метод")),
    "пересекает": (_kinds("структура", "порода"), _kinds("структура", "порода")),
    "сменяется": (
        _kinds("процесс", "порода", "оруденение", "минерал"),
        _kinds("процесс", "порода", "оруденение", "минерал"),
    ),
    "относится к": (
        _kinds("объект", "оруденение", "порода", "структура", "минерал"),
        _kinds("оруденение", "порода", "объект", "структура", "минерал"),
    ),
}
# Как назвать связь со стороны «от» (→) и со стороны «к» (←) — в сводке о предмете.
TITLES: dict[str, tuple[str, str]] = {
    "входит в": ("входит в", "включает"),
    "сложено": ("сложено", "слагает"),
    "локализовано в": ("локализовано в", "вмещает"),
    "контролируется": ("контролируется", "контролирует"),
    "содержит": ("содержит", "содержится в"),
    "ассоциирует с": ("ассоциирует с", "ассоциирует с"),
    "сопровождается": ("сопровождается", "сопровождает"),
    "генетически связано с": ("генетически связано с", "генетически связано с"),
    "образовано при": ("образовано при", "приводит к образованию"),
    "указывает на": ("указывает на", "выявляется по признакам"),
    "выявлено методом": ("выявлено методом", "позволяет выявить"),
    "пересекает": ("пересекает", "пересекается"),
    "сменяется": ("сменяется", "сменяет"),
    "относится к": ("относится к", "объединяет"),
}
_MEANING = {
    "входит в": "часть в целом, объект в районе («Билляхская зона» → «Анабарский щит»)",
    "сложено": "из каких пород или минералов состоит («Igarra Schist Belt» → «phyllite»)",
    "локализовано в": "где находится, что вмещает («оруденение» → «черные сланцы»)",
    "контролируется": "что управляет размещением («оруденение» → «Персияновский разлом»)",
    "содержит": "состав («кварцевые жилы» → «арсенопирит», «арсенопирит» → «Au»)",
    "ассоциирует с": "встречаются вместе («самородное золото» → «алтаит»)",
    "сопровождается": "что идёт вместе с ним («оруденение» → «березитизация»)",
    "генетически связано с": "источник или причина («оруденение» → «гранитоидный магматизм»)",
    "образовано при": "при каком процессе или этапе образовано",
    "указывает на": "поисковый признак («аномалии As» → «золотое оруденение»)",
    "выявлено методом": "чем найдено («зоны аргиллизации» → «ASTER»)",
    "пересекает": "одна структура или порода пересекает другую",
    "сменяется": "смена во времени или в пространстве, зональность",
    "относится к": "тип или класс («Белоусовское» → «колчеданно-полиметаллические месторождения»)",
}
# Подвиды, которыми модель называет вид, → вид: «месторождение» → объект.
_OBJECT_ALIASES = (
    "месторождение", "рудопроявление", "массив", "район", "рудный район", "рудный узел",
    "рудное поле", "щит", "пояс", "бассейн", "террейн", "свита", "комплекс", "провинция",
    "место", "территория",
)  # fmt: skip
_KIND_ALIASES = {
    **dict.fromkeys(_OBJECT_ALIASES, "объект"),
    **dict.fromkeys(("разлом", "зона", "линеамент", "складка"), "структура"),
    **dict.fromkeys(("интрузия", "жила", "дайка", "руда", "толща", "горная порода"), "порода"),
    **dict.fromkeys(("минерализация", "тип месторождений"), "оруденение"),
    **dict.fromkeys(("изменение", "метасоматоз", "этап"), "процесс"),
    **dict.fromkeys(("аномалия", "ореол"), "признак"),
    **dict.fromkeys(("данные", "съёмка", "съемка"), "метод"),
    **dict.fromkeys(("химический элемент", "полезное ископаемое", "металл"), "элемент"),
}
# Объект без названия: «рудный район», «изучаемая площадь», «study area» — какой?
_OBJECT_WORDS = {
    "рудный", "рудная", "рудное", "рудные", "район", "узел", "поле", "площадь", "участок",
    "территория", "месторождение", "рудопроявление", "проявление", "бассейн", "пояс", "щит",
    "массив", "свита", "комплекс", "блок", "провинция", "регион", "изучаемая", "изучаемый",
    "исследуемая", "исследуемый", "эталонный", "эталон", "золоторудное", "золоторудный",
    "deposit", "deposits", "district", "belt", "field", "area", "basin", "province",
    "complex", "block", "study", "the", "ore", "region", "zone",
}  # fmt: skip
_WORD = re.compile(r"[a-zа-яё]+(?:-[a-zа-яё]+)?")
# Прилагательное от имени собственного: «шахтаминский комплекс», «хомолхинская свита»
# (а «магматический», «тектоническая» — нет).
_PROPER_ADJ = re.compile(r"(?<!че)(?:ск|цк)(?:ий|ая|ое|ие|ой|ого|ую)$")
# Сторона или часть территории, а не предмет: «north-east», «N, NW and southern
# regions», «западная часть района»; положение: «between the SNNF and SNSF».
_SIDE = re.compile(
    r"^(?:[nsew]{1,3}|(?:north|south)?(?:east|west)?(?:ern|erly)?|north(?:ern)?|"
    r"south(?:ern)?|central|центральн\w*|(?:северо-|юго-)?(?:север|юг|восток|запад)\w*)$"
)
_SIDE_FILLER = {
    "and", "или", "и", "the", "of", "part", "parts", "часть", "части", "region", "regions",
    "area", "areas", "sector", "сектор", "side", "margin", "окраина", "района", "территории",
}  # fmt: skip
_POSITION = re.compile(r"^(?:between|within|among|между|в\s+пределах|пределах)\b", re.IGNORECASE)
HEDGE = "предположительно"
# Объект без названия — обычно вид перепутан: «bauxites», «basement rocks», «gold
# occurrences» модель зовёт объектом. Вид угадывается по словам имени; не угадался —
# факт не в граф. Первый подошедший — по порядку.
_GUESS = (
    (
        "оруденение",
        re.compile(
            r"deposit|occurrence|minerali[sz]|ore\s?bod|месторожд|проявлени|оруденен|"
            r"минерализац|рудн\w*\s+(?:тел|зон|жил)",
            re.IGNORECASE,
        ),
    ),
    (
        "структура",
        re.compile(
            r"\bzones?\b|fault|shear|fold|lineament|\bзон|разлом|разрыв|складк|трещин", re.I
        ),
    ),
    (
        "порода",
        re.compile(
            r"rock|basement|gneiss|granit|schist|sediment|bauxite|volcanic|intrusi|dyke|dike|"
            r"vein|\bжил|пород|толщ|интрузи|дайк|гранит|сланц|массив",
            re.IGNORECASE,
        ),
    ),
)
# Близкая, но не та связь при этих видах (от, к) → (связь, развернуть ли). «Узел включает
# золото-сурьмяное оруденение» — модель берёт «входит в», а это «локализовано в».
_ORE = frozenset({"оруденение", "минерал", "элемент"})
_HOST = frozenset({"порода", "структура", "объект"})


def _repair(relation: str, a: str, b: str) -> tuple[str, bool]:
    if relation == "содержит" and b == "оруденение":
        return "локализовано в", True
    if relation == "содержит" and b == "структура":
        return "входит в", True
    if relation == "содержит" and b == "порода":
        return "сложено", False
    if relation == "входит в" and a in _ORE:
        return "локализовано в", False
    if relation == "входит в" and b == "порода":
        return "относится к", False
    if relation in {"ассоциирует с", "сопровождается"} and a in _ORE and b in _HOST:
        return "локализовано в", False
    if relation in {"ассоциирует с", "сопровождается", "генетически связано с"} and (
        a == "признак" and b in {"элемент", "оруденение", "объект"}
    ):
        return "указывает на", False
    if relation == "генетически связано с" and a == "структура" and b == "процесс":
        return "образовано при", False
    return relation, False


def _guess_kind(name: str) -> str | None:
    return next((kind for kind, words in _GUESS if words.search(name)), None)


SCHEMA = """
ALTER TABLE facts ADD COLUMN IF NOT EXISTS rel_type text;
ALTER TABLE facts ADD COLUMN IF NOT EXISTS rel_flip boolean;
ALTER TABLE facts ADD COLUMN IF NOT EXISTS src_kind text;
ALTER TABLE facts ADD COLUMN IF NOT EXISTS dst_kind text;
ALTER TABLE facts ADD COLUMN IF NOT EXISTS typed_at timestamptz;
ALTER TABLE facts ADD COLUMN IF NOT EXISTS type_note text;
"""

SYSTEM = "Ты геолог-редактор графа знаний о рудных месторождениях. Отвечай только JSON."
PROMPT = """Ниже факты из геологических статей: два предмета, А и Б, как их связала статья,
и цитата. Для каждого факта определи:

«А», «Б» — вид каждого предмета, одним словом: объект (с названием: месторождение,
рудный район, щит, пояс, свита, комплекс), структура (разлом, зона сдвига, линеамент,
складка), порода (порода, толща, интрузия, дайка, жила, руда), минерал, элемент
(химический элемент, полезный компонент), оруденение (оруденение, минерализация, тип
месторождений), процесс (метасоматоз, изменение пород, магматизм, метаморфизм, этап),
признак (аномалия, ореол, геохимическая ассоциация), метод (метод, данные, съёмка) —
или «не предмет»: общее слово без названия, сторона света, часть территории, проба,
рисунок, число, свойство.

«связь» — какая это связь из списка, слово в слово, или «нет»:
{relations}
«нет» — если факт не о геологии (рисунок, таблица, цвет на карте, расчёт, параметр
модели, свойство вроде размера, формы, пробности, содержания), если цитата этого не
утверждает или если ни одна связь не подходит.

«от» — с какого предмета связь читается: "А" или "Б". «Щит включает Билляхскую
зону» — это «Билляхская зона входит в щит»: «от» — тот предмет, что зона.

Например:
1. А = «щит», Б = «Билляхская зона»; в статье: А — включает — Б.
2. А = «Figure 2», Б = «разлом»; в статье: А — иллюстрирует — Б.
Ответ: {{"1": {{"А": "объект", "Б": "структура", "связь": "входит в", "от": "Б"}},
 "2": {{"А": "не предмет", "Б": "структура", "связь": "нет", "от": "А"}}}}

Факты:
{facts}

Верни JSON: {{"1": {{"А": "...", "Б": "...", "связь": "...", "от": "А"}}, ...}}"""


def init(conn: Any) -> None:
    """Колонки типа у facts — если их ещё нет. ALTER TABLE ждёт, пока закончатся все
    транзакции с facts, и, пока ждёт, не пускает к ним других: на работающем сервере
    чат-бот вставал бы. Поэтому — только когда колонок нет, и ждать не дольше 10 с."""
    if store.has_types(conn):
        return
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '10s'")
        cur.execute(SCHEMA)
    conn.commit()


def kind_of(raw: Any) -> str | None:
    kind = " ".join(str(raw or "").split()).strip(" .,;:«»\"'").lower()
    return kind if kind in KINDS else _KIND_ALIASES.get(kind)


def relation_of(raw: Any) -> str | None:
    text = " ".join(str(raw or "").split()).strip(" .,;:«»\"'").lower()
    return text if text in RELATIONS else None


# Свойство или проба, а не предмет — в любом падеже: «аншлифах», «гранулометрией»,
# «составом». Имя целиком из таких слов (и определений к ним) не годится.
_PROPERTY = re.compile(
    r"^(?:аншлиф|шлиф|проб|протолочн|керн|гранулометр|морфолог|состав|пробност|содержани|"
    r"размер|форм|текстур|структур[аеуы]\s+зерен|цвет|sample|specimen|grain|size|shape|"
    r"outcrop|colou?r|texture|content)",
)


def bad_name(name: str, kind: str) -> str | None:
    """Имя не годится для вида: объект без названия, сторона света вместо предмета,
    свойство или проба."""
    from .facts import _bad_name

    words = _WORD.findall(name.lower())
    if words and all(_PROPERTY.match(w) for w in words):
        return "свойство или проба, а не предмет"
    reason = _bad_name(name)
    if reason:
        return reason
    tokens = re.findall(r"[a-zа-яё]+(?:-[a-zа-яё]+)?", name.lower())
    sides = [t for t in tokens if _SIDE.match(t)]
    if sides and all(t in _SIDE_FILLER or _SIDE.match(t) for t in tokens):
        return "сторона или часть территории"
    if _POSITION.match(name.strip()):
        return "положение, а не предмет"
    if kind == "объект":
        words = _WORD.findall(name.lower())
        named = (
            any(c.isupper() or c.isdigit() for c in name)
            or any(_PROPER_ADJ.search(w) for w in words)
        ) and not all(w in _OBJECT_WORDS for w in words)
        if not named:
            return "объект без названия"
    return None


def judge(answer: Any, src: str, dst: str) -> tuple[str | None, bool, str, str, str]:
    """Ответ модели по факту → (связь или None, развернуть ли, вид от, вид к, пометка).

    Виды — для «от» (А) и «к» (Б) в том виде, как факт хранится; развернуть — связь
    читается от Б к А. Пометка — почему отброшен (или пусто)."""
    if not isinstance(answer, dict):
        return None, False, "", "", "нет ответа"
    relation = relation_of(answer.get("связь"))
    sk, dk = kind_of(answer.get("А")), kind_of(answer.get("Б"))
    if relation is None:
        said = str(answer.get("связь") or "").strip().lower()
        return None, False, sk or "", dk or "", "не о геологии" if said == "нет" else "связь"
    if sk is None or dk is None:
        return None, False, sk or "", dk or "", "не предмет"
    flip = str(answer.get("от") or "А").strip().upper() in {"Б", "B"}
    ends = [sk, dk]
    for i, end in enumerate((src, dst)):
        if ends[i] == "объект" and bad_name(end, "объект"):
            guess = _guess_kind(end)
            if guess is None:
                return None, flip, sk, dk, "объект без названия"
            ends[i] = guess
    sk, dk = ends
    a, b = (dk, sk) if flip else (sk, dk)
    relation, turn = _repair(relation, a, b)
    flip = flip != turn
    allowed_src, allowed_dst = RELATIONS[relation]
    a, b = (dk, sk) if flip else (sk, dk)
    if not (a in allowed_src and b in allowed_dst):
        if b in allowed_src and a in allowed_dst:  # направление перепутано — развернуть
            flip = not flip
        else:
            return None, flip, sk, dk, f"виды не подходят к связи: {relation}"
    reason = bad_name(src, sk) or bad_name(dst, dk)
    if reason:
        return None, flip, sk, dk, reason
    return relation, flip, sk, dk, ""


def prompt(rows: list[tuple[Any, ...]]) -> str:
    relations = "\n".join(
        f"- «{r}» — {', '.join(sorted(s))} → {', '.join(sorted(d))}: {_MEANING[r]}"
        for r, (s, d) in RELATIONS.items()
    )
    facts = "\n".join(
        f"{i}. А = «{src}», Б = «{dst}»; в статье: А — {rel} — Б. "
        f"Цитата: «{' '.join(quote.split())[:QUOTE_CHARS]}»"
        for i, (_, src, rel, dst, quote) in enumerate(rows, start=1)
    )
    return PROMPT.format(relations=relations, facts=facts)


def recheck(conn: Any) -> dict[str, int]:
    """Правила кода (bad_name) — заново у фактов с типом, без модели: правила поменялись."""
    init(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id, src, dst, src_kind, dst_kind FROM facts WHERE rel_type IS NOT NULL")
        rows = cur.fetchall()
        dropped = 0
        for fact_id, src, dst, sk, dk in rows:
            reason = bad_name(src, sk or "") or bad_name(dst, dk or "")
            if reason:
                cur.execute(
                    "UPDATE facts SET rel_type = NULL, type_note = %s WHERE id = %s",
                    (reason, fact_id),
                )
                dropped += 1
    conn.commit()
    return {"checked": len(rows), "dropped": dropped}


def todo(conn: Any, limit: int | None = None) -> list[tuple[Any, ...]]:
    """Подтверждённые факты без типа: (id, от, связь, к, цитата)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, src, relation, dst, quote FROM facts
            WHERE typed_at IS NULL AND votes >= %s ORDER BY id LIMIT %s
            """,
            (store.MIN_VOTES, limit),
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def run(
    conn: Any,
    llm: Any,
    limit: int | None = None,
    workers: int = 1,
    redo: bool = False,
    log: Any = print,
    again: list[str] | None = None,
) -> dict[str, Any]:
    """Типы для подтверждённых фактов, у которых их ещё нет. Возобновляемо.

    redo — всё заново; again — заново только отброшенные по этим причинам (правила
    кода поменялись: «виды не подходят к связи», «объект без названия»)."""
    init(conn)
    if redo or again:
        where = "" if redo else "WHERE rel_type IS NULL AND split_part(type_note, ':', 1) = ANY(%s)"
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE facts SET rel_type = NULL, rel_flip = NULL, src_kind = NULL, "
                f"dst_kind = NULL, typed_at = NULL, type_note = NULL {where}",
                () if redo else (list(again or []),),
            )
        conn.commit()
    rows = todo(conn, limit)
    batches = [rows[i : i + BATCH] for i in range(0, len(rows), BATCH)]
    stats: dict[str, Any] = {"todo": len(rows), "typed": 0, "dropped": 0, "errors": 0}
    notes: Counter[str] = Counter()
    types: Counter[str] = Counter()

    def ask(batch: list[tuple[Any, ...]]) -> Any:
        return llm.chat_json(SYSTEM, prompt(batch), temperature=0, max_tokens=ANSWER_TOKENS)

    started = time.monotonic()
    pool = ThreadPoolExecutor(max(1, workers))
    try:
        for start in range(0, len(batches), max(1, workers)):
            group = batches[start : start + max(1, workers)]
            futures = [pool.submit(ask, b) for b in group]
            for batch, future in zip(group, futures, strict=True):
                try:
                    answer = future.result()
                except Exception as exc:  # noqa: BLE001 — эти факты пройдут в следующий раз
                    stats["errors"] += 1
                    if log:
                        log(f"  модель не ответила: {type(exc).__name__}: {exc}")
                    continue
                answer = answer if isinstance(answer, dict) else {}
                with conn.cursor() as cur:
                    for i, (fact_id, src, _rel, dst, _quote) in enumerate(batch, start=1):
                        relation, flip, sk, dk, note = judge(answer.get(str(i)), src, dst)
                        if note == "нет ответа":
                            continue  # модель пропустила факт — спросим в следующий раз
                        cur.execute(
                            """
                            UPDATE facts SET rel_type = %s, rel_flip = %s, src_kind = %s,
                                dst_kind = %s, typed_at = now(), type_note = %s WHERE id = %s
                            """,
                            (relation, flip, sk or None, dk or None, note or None, fact_id),
                        )
                        if relation:
                            stats["typed"] += 1
                            types[relation] += 1
                        else:
                            stats["dropped"] += 1
                            notes[note.split(":")[0]] += 1
                conn.commit()
            done = min(len(rows), (start + len(group)) * BATCH)
            if log and (start // max(1, workers)) % 10 == 0:
                pace = (time.monotonic() - started) / max(1, done)
                log(
                    f"  [{done}/{len(rows)}] с типом {stats['typed']}, не в граф "
                    f"{stats['dropped']} · {pace:.2f} с на факт"
                )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    stats["notes"] = dict(notes.most_common())
    stats["types"] = dict(types.most_common())
    return stats
