"""Разметка фрагментов по понятиям словаря: какие фрагменты что описывают.

Два шага, и первый обходится без языковой модели.

**Шаг 1 — кандидаты.** У каждого понятия в словаре есть определение в
несколько предложений. Оно переводится в вектор той же моделью, что и
фрагменты, и сравнивается со всеми фрагментами базы. Параллельно фрагменты
проверяются на слова словаря. Кандидат — фрагмент, который:

* близок к определению по смыслу (похожесть не ниже `min_similarity`) — «вектор»;
* или содержит слово словаря и при этом хоть сколько-то близок к определению
  (не ниже `min_similarity_terms`) — «словарь». Одного слова мало: «разлом»
  встречается и там, где о структурном контроле нет речи;
* или то и другое сразу — «оба». Самый надёжный случай.

Цитатой становится предложение фрагмента с наибольшим числом слов словаря,
а если слов нет — предложение, ближе всего стоящее к определению по смыслу.

**Шаг 2 — проверка моделью.** Каждому кандидату один закрытый вопрос:
описывает ли фрагмент понятие по существу, и если да — дословная цитата.
Цитата сверяется с текстом кодом. Если её во фрагменте нет, модель
переспрашивается один раз; нет и тогда — кандидат отклоняется. Так выдумка
модели не проходит в базу без участия человека.

Шаг 2 возобновляемый: проверяется только то, что ещё не проверено. Прогон
можно прервать и продолжить, решения модели переживают перестроение разметки.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from . import store
from .vocabulary import Vocabulary

MIN_SIMILARITY = 0.55          # только по смыслу: должно быть действительно близко
MIN_SIMILARITY_TERMS = 0.45    # слово словаря есть: достаточно, чтобы не было совсем мимо
MAX_PER_CONCEPT = 300          # потолок кандидатов на понятие — модели их потом проверять
QUOTE_MAX = 500


@dataclass(frozen=True)
class Candidate:
    concept: str
    chunk_id: int
    doc_id: str
    quote: str
    similarity: float | None
    terms: tuple[str, ...]
    found_by: str               # вектор | словарь | оба


# --------------------------------------------------------------------------- #
#  Предложения и цитаты
# --------------------------------------------------------------------------- #
# Сокращения, после которых точка не кончает предложение.
_ABBREV = re.compile(
    r"(?:\b(?:т\.\s?е|т\.\s?к|т\.\s?н|и\s?др|и\s?т\.\s?д|и\s?т\.\s?п|см|рис|табл|гг?|вв?|"
    r"с|стр|им|ок|млн|млрд|тыс|кв|e\.g|i\.e|et\s?al|fig|figs|tab|vs|approx)"
    r"|\b[А-ЯЁA-Z])\.$",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[]?[А-ЯЁA-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Текст → предложения. Сокращения и инициалы предложение не рвут."""
    text = " ".join((text or "").split())
    if not text:
        return []
    pieces = _SPLIT.split(text)
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ABBREV.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


def _cut(sentence: str, limit: int = QUOTE_MAX) -> str:
    """Длинное предложение обрезается по слову. Остаётся дословным началом."""
    if len(sentence) <= limit:
        return sentence
    head = sentence[:limit]
    return head[: head.rfind(" ")] if " " in head else head


def quote_by_terms(sentences: list[str], matcher) -> str | None:
    """Предложение с наибольшим числом разных слов словаря; при равенстве — раннее."""
    best, best_n = None, 0
    for sentence in sentences:
        n = len(matcher.found(sentence))
        if n > best_n:
            best, best_n = sentence, n
    return _cut(best) if best else None


# --------------------------------------------------------------------------- #
#  Сверка цитаты с текстом
# --------------------------------------------------------------------------- #
_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‘": "'",
                         "’": "'", "–": "-", "—": "-", "ё": "е", "Ё": "е"})


def _canon_with_map(text: str) -> tuple[str, list[int]]:
    """Текст для сравнения и карта: символ сравниваемого текста → место в исходном.

    Сравнение не различает регистр, «ё/е», виды кавычек и тире и пробелы —
    модель переписывает их как хочет. Карта нужна, чтобы вернуть цитату
    в том виде, в каком она стоит в статье, а не в пересказе модели.
    """
    out: list[str] = []
    where: list[int] = []
    last_space = True
    for i, ch in enumerate(text):
        if ch == "­":            # мягкий перенос
            continue
        if ch.isspace():
            if not last_space:
                out.append(" ")
                where.append(i)
            last_space = True
            continue
        out.append(ch.translate(_QUOTES).lower())
        where.append(i)
        last_space = False
    return "".join(out), where


def locate_quote(quote: str, text: str) -> str | None:
    """Найти цитату во фрагменте; вернуть её в написании статьи или None.

    Модель иногда сокращает цитату многоточием — тогда каждая часть должна
    найтись в тексте по порядку, и возвращается кусок от первой до последней.
    """
    parts = [p.strip(" ,;:") for p in re.split(r"…|\.\.\.", quote or "")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 20:
        return None

    canon_text, where = _canon_with_map(text)
    start_at, first, last = 0, None, None
    for part in parts:
        canon_part = _canon_with_map(part)[0].strip()
        pos = canon_text.find(canon_part, start_at)
        if pos < 0:
            return None
        if first is None:
            first = pos
        last = pos + len(canon_part)
        start_at = last
    begin = where[first]
    end = where[last - 1] + 1
    return " ".join(text[begin:end].split())


# --------------------------------------------------------------------------- #
#  Шаг 1: кандидаты
# --------------------------------------------------------------------------- #
def _similarities(conn, vector: list[float], floor: float) -> dict[int, float]:
    """Похожесть определения на все фрагменты не ниже порога — точно, без индекса.

    Индекс HNSW приблизительный и отдаёт первые N; здесь нужен полный список
    всего, что выше порога, а фрагментов в базе — десятки тысяч, не миллионы.
    """
    from ..index.db import vector_literal

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %(q)s::vector)
            FROM chunks
            WHERE embedding IS NOT NULL AND 1 - (embedding <=> %(q)s::vector) >= %(floor)s
            """,
            {"q": vector_literal(vector), "floor": floor},
        )
        return {int(r[0]): float(r[1]) for r in cur.fetchall()}


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def find_candidates(
    conn,
    embedder,
    vocab: Vocabulary,
    chunks: list[tuple],
    min_similarity: float = MIN_SIMILARITY,
    min_similarity_terms: float = MIN_SIMILARITY_TERMS,
    max_per_concept: int = MAX_PER_CONCEPT,
    log=print,
) -> tuple[list[Candidate], dict[str, dict]]:
    """Кандидаты по всем понятиям и сводка по каждому: сколько, чем найдено."""
    texts = {cid: (doc_id, text or "") for cid, doc_id, text in chunks}
    concepts = list(vocab.concepts.values())
    vectors = embedder.encode([f"{c.code}. {c.definition}" for c in concepts])
    floor = min(min_similarity, min_similarity_terms)

    candidates: list[Candidate] = []
    need_sentence_quote: list[tuple[int, list[str], int]] = []  # (индекс, предложения, понятие)
    report: dict[str, dict] = {}

    for ci, (concept, vector) in enumerate(zip(concepts, vectors)):
        sims = _similarities(conn, vector, floor)
        picked: list[tuple[int, float, tuple[str, ...], str]] = []
        for chunk_id, sim in sims.items():
            if chunk_id not in texts:
                continue
            terms = tuple(concept.matcher.found(texts[chunk_id][1]))
            if terms and sim >= min_similarity:
                picked.append((chunk_id, sim, terms, "оба"))
            elif sim >= min_similarity:
                picked.append((chunk_id, sim, terms, "вектор"))
            elif terms and sim >= min_similarity_terms:
                picked.append((chunk_id, sim, terms, "словарь"))
        picked.sort(key=lambda p: (p[3] == "оба", p[1]), reverse=True)
        picked = picked[:max_per_concept]

        for chunk_id, sim, terms, found_by in picked:
            doc_id, text = texts[chunk_id]
            sentences = split_sentences(text)
            quote = quote_by_terms(sentences, concept.matcher) if terms else None
            if quote is None:
                need_sentence_quote.append((len(candidates), sentences, ci))
                quote = _cut(sentences[0]) if sentences else _cut(" ".join(text.split()))
            candidates.append(Candidate(concept.code, chunk_id, doc_id, quote,
                                        round(sim, 4), terms, found_by))

        by = {k: sum(1 for p in picked if p[3] == k) for k in ("оба", "вектор", "словарь")}
        report[concept.code] = {"total": len(picked), **by,
                                "best": round(max((p[1] for p in picked), default=0), 3)}
        log(f"  {concept.code:<40} {len(picked):>4}  "
            f"(оба {by['оба']}, вектор {by['вектор']}, словарь {by['словарь']})")

    # Цитаты для найденного только по смыслу: предложение, ближайшее к определению.
    # Предложения переводятся в векторы пачками: всё разом — это тысячи векторов
    # по 1024 числа, сотни мегабайт памяти ради одной строки на фрагмент.
    if need_sentence_quote:
        total = sum(min(len(s), 20) for _, s, _ in need_sentence_quote)
        log(f"  подбираю цитаты: {total} предложений…")
        batch: list[tuple[int, list[str], int]] = []
        size = 0
        for item in need_sentence_quote + [None]:
            if item is not None:
                batch.append(item)
                size += min(len(item[1]), 20)
            if batch and (item is None or size >= 512):
                _best_sentences(embedder, vectors, candidates, batch)
                batch, size = [], 0
    return candidates, report


def _best_sentences(embedder, concept_vectors, candidates: list[Candidate],
                    batch: list[tuple[int, list[str], int]]) -> None:
    flat = [s for _, sentences, _ in batch for s in sentences[:20]]
    if not flat:
        return
    sent_vectors = embedder.encode(flat)
    pos = 0
    for index, sentences, ci in batch:
        n = min(len(sentences), 20)
        if n:
            scores = [_cosine(v, concept_vectors[ci]) for v in sent_vectors[pos:pos + n]]
            best = sentences[max(range(n), key=scores.__getitem__)]
            old = candidates[index]
            candidates[index] = Candidate(old.concept, old.chunk_id, old.doc_id, _cut(best),
                                          old.similarity, old.terms, old.found_by)
        pos += n


# --------------------------------------------------------------------------- #
#  Шаг 2: проверка моделью
# --------------------------------------------------------------------------- #
VERIFY_SYSTEM = (
    "Ты проверяешь разметку геологических статей. Отвечай только JSON, без пояснений. "
    "Опирайся только на текст фрагмента, ничего не добавляй по своим знаниям."
)

VERIFY_USER = """Понятие: «{code}».
Определение: {definition}

Фрагмент статьи:
\"\"\"{text}\"\"\"

Описывает ли фрагмент это понятие по существу — говорит о самом явлении,
а не упоминает слово мимоходом, в перечне или в списке литературы?

Если да — выпиши из фрагмента одну цитату, которая это показывает: одно-два
предложения, дословно, без пересказа.
Если нет — оставь цитату пустой.

Верни JSON: {{"описывает": true, "цитата": "...", "причина": "не больше 15 слов"}}"""

RETRY_NOTE = ("\n\nПрошлая цитата не найдена во фрагменте дословно. "
              "Скопируй предложение из текста выше точно, символ в символ.")


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "да", "yes", "1"}


def judge(payload: dict, text: str) -> tuple[str, str | None, str]:
    """Ответ модели → (решение, цитата в написании статьи, причина).

    «Описывает», но цитаты в тексте нет — это не подтверждение: решение
    модели должно опираться на то, что можно открыть и прочитать.
    """
    reason = str(payload.get("причина") or "").strip()[:200]
    if not _truthy(payload.get("описывает")):
        return store.REJECTED, None, reason or "модель: не описывает"
    found = locate_quote(str(payload.get("цитата") or ""), text)
    if found is None:
        return store.UNCHECKED, None, "цитаты нет во фрагменте"
    return store.CONFIRMED, _cut(found, 800), reason


def verify(conn, llm, vocab: Vocabulary, limit: int | None = None, log=print,
           stop_after_errors: int = 3) -> dict:
    """Проверить моделью непроверенных кандидатов. Можно прервать и продолжить."""
    rows = store.unchecked(conn, limit)
    stats = {"checked": 0, "confirmed": 0, "rejected": 0, "no_quote": 0,
             "errors": 0, "left": len(rows), "stopped": False}
    if not rows:
        return stats

    model = getattr(llm, "model", "модель")
    errors_in_row = 0
    started = time.monotonic()
    for i, (evidence_id, code, text) in enumerate(rows, start=1):
        concept = vocab.concepts.get(code)
        if concept is None:
            continue
        prompt = VERIFY_USER.format(code=concept.code, definition=concept.definition,
                                    text=(text or "")[:4000])
        try:
            verdict, quote, reason = judge(llm.chat_json(VERIFY_SYSTEM, prompt), text or "")
            if verdict == store.UNCHECKED:            # цитата не нашлась — переспросить раз
                verdict, quote, reason = judge(
                    llm.chat_json(VERIFY_SYSTEM, prompt + RETRY_NOTE), text or "")
                if verdict == store.UNCHECKED:
                    verdict, stats["no_quote"] = store.REJECTED, stats["no_quote"] + 1
                    reason = "дважды привела цитату, которой нет во фрагменте"
            errors_in_row = 0
        except Exception as exc:  # noqa: BLE001 — модель молчит: решение не принимаем
            stats["errors"] += 1
            errors_in_row += 1
            log(f"  [{i}/{len(rows)}] модель не ответила: {type(exc).__name__}: {exc}")
            if errors_in_row >= stop_after_errors:
                stats["stopped"] = True
                log("  Модель молчит несколько раз подряд — останавливаюсь. "
                    "Проверьте Ollama и запустите снова: продолжится с этого места.")
                break
            continue

        store.set_verdict(conn, evidence_id, verdict, quote, model, reason)
        conn.commit()                      # после каждого: прерывание ничего не теряет
        stats["checked"] += 1
        stats["confirmed" if verdict == store.CONFIRMED else "rejected"] += 1
        if i % 10 == 0 or i == len(rows):
            rate = (time.monotonic() - started) / i
            log(f"  проверено {i}/{len(rows)}: подтверждено {stats['confirmed']}, "
                f"отклонено {stats['rejected']} · {rate:.1f} с на фрагмент")
    stats["left"] = len(rows) - stats["checked"]
    return stats
