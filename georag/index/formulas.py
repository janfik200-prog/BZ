"""Формулы в базе: пометка во фрагментах, формула словами, таблица формул и ссылки.

Откуда: разметка docling статьи (<doc_id>.docling.json в .local/acquired) — LaTeX там
уже чистый и с номером «\\quad (N)» (parse/orchestrator/formulas.py). При загрузке
статьи (index/ingest.py):

1. формулы во фрагментах помечаются $$…$$ — страница рисует их (KaTeX), модель ответа
   видит, где формула, и цитирует её дословно с номером (правило 8, chat/messages.py);
2. в текст для вектора фрагмента вместо голого LaTeX идёт формула словами — знаки,
   фраза перед ней, «где ρ — …» (describe): запрос по смыслу её находит;
3. в таблице formulas — каждая формула: номер, LaTeX, текст из PDF, страница, рамка,
   абзацы до и после, фрагмент, где она стоит, описание и его вектор;
4. в formula_refs — какие фрагменты на неё ссылаются: «из формулы (55)», «уравнение
   3», «(2)» (если формула с таким номером в статье есть).

Формулу картинкой (Телеграм) рисует matplotlib по LaTeX: PDF статей не хранятся.

    python georag.py formula <doc_id> [номер]     формулы статьи: LaTeX, словами, ссылки
    python georag.py formula --find "…"          формулы по смыслу

Перенесено из AgentKB (kbrag/library/objects.py, kbrag/index/formulas.py) — без
деревьев баз и прав доступа.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

from ..paths import ARTICLES
from . import db

CONTEXT_CHARS = 500  # текста до и после формулы
DESCRIPTION_CHARS = 700
REF_TEXT = 300  # знаков текста ссылающегося фрагмента — в выдаче команды
_NUMBER = re.compile(r"\\quad\s*\((\d{1,3})\)\s*$")  # номер, который дописала чистка
_ORIG_NUMBER = re.compile(r"(?<![\w.])\((\d{1,3})\)\s*$")  # номер в конце текста из PDF
TEXT_LABELS = {"text", "paragraph", "list_item", "caption", "section_header", "title"}


# --------------------------------------------------------------------------- #
#  Формулы статьи из разметки docling
# --------------------------------------------------------------------------- #
def _page(item: dict[str, Any]) -> int | None:
    prov = item.get("prov") or []
    page = prov[0].get("page_no") if prov else None
    return int(page) if isinstance(page, int) else None


def _text(value: Any) -> str:
    """Текст из разметки без нулевого символа из PDF: Postgres не принимает его в тексте,
    и запись формул роняла загрузку всей статьи (как _clean в ingest.py у фрагментов)."""
    return str(value or "").replace("\x00", "")


def _near(texts: list[dict[str, Any]], start: int, step: int) -> str:
    """Текст соседних абзацев (не формул) — до CONTEXT_CHARS, в сторону step."""
    out: list[str] = []
    i = start + step
    while 0 <= i < len(texts) and sum(len(t) for t in out) < CONTEXT_CHARS:
        item = texts[i]
        if item.get("label") in TEXT_LABELS and item.get("text"):
            out.append(" ".join(_text(item["text"]).split()))
        i += step
    joined = " ".join(out if step > 0 else reversed(out))
    return joined[:CONTEXT_CHARS] if step > 0 else joined[-CONTEXT_CHARS:]


def extract(doc_id: str, articles: Path = ARTICLES) -> list[dict[str, Any]]:
    """Формулы статьи по порядку: номер «(N)», LaTeX, текст из PDF, страница, рамка и
    абзацы до и после. Разметки нет или формулы в ней без текста (статья разобрана до
    распознавания формул, 13.10.2026) — пусто."""
    path = articles / f"{doc_id}.docling.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    texts = [t for t in doc.get("texts") or [] if isinstance(t, dict)]
    out: list[dict[str, Any]] = []
    for i, item in enumerate(texts):
        if item.get("label") != "formula" or not item.get("text"):
            continue
        latex = _text(item["text"]).strip()
        orig = " ".join(_text(item.get("orig")).split())
        number = _NUMBER.search(latex) or _ORIG_NUMBER.search(orig)
        prov = (item.get("prov") or [{}])[0]
        out.append(
            {
                "index": len(out) + 1,
                "number": number.group(1) if number else None,
                "latex": latex,
                "pdf_text": orig,
                "page": _page(item),
                "bbox": prov.get("bbox"),
                "before": _near(texts, i, -1),
                "after": _near(texts, i, +1),
            }
        )
    return out


# --------------------------------------------------------------------------- #
#  Формула словами — для векторов (поиск по смыслу) и для людей
# --------------------------------------------------------------------------- #
# Команды LaTeX → знаки: BGE-M3 понимает «ρ = ρ0 (1 − φ)», а «\rho = \rho_0 (1 - \varphi)»
# для неё — набор служебных слов.
SYMBOLS = {
    **{
        f"\\{name}": char
        for name, char in zip(
            "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
            "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi "
            "psi omega".split(),
            "αβγδεεζηθϑικλμνξπϖρϱσςτυφφχψω",
            strict=True,
        )
    },
    **{
        f"\\{name}": char
        for name, char in zip(
            "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split(),
            "ΓΔΘΛΞΠΣΥΦΨΩ",
            strict=True,
        )
    },
    "\\cdot": "·",
    "\\times": "×",
    "\\pm": "±",
    "\\mp": "∓",
    "\\leq": "≤",
    "\\le": "≤",
    "\\geq": "≥",
    "\\ge": "≥",
    "\\neq": "≠",
    "\\ne": "≠",
    "\\approx": "≈",
    "\\sim": "∼",
    "\\propto": "∝",
    "\\infty": "∞",
    "\\sum": "Σ",
    "\\prod": "Π",
    "\\int": "∫",
    "\\oint": "∮",
    "\\partial": "∂",
    "\\nabla": "∇",
    "\\to": "→",
    "\\rightarrow": "→",
    "\\leftarrow": "←",
    "\\in": "∈",
    "\\degree": "°",
    "\\circ": "°",
    "\\%": "%",
    "\\ln": "ln",
    "\\log": "log",
    "\\exp": "exp",
    "\\sin": "sin",
    "\\cos": "cos",
    "\\tan": "tan",
    "\\max": "max",
    "\\min": "min",
    "\\lim": "lim",
}
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_WRAP = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|text|textrm|operatorname|boldsymbol)\s*\{([^{}]*)\}"
)
_COMMAND = re.compile(r"\\[A-Za-z]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Абзац после формулы, который объясняет её обозначения: «где ρ — плотность…».
_DEFINES = re.compile(r"^(?:где|здесь|в которой|в котором|причём|where|here|with)\b", re.IGNORECASE)


def plain_latex(latex: str) -> str:
    """LaTeX → строка знаками: \\frac{a}{b} → (a)/(b), \\rho → ρ, ^{2} → ^2. Не точная
    отрисовка, а то, что читается моделью векторов и человеком в описании."""
    text = re.sub(r"\\quad\s*\(\d{1,3}\)\s*$", "", latex or "")  # номер — отдельно
    # \left и \right — только сами по себе: \leftarrow и \rightarrow — стрелки (SYMBOLS).
    text = re.sub(r"\\(?:left|right)(?![A-Za-z])|\\[,;:!]|\\quad|\\qquad|~", " ", text)
    for _ in range(4):  # изнутри наружу: индексы и степени, потом дроби и корни вокруг них
        text = _WRAP.sub(r"\1", text)
        text = re.sub(r"\^\{([^{}]*)\}", r"^\1", text)
        text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
        text = _FRAC.sub(r"(\1)/(\2)", text)
        text = _SQRT.sub(r"√(\1)", text)
    for command in sorted(SYMBOLS, key=len, reverse=True):
        text = re.sub(re.escape(command) + r"(?![A-Za-z])", SYMBOLS[command], text)
    text = _COMMAND.sub(" ", text)
    text = text.replace("{", "").replace("}", "").replace("-", "−")
    return " ".join(text.split())


def describe(item: dict[str, Any]) -> str:
    """Формула словами: номер, она сама знаками, фраза перед ней и объяснение
    обозначений после («где …»). Это идёт в вектор вместо голого LaTeX — так формулу
    находит запрос по смыслу («как плотность зависит от пористости»)."""
    number = f" ({item['number']})" if item.get("number") else ""
    parts = [f"Формула{number}: {plain_latex(item.get('latex', ''))}."]
    before = _SENTENCE_END.split(str(item.get("before") or "").strip())
    if before and before[-1]:
        lead = before[-1].rstrip(":").strip()  # законченная фраза — без второй точки
        parts.append(lead if lead.endswith((".", "!", "?", "…")) else lead + ".")
    after = str(item.get("after") or "").strip()
    if _DEFINES.match(after):
        parts.append(" ".join(_SENTENCE_END.split(after)[:2]))
    return " ".join(" ".join(parts).split())[:DESCRIPTION_CHARS]


# Ссылка на формулу в тексте: «формула (55)», «уравнение 3», «eq. (2)», «Eqs. 4» — или
# просто «(55)», если формула с таким номером в статье есть («(2015)» номером не бывает:
# до 3 цифр). Голое «(2)» бывает и номером пункта — такая ссылка может быть ложной.
_NAMED_REF = re.compile(
    r"(?:формул\w*|уравнени\w*|выражени\w*|соотношени\w*|равенств\w*|"
    r"eqs?\.?|equations?|formulas?|formulae)\s*\(?\s*(\d{1,3})\s*\)?",
    re.IGNORECASE,
)
_BARE_REF = re.compile(r"(?<![\w.(])\((\d{1,3})\)(?![\w(])")


def references(text: str, numbers: set[str]) -> set[str]:
    """Номера формул статьи, на которые ссылается текст. Сами формулы ($$…$$) не в счёт:
    их собственный номер «\\quad (N)» — не ссылка."""
    text = re.sub(r"\$\$[\s\S]*?\$\$", " ", text or "")
    found = {m.group(1) for m in _NAMED_REF.finditer(text)}
    found |= {m.group(1) for m in _BARE_REF.finditer(text)}
    return found & numbers


def latex_pattern(latex: str) -> re.Pattern[str] | None:
    """Где LaTeX формулы стоит в тексте фрагмента: те же знаки, пробелы — любые."""
    tokens = (latex or "").split()
    if not tokens:
        return None
    return re.compile(r"\${0,2}" + r"\s*".join(re.escape(t) for t in tokens) + r"\${0,2}")


def mark_formulas(
    text: str, items: list[dict[str, Any]], replace: Any = None
) -> tuple[str, list[int]]:
    """Формулы во фрагменте — в $$…$$ (по ним рисует страница и их видит модель ответа),
    или replace(формула) вместо них (описание — для вектора). Какие формулы нашлись —
    их index в статье."""
    found: list[int] = []
    for item in items:
        pattern = latex_pattern(item.get("latex", ""))
        if pattern is None:
            continue
        new: str = replace(item) if replace else f"$${item['latex']}$$"
        # Строка замены — функцией: в LaTeX обратные косые, а в строке замены re их читает.
        text, count = pattern.subn(lambda _m: new, text, count=1)  # noqa: B023 — сразу
        if count:
            found.append(int(item["index"]))
    return text, found


def render_png(latex: str) -> bytes | None:
    """Формула картинкой — LaTeX, нарисованный mathtext matplotlib (подмножество LaTeX).
    Нет matplotlib или формула ему не по силам — None (тогда показывается сам LaTeX)."""
    clean = re.sub(r"\\quad\s*\((\d{1,3})\)\s*$", r"\\quad (\1)", latex or "").strip()
    if not clean:
        return None
    try:
        from matplotlib import mathtext

        buf = io.BytesIO()
        mathtext.math_to_image(f"${clean}$", buf, dpi=200, format="png")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 — нет matplotlib или не разобрал формулу
        return None


# --------------------------------------------------------------------------- #
#  Загрузка статьи (index/ingest.py)
# --------------------------------------------------------------------------- #
def prepare(doc_id: str, articles: Path = ARTICLES) -> list[dict[str, Any]]:
    """Формулы статьи для загрузки — с описанием словами."""
    items = extract(doc_id, articles)
    for item in items:
        item["description"] = describe(item)
    return items


def mark_chunks(chunks: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Формулы во фрагментах: в тексте — $$…$$, в тексте для вектора — описание словами.
    Возвращает: номер фрагмента (index) → index формул, которые в нём стоят."""
    where: dict[int, list[int]] = {}
    if not items:
        return where
    for chunk in chunks:
        text, found = mark_formulas(str(chunk.get("text") or ""), items)
        if not found:
            continue
        chunk["text"] = text
        embed = str(chunk.get("embed_text") or chunk.get("text") or "")
        chunk["embed_text"], _ = mark_formulas(
            embed, items, replace=lambda item: f"[{item['description']}]"
        )
        where[int(chunk.get("index", 0))] = found
    return where


def store(
    conn: Any,
    doc_id: str,
    items: list[dict[str, Any]],
    where: dict[int, list[int]],
    vectors: list[list[float]],
) -> int:
    """Формулы статьи — заново (прежние уходят вместе со ссылками), и ссылки на них из
    её фрагментов. Сколько формул записано."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM formulas WHERE doc_id = %s", (doc_id,))
        if not items:
            return 0
        cur.execute("SELECT ord, id, text FROM chunks WHERE doc_id = %s", (doc_id,))
        chunk_rows = cur.fetchall()
        chunk_by_ord = {int(r[0]): int(r[1]) for r in chunk_rows}
        chunk_of = {
            index: chunk_by_ord[ord_]
            for ord_, indexes in where.items()
            if ord_ in chunk_by_ord
            for index in indexes
        }
        ids: dict[str, int] = {}
        for item, vector in zip(items, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO formulas (doc_id, idx, number, latex, pdf_text, page, bbox,
                                      before, after, description, chunk_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                RETURNING id
                """,
                (
                    doc_id,
                    item["index"],
                    item.get("number"),
                    item["latex"],
                    item.get("pdf_text") or "",
                    item.get("page"),
                    json.dumps(item["bbox"]) if item.get("bbox") else None,
                    item.get("before") or "",
                    item.get("after") or "",
                    item["description"],
                    chunk_of.get(int(item["index"])),
                    db.vector_literal(vector),
                ),
            )
            row = cur.fetchone()
            if row and item.get("number"):
                ids.setdefault(str(item["number"]), int(row[0]))
        numbers = set(ids)
        for _ord, chunk_id, text in chunk_rows:
            for number in references(str(text or ""), numbers):
                cur.execute(
                    "INSERT INTO formula_refs (chunk_id, formula_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (chunk_id, ids[number]),
                )
    return len(items)


# --------------------------------------------------------------------------- #
#  Чтение из базы
# --------------------------------------------------------------------------- #
_COLUMNS = "id doc_id idx number latex pdf_text page bbox before after description chunk_id"
_KEYS = (*_COLUMNS.replace("idx", "index").split(), "title", "year", "url")
FIELDS = ", ".join(f"f.{c}" for c in _COLUMNS.split()) + ", d.title, d.year, d.url"
SOURCE = "formulas f JOIN documents d ON d.doc_id = f.doc_id"


def _row(r: tuple[Any, ...]) -> dict[str, Any]:
    out = dict(zip(_KEYS, r[: len(_KEYS)], strict=True))
    if isinstance(out.get("bbox"), str):
        out["bbox"] = json.loads(out["bbox"])
    return out


def has_rows(conn: Any, doc_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM formulas WHERE doc_id = %s LIMIT 1", (doc_id,))
        return cur.fetchone() is not None


def by_id(conn: Any, formula_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {FIELDS} FROM {SOURCE} WHERE f.id = %s", (formula_id,))
        row = cur.fetchone()
    return _row(row) if row else None


def of_document(conn: Any, doc_id: str, number: str | None = None) -> list[dict[str, Any]]:
    """Формулы статьи по порядку (или одна — с номером «(number)»), у каждой — фрагменты,
    которые на неё ссылаются (referenced_by)."""
    params = [doc_id]
    clause = ""
    if number is not None:
        clause = " AND f.number = %s"
        params.append(str(number).strip().strip("()"))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {FIELDS} FROM {SOURCE} WHERE f.doc_id = %s{clause} ORDER BY f.idx", params
        )
        found = [_row(r) for r in cur.fetchall()]
        for item in found:
            cur.execute(
                "SELECT c.id, c.ord, c.pages, c.text FROM formula_refs r "
                "JOIN chunks c ON c.id = r.chunk_id WHERE r.formula_id = %s ORDER BY c.ord",
                (item["id"],),
            )
            item["referenced_by"] = [
                {"chunk_id": c[0], "ord": c[1], "pages": list(c[2] or []), "text": c[3][:REF_TEXT]}
                for c in cur.fetchall()
            ]
    return found


def in_chunks(conn: Any, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Формулы, стоящие в этих фрагментах: фрагмент → его формулы (кнопки в Телеграме)."""
    ids = [int(c) for c in chunk_ids if c]
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {FIELDS} FROM {SOURCE} WHERE f.chunk_id = ANY(%s) ORDER BY f.chunk_id, f.idx",
            (ids,),
        )
        out: dict[int, list[dict[str, Any]]] = {}
        for r in cur.fetchall():
            item = _row(r)
            out.setdefault(int(item["chunk_id"]), []).append(item)
    return out


def search(conn: Any, embedder: Any, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Формулы по смыслу — по вектору описания (формула словами с контекстом)."""
    vector = db.vector_literal(embedder.encode_one(query))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {FIELDS}, 1 - (f.embedding <=> %(q)s::vector) AS similarity
            FROM {SOURCE}
            WHERE f.embedding IS NOT NULL
            ORDER BY f.embedding <=> %(q)s::vector
            LIMIT %(limit)s
            """,
            {"q": vector, "limit": limit},
        )
        rows = cur.fetchall()
    return [{**_row(r), "similarity": round(float(r[-1]), 4)} for r in rows]


# --------------------------------------------------------------------------- #
#  Команда: python georag.py formula <doc_id> [номер] | formula --find «описание»
# --------------------------------------------------------------------------- #
def show(item: dict[str, Any]) -> str:
    number = f" ({item['number']})" if item.get("number") else ""
    page = f", с. {item['page']}" if item.get("page") else ""
    lines = [
        f"Формула{number} — {item.get('title') or item.get('doc_id')}{page}",
        f"  LaTeX: {item.get('latex')}",
        f"  словами: {item.get('description')}",
    ]
    refs = item.get("referenced_by") or []
    if refs:
        lines.append("  ссылаются фрагменты: " + ", ".join(str(r["ord"]) for r in refs))
    if item.get("similarity") is not None:
        lines.append(f"  близость: {item['similarity']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ..common import add_db_args, add_embedder_args

    parser = argparse.ArgumentParser(prog="python georag.py formula", description="Формулы статей")
    parser.add_argument("doc_id", nargs="?", help="статья (doc_id)")
    parser.add_argument("number", nargs="?", help="номер формулы — как «(N)» в статье")
    parser.add_argument("--find", default="", help="найти формулы по описанию словами")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="одним объектом JSON в stdout")
    add_db_args(parser)
    add_embedder_args(parser)
    args = parser.parse_args(argv)
    if not args.doc_id and not args.find:
        parser.print_usage(sys.stderr)
        return 1
    with db.connect(args.dsn) as conn:
        db.init_db(conn, with_hnsw=False)
        if args.find:
            from .embed import build_embedder

            found = search(conn, build_embedder(device=args.device), args.find, args.limit)
        else:
            found = of_document(conn, args.doc_id, args.number)
    if args.json:
        print(json.dumps({"formulas": found}, ensure_ascii=False, indent=1, default=str))
        return 0 if found else 1
    if not found:
        print(
            "Формул не нашлось. Формулы есть только у статей, разобранных с распознаванием "
            "формул (с 13.10.2026) и загруженных после этого (python georag.py ingest).",
            file=sys.stderr,
        )
        return 1
    for item in found:
        print(show(item) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
