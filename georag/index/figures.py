"""Рисунки статей: картинки из PDF, таблица рисунков, ссылки «рис. N» на них из текста —
и рисунки под ответом чат-бота.

Откуда: разметка docling статьи (<doc_id>.docling.json) знает у каждого рисунка
страницу, рамку и подпись, а саму картинку не хранит (разбор идёт без
generate_picture_images — так быстрее). Картинка вырезается из PDF статьи по рамке
(PyMuPDF): файл .local/acquired/<doc_id>.figures/fig-NNN.jpg, перечень рисунков —
номер в статье по подписи («Рис. 3» → label «3»), страница, рамка, подпись, размер — в
поле figures записи статьи. Берутся только рисунки с подписью: без подписи — обычно
логотип журнала, значок или обложка, и сослаться на них текст не может.

При загрузке статьи (index/ingest.py):

1. в таблице figures — каждый рисунок и фрагмент, где стоит его подпись;
2. в figure_refs — какие фрагменты на него ссылаются: «рис. 3», «на рисунке 2а»,
   «(рис. 5, 6 и 7)», «Fig. 4», «figs. 2 and 3», а также фрагмент с подписью.

Под ответом — рисунки фрагментов, на которые ответ сослался (for_answer): страница
показывает их картинками (событие figures, адрес /api/figure), Телеграм — фото после
ответа.

    python georag.py figures [--pdf ПАПКА] [doc_id …] [--redo]    вырезать и записать в базу

Перенесено из AgentKB(GEO) (kbrag/chunks/figures.py, kbrag/index/figures.py) — без
описания рисунков моделью со зрением и поиска рисунков по смыслу.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from ..paths import ARTICLES
from . import db

FIGURES_DIR = "{doc_id}.figures"  # папка рисунков статьи — рядом с её файлами
DPI = 144  # мелкие подписи на карте читаются
MAX_SIDE = 1600  # пикселей по большей стороне: больше на странице не нужно
MIN_SIDE = 40  # пунктов (около 1,4 см) по меньшей стороне: меньше — значок или линия
JPEG_QUALITY = 85
FIGURES_IN_ANSWER = 4  # рисунков под ответом: больше — не смотрят

# Номер рисунка в самой статье — из подписи: «Рис. 3.», «Рисунок 2а», «Fig. 4:», «Figure 1».
_LABEL = re.compile(r"^\s*(?:рис(?:унок)?|fig(?:ure)?)\.?\s*(\d{1,3}[а-яa-z]?)\b", re.IGNORECASE)
_NUMBER = r"\d{1,3}[а-яa-z]?"
# «рис. 3», «рисунке 2а», «рис. 2, 3 и 5», «Fig. 4», «figs. 2 and 3», «Figure 1b».
_REF = re.compile(
    rf"(?<![\w])(?:рис(?:ун\w*)?\.?|figs?\.|figures?)\s*({_NUMBER}"
    rf"(?:\s*(?:,|и|and|&)\s*{_NUMBER})*)(?![\w])",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
#  Рисунки статьи из разметки docling и картинки из PDF
# --------------------------------------------------------------------------- #
def label_of(caption: str) -> str | None:
    """Номер рисунка в статье по его подписи («3», «2а»); нет — None."""
    found = _LABEL.match(caption or "")
    return found.group(1).lower() if found else None


def _ref_text(doc: dict[str, Any], ref: dict[str, Any]) -> str:
    """Текст по ссылке docling вида {"$ref": "#/texts/12"}."""
    path = str(ref.get("$ref", "")).lstrip("#/").split("/")
    if len(path) != 2 or not path[1].isdigit():
        return ""
    items = doc.get(path[0]) or []
    index = int(path[1])
    return str(items[index].get("text", "")).strip() if index < len(items) else ""


def pictures(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Рисунки с подписью из DoclingDocument по порядку: страница, рамка (в пунктах, начало
    координат — как в разметке), размер страницы в разметке, подпись и номер в статье.
    Мелкие (значки, линии) — мимо."""
    out = []
    for pic in doc.get("pictures") or []:
        caption = " ".join(
            " ".join(_ref_text(doc, c) for c in pic.get("captions") or []).split()
        ).replace("\x00", "")
        prov = (pic.get("prov") or [{}])[0]
        box, page = prov.get("bbox") or {}, prov.get("page_no")
        if not caption or not isinstance(page, int) or not box:
            continue
        width, height = abs(box["r"] - box["l"]), abs(box["t"] - box["b"])
        if min(width, height) < MIN_SIDE:
            continue
        size = ((doc.get("pages") or {}).get(str(page)) or {}).get("size") or {}
        out.append(
            {
                "page": page,
                "bbox": box,
                "page_size": [size.get("width"), size.get("height")],
                "caption": caption,
                "label": label_of(caption),
            }
        )
    return out


def _clip(item: dict[str, Any], page: Any) -> Any:
    """Рамка рисунка в координатах страницы PyMuPDF (начало — левый верхний угол).
    Размер страницы в разметке может не совпасть с PDF — рамка масштабируется."""
    import pymupdf

    box = item["bbox"]
    width, height = item.get("page_size") or [None, None]
    sx = page.rect.width / width if width else 1.0
    sy = page.rect.height / height if height else 1.0
    if str(box.get("coord_origin", "")).upper() == "BOTTOMLEFT":
        full = height or page.rect.height / sy
        top, bottom = full - box["t"], full - box["b"]
    else:
        top, bottom = box["t"], box["b"]
    corners = (box["l"] * sx, min(top, bottom) * sy, box["r"] * sx, max(top, bottom) * sy)
    return pymupdf.Rect(*corners) & page.rect  # type: ignore[no-untyped-call]


def save(
    doc: dict[str, Any], pdf: Path | bytes, doc_id: str, articles: Path = ARTICLES
) -> list[dict[str, Any]]:
    """Рисунки статьи — картинками из PDF в папку <doc_id>.figures; возвращает перечень
    для записи статьи (поле figures). Прежние файлы рисунков статьи стираются: номера идут
    заново. PDF не открылся — пусто."""
    import pymupdf

    folder = articles / FIGURES_DIR.format(doc_id=doc_id)
    if folder.is_dir():
        for old in folder.glob("fig-*.*"):
            old.unlink()
    items = pictures(doc)
    if not items:
        return []
    try:
        if isinstance(pdf, Path):
            opened = pymupdf.open(str(pdf))  # type: ignore[no-untyped-call]
        else:
            opened = pymupdf.open(stream=pdf, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception:  # noqa: BLE001 — битый PDF: статья без картинок, но не без текста
        return []
    saved: list[dict[str, Any]] = []
    with opened:
        for item in items:
            if not 1 <= item["page"] <= opened.page_count:
                continue
            page = opened[item["page"] - 1]
            rect = _clip(item, page)
            if rect.is_empty or min(rect.width, rect.height) < MIN_SIDE:
                continue
            dpi = min(DPI, int(MAX_SIDE * 72 / max(rect.width, rect.height)))
            try:
                pix = page.get_pixmap(clip=rect, dpi=dpi)
                data = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            except Exception:  # noqa: BLE001 — не вырезался один рисунок: остальные идут
                continue
            number = len(saved) + 1
            name = f"fig-{number:03d}.jpg"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
            saved.append(
                {
                    "n": number,
                    "label": item["label"],
                    "file": f"{folder.name}/{name}",
                    "page": item["page"],
                    "bbox": item["bbox"],
                    "caption": item["caption"],
                    "width": pix.width,
                    "height": pix.height,
                }
            )
    return saved


def file_of(item: dict[str, Any], articles: Path = ARTICLES) -> Path | None:
    """Файл картинки рисунка; нет файла — None."""
    name = str(item.get("file") or "")
    path = (articles / name).resolve()
    # Только из папки статей: имя файла приходит из базы, а отдаётся оно по адресу.
    if not name or articles.resolve() not in path.parents or not path.is_file():
        return None
    return path


# --------------------------------------------------------------------------- #
#  В базе: рисунки и ссылки на них
# --------------------------------------------------------------------------- #
def references(text: str, labels: set[str]) -> set[str]:
    """Номера рисунков статьи, на которые ссылается текст."""
    found: set[str] = set()
    for match in _REF.finditer(text or ""):
        found |= {n.lower() for n in re.findall(_NUMBER, match.group(1))}
    return found & labels


def store(conn: Any, doc_id: str, items: list[dict[str, Any]]) -> int:
    """Рисунки статьи — заново (прежние уходят вместе со ссылками), и ссылки на них из её
    фрагментов. Сколько рисунков записано."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM figures WHERE doc_id = %s", (doc_id,))
        items = [i for i in items or [] if isinstance(i, dict) and i.get("n")]
        if not items:
            return 0
        cur.execute("SELECT id, text FROM chunks WHERE doc_id = %s ORDER BY ord", (doc_id,))
        chunk_rows = [(int(r[0]), " ".join(str(r[1] or "").split())) for r in cur.fetchall()]
        ids: dict[str, int] = {}
        for item in items:
            caption = " ".join(str(item.get("caption") or "").split())
            captioned = (
                next((c for c, text in chunk_rows if caption[:80] in text), None)
                if len(caption) >= 12
                else None
            )
            cur.execute(
                """
                INSERT INTO figures (doc_id, n, label, file, page, bbox, caption, width,
                                     height, chunk_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    doc_id,
                    int(item["n"]),
                    item.get("label"),
                    item.get("file") or "",
                    item.get("page"),
                    json.dumps(item["bbox"]) if item.get("bbox") else None,
                    caption,
                    item.get("width"),
                    item.get("height"),
                    captioned,
                ),
            )
            row = cur.fetchone()
            if row is None:
                continue
            if item.get("label"):
                ids.setdefault(str(item["label"]), int(row[0]))
            if captioned is not None:
                _ref(cur, captioned, int(row[0]))
        labels = set(ids)
        for chunk_id, text in chunk_rows:
            for label in references(text, labels):
                _ref(cur, chunk_id, ids[label])
    return len(items)


def _ref(cur: Any, chunk_id: int, figure_id: int) -> None:
    cur.execute(
        "INSERT INTO figure_refs (chunk_id, figure_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (chunk_id, figure_id),
    )


_COLUMNS = "id doc_id n label file page bbox caption width height chunk_id"
_KEYS = (*_COLUMNS.split(), "title", "year")
FIELDS = ", ".join(f"g.{c}" for c in _COLUMNS.split()) + ", d.title, d.year"


def _row(r: tuple[Any, ...]) -> dict[str, Any]:
    out = dict(zip(_KEYS, r[: len(_KEYS)], strict=True))
    if isinstance(out.get("bbox"), str):
        out["bbox"] = json.loads(out["bbox"])
    return out


def by_id(conn: Any, figure_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {FIELDS} FROM figures g JOIN documents d ON d.doc_id = g.doc_id "
            "WHERE g.id = %s",
            (figure_id,),
        )
        row = cur.fetchone()
    return _row(row) if row else None


def in_chunks(conn: Any, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Рисунки фрагментов: тот, чья подпись во фрагменте, и те, на которые фрагмент
    ссылается. Фрагмент → его рисунки по порядку."""
    ids = [int(c) for c in chunk_ids if c]
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {FIELDS}, r.chunk_id
            FROM figure_refs r
            JOIN figures g ON g.id = r.figure_id
            JOIN documents d ON d.doc_id = g.doc_id
            WHERE r.chunk_id = ANY(%s)
            ORDER BY r.chunk_id, g.n
            """,
            (ids,),
        )
        out: dict[int, list[dict[str, Any]]] = {}
        for r in cur.fetchall():
            out.setdefault(int(r[-1]), []).append(_row(r))
    return out


def for_answer(
    conn: Any,
    sources: list[dict[str, Any]],
    used: list[int] | None,
    limit: int = FIGURES_IN_ANSWER,
    question: str = "",
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """Рисунки фрагментов, на которые сослался ответ ([n] из used; не сослался ни на один
    — ни одного), у каждого — n фрагмента ответа (source).

    Фрагмент ссылается и на рисунки о другом: абзац о магматизме — на карту магнитного
    поля. Поэтому (замер 22.10.2026, 18 вопросов, 30 рисунков-кандидатов):

    1. сначала рисунки, чья подпись стоит в самом фрагменте, по порядку ссылок, потом те,
       на которые фрагмент только ссылается;
    2. из одной статьи — не больше PER_DOC: одна статья занимала все 4 места;
    3. с вопросом и embedder — только рисунки, чья подпись близка к вопросу
       (MIN_SIMILARITY): подходящие — от 0,44 (схема порфировых месторождений 0,61,
       карта ореолов золота 0,56), мимо — ниже (магнитные карты в вопросе о порфировой
       меди 0,29–0,43, возраст по цирконам в вопросе о РЗЭ 0,33). Векторы не
       посчитались — без этого шага."""
    chosen = set(used or [])
    cited = [s for s in sources if s.get("chunk_id") and s.get("n") in chosen]
    if not cited:
        return []
    by_chunk = in_chunks(conn, [int(s["chunk_id"]) for s in cited])
    own: list[dict[str, Any]] = []
    referenced: list[dict[str, Any]] = []
    for src in cited:
        for item in by_chunk.get(int(src["chunk_id"]), []):
            mine = item.get("chunk_id") == src["chunk_id"]
            (own if mine else referenced).append({**item, "source": src["n"]})
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    per_doc: dict[str, int] = {}
    for item in own + referenced:
        if item["id"] in seen or per_doc.get(item["doc_id"], 0) >= PER_DOC:
            continue
        seen.add(item["id"])
        per_doc[item["doc_id"]] = per_doc.get(item["doc_id"], 0) + 1
        candidates.append(item)
    if question and embedder is not None and candidates:
        candidates = _close_to(question, candidates, embedder)
    return candidates[:limit]


PER_DOC = 2  # рисунков одной статьи под ответом
MIN_SIMILARITY = 0.44  # близость подписи к вопросу (BGE-M3, косинус) — см. for_answer


def _close_to(question: str, items: list[dict[str, Any]], embedder: Any) -> list[dict[str, Any]]:
    """Рисунки, чья подпись близка к вопросу, — в прежнем порядке; у каждого — similarity."""
    try:
        vectors = embedder.encode([question] + [str(i.get("caption") or "") for i in items])
    except Exception:  # noqa: BLE001 — не посчиталось: рисунки без отбора
        return items
    asked = vectors[0]
    out = []
    for item, vector in zip(items, vectors[1:], strict=True):
        similarity = sum(a * b for a, b in zip(asked, vector, strict=True))
        if similarity >= MIN_SIMILARITY:
            out.append({**item, "similarity": round(float(similarity), 3)})
    return out


def caption_line(item: dict[str, Any]) -> str:
    """«Рис. 3 к [2], с. 9. <подпись>» — под картинкой на странице и в Телеграме."""
    name = f"Рис. {item['label']}" if item.get("label") else f"Рисунок {item.get('n')}"
    source = f" к [{item['source']}]" if item.get("source") else ""
    page = f", с. {item['page']}" if item.get("page") else ""
    return f"{name}{source}{page}. {item.get('caption') or ''}".strip()


# --------------------------------------------------------------------------- #
#  Команда: python georag.py figures — вырезать рисунки уже разобранных статей
# --------------------------------------------------------------------------- #
def find_pdf(doc_id: str, meta: dict[str, Any], folders: dict[str, Path]) -> Path | None:
    """PDF статьи: путь из записи, если файл есть, иначе — файл <doc_id>.pdf в папках --pdf."""
    for key in ("source_path", "path"):
        value = str(meta.get(key) or "")
        if value.lower().endswith(".pdf") and Path(value).is_file():
            return Path(value)
    return folders.get(doc_id)


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from ..common import add_db_args

    parser = argparse.ArgumentParser(
        prog="python georag.py figures",
        description="Рисунки статей: вырезать из PDF и записать в базу",
    )
    parser.add_argument("doc_ids", nargs="*", help="статьи (doc_id); без них — все разобранные")
    parser.add_argument(
        "--pdf", action="append", default=[], type=Path, help="папка с PDF статей (<doc_id>.pdf)"
    )
    parser.add_argument("--redo", action="store_true", help="вырезать заново и уже вырезанные")
    parser.add_argument("--articles", type=Path, default=ARTICLES)
    add_db_args(parser)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    folders = {p.stem: p for folder in args.pdf for p in sorted(Path(folder).rglob("*.pdf"))}
    ids = args.doc_ids or sorted(
        p.name[: -len(".docling.json")] for p in args.articles.glob("*.docling.json")
    )
    cut = kept = no_pdf = total = stored = 0
    with db.connect(args.dsn) as conn:
        db.init_db(conn, with_hnsw=False)
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents")
            in_base = {r[0] for r in cur.fetchall()}
        for i, doc_id in enumerate(ids, start=1):
            meta_path = args.articles / f"{doc_id}.json"
            marked = args.articles / f"{doc_id}.docling.json"
            if not meta_path.is_file() or not marked.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "figures" in meta and not args.redo:
                kept += 1
            else:
                pdf = find_pdf(doc_id, meta, folders)
                if pdf is None:
                    no_pdf += 1
                    continue
                doc = json.loads(marked.read_text(encoding="utf-8"))
                meta["figures"] = save(doc, pdf, doc_id, args.articles)
                _write_meta(meta_path, meta)
                cut += 1
            total += len(meta["figures"])
            if doc_id in in_base:
                stored += store(conn, doc_id, meta["figures"])
                conn.commit()
            if i % 100 == 0:
                print(f"  [{i}/{len(ids)}] рисунков {total}", flush=True)
    print(
        f"Статей: вырезано {cut}, уже были {kept}, без PDF {no_pdf}; рисунков {total}, "
        f"в базе {stored}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
