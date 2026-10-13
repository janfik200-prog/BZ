"""Настройки этапа парсинга.

Всё, что можно захотеть покрутить, живёт здесь, а не разбросано по коду.
Пути по умолчанию рассчитаны на структуру:

    .local/pdf/      — исходные PDF
    .local/manual/   — ручные manual.txt (последний fallback), имя = имя PDF + .txt
    .local/parsed/   — результат: <stem>.json, <stem>.md, <stem>.chunks.json
    .local/tests/golden/ — эталоны (golden standard), имя = имя PDF + .json
    .local/logs/     — JSONL-лог по шагам
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ..paths import LOCAL, LOGS


@dataclass
class Settings:
    # --- пути ---
    pdf_dir: Path = LOCAL / "pdf"
    manual_dir: Path = LOCAL / "manual"
    out_dir: Path = LOCAL / "parsed"
    golden_dir: Path = LOCAL / "tests" / "golden"
    log_dir: Path = LOGS

    # --- OCR ---
    # ВАЖНО: у EasyOCR в docling язык по умолчанию ["fr","de","es","en"] — русского там нет.
    # Если не задать явно, русские сканы распознаются мусором и молча уедут в базу.
    ocr_engine: str = "easyocr"  # easyocr | tesseract | rapidocr
    ocr_langs: tuple[str, ...] = ("ru", "en")
    # OCR нужен только там, где текстового слоя нет. У статьи, свёрстанной в издательстве,
    # он есть, и распознавание картинок на каждой странице — это минуты впустую.
    # Выключение (--no-ocr) ускоряет разбор в разы; сканы ловит запасной проход.
    use_ocr: bool = True

    # --- таблицы ---
    table_mode: str = "accurate"  # accurate | fast (accurate медленнее, но держит сложные шапки)
    do_cell_matching: bool = True

    # --- формулы ---
    # Распознавать формулы в LaTeX (модель CodeFormulaV2, 0,64 ГБ, скачивается сама при
    # первом разборе). Без неё формула — пустая заглушка, и из базы она пропадает.
    # Проверка на проверочном PDF (.local/tests/formulas): 13 из 13 формул с латинскими
    # обозначениями верно; русские индексы модель читает латиницей («обр» → «o6p»),
    # поэтому к такой формуле приписывается её текст из PDF (docling.tidy_formula).
    formulas: bool = True

    # --- рисунки ---
    # Вырезать рисунки, чтобы их описала модель со зрением (добыча с --figures).
    # Без этого docling картинок не сохраняет, а PDF после разбора не хранится.
    figures: bool = False
    figures_scale: float = 2.0  # 144 точки на дюйм: мелкие подписи на карте читаются

    # --- железо ---
    device: str = "auto"  # auto | cuda | cpu | mps
    num_threads: int = 8

    # --- защита от зависаний ---
    # document_timeout внутри docling срабатывает не всегда (парсер может залипнуть
    # в нативном слое), поэтому документ считает отдельный процесс, который мы убиваем.
    doc_timeout_sec: int = 900
    ocr_doc_timeout_sec: int = 2400  # полностраничный OCR легально работает дольше
    max_pages: int = 400

    # --- чанкинг ---
    # Токенайзер обязан совпадать с моделью эмбеддинга, иначе счёт токенов врёт:
    # XLM-R (BGE-M3) режет русский примерно вдвое экономнее, чем англоязычные BERT-токенайзеры.
    embed_model_id: str = "BAAI/bge-m3"
    max_tokens: int = 512
    overlap_tokens: int = 0  # см. README: у HybridChunker нет нативного overlap

    # --- пороги валидации ---
    min_chars_per_page: int = 200  # меньше — похоже на скан без текстового слоя
    max_garbage_ratio: float = 0.10  # доля «мусорных» символов (битые шрифты, cid)
    min_sections_ratio: float = 0.8  # доля найденных разделов эталона
    min_entities_ratio: float = 0.8  # доля найденных сущностей эталона
    table_tolerance: int = 1  # допустимое расхождение в числе таблиц
    max_line_repeats: int = 30  # одна и та же строка N раз — признак поломки

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Settings:
        kwargs = {}
        types = {f.name: f.type for f in fields(cls)}
        for k, v in d.items():
            if k not in types:
                continue
            if k.endswith("_dir"):
                v = Path(v)
            elif k == "ocr_langs":
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)

    def ensure_dirs(self) -> None:
        for p in (self.out_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


# Коды языков у движков разные: easyocr — ISO 639-1, tesseract — ISO 639-2.
OCR_LANG_MAP = {
    "tesseract": {"ru": "rus", "en": "eng", "de": "deu", "fr": "fra", "es": "spa"},
}


def ocr_langs_for(engine: str, langs: tuple[str, ...]) -> list[str]:
    mapping = OCR_LANG_MAP.get(engine)
    if not mapping:
        return list(langs)
    return [mapping.get(lang, lang) for lang in langs]
