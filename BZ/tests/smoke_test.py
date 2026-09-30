"""Прогон логики без docling и без сети.

Запуск:  python tests/smoke_test.py [путь_к.pdf]

Проверяем то, что не зависит от загрузки моделей: эвристики валидации,
русскую морфологию в сверке с эталоном, бюджет токенов и перекрытие в чанкинге,
а также полный путь пайплайна с запасным парсером.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.parse.config import Settings  # noqa: E402
from georag.parse.models import OK, ParsedDoc  # noqa: E402
from georag.parse.validate import _entity_pattern, garbage_ratio, validate  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "OK  " if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def test_garbage_ratio() -> None:
    print("\nМусор и кодировка")
    clean = "Плотность линеаментов на Анабарском щите оценена методом кокригинга (n = 128)."
    broken = "(cid:34)(cid:9)(cid:120) ��� \x01\x02\x03 ???"
    check("чистый русский текст проходит", garbage_ratio(clean) < 0.05, f"{garbage_ratio(clean):.1%}")
    check("битые шрифты ловятся", garbage_ratio(broken) > 0.10, f"{garbage_ratio(broken):.1%}")


def test_morphology() -> None:
    print("\nМорфология при сверке с эталоном")
    text = (
        "в пределах анабарского щита выделены зоны окварцевания; "
        "прогноз выполнен методом кокригинга по плотности линеаментов (aster)"
    )
    cases = [
        ("Анабарский щит", True),
        ("кокригинг", True),
        ("плотность линеаментов", True),
        ("зона окварцевания", True),
        ("ASTER", True),
        ("кимберлитовая трубка", False),
    ]
    for entity, expected in cases:
        found = bool(_entity_pattern(entity).search(text))
        check(f"«{entity}» → {'найдено' if expected else 'не найдено'}", found == expected)


def test_validation_flow() -> None:
    print("\nВалидация и подсказки пайплайну")
    s = Settings()
    golden = {
        "pages": 2,
        "tables": 1,
        "sections": ["Введение", "Методы", "Результаты"],
        "entities": ["Анабарский щит", "кокригинг"],
    }

    good = ParsedDoc(
        doc_id="good",
        source_path="good.pdf",
        parser="docling",
        status=OK,
        page_count=2,
        table_count=1,
        pages_text={
            1: "Введение. " + "Исследование Анабарского щита. " * 20,
            2: "Методы. Применён кокригинг. Результаты. " + "Оценка перспективности. " * 20,
        },
    )
    report = validate(good, s, golden)
    check("корректный документ проходит", report.ok, "; ".join(c.name for c in report.failed))

    scan = ParsedDoc(
        doc_id="scan",
        source_path="scan.pdf",
        parser="docling",
        status=OK,
        page_count=10,
        pages_text={i: "" for i in range(1, 11)},
    )
    report_scan = validate(scan, s, golden)
    check("скан без текста отбраковывается", not report_scan.ok)
    check("скан отправляется на OCR", report_scan.suggestion == "rerun_ocr", report_scan.suggestion)

    broken = ParsedDoc(
        doc_id="broken",
        source_path="broken.pdf",
        parser="docling",
        status=OK,
        page_count=1,
        pages_text={1: "(cid:12)(cid:45)(cid:7) " * 200},
    )
    report_broken = validate(broken, s, golden)
    check("битые шрифты отбраковываются", not report_broken.ok)
    check(
        "битые шрифты уходят на другой парсер, а не на OCR",
        report_broken.suggestion == "fallback",
        report_broken.suggestion,
    )

    missing = ParsedDoc(
        doc_id="missing",
        source_path="missing.pdf",
        parser="docling",
        status=OK,
        page_count=2,
        table_count=1,
        pages_text={1: "Введение. " * 40, 2: "Совершенно другой текст. " * 40},
    )
    report_missing = validate(missing, s, golden)
    check("потерянные разделы эталона ловятся", not report_missing.ok)


# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    """Псевдотокенайзер: слово = токен. Нужен, чтобы проверить бюджет и overlap без сети."""

    def __init__(self) -> None:
        self.vocab: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            self.vocab.append(word)
            ids.append(len(self.vocab) - 1)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.vocab[i] for i in ids)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> "_FakeTokenizer":
        return cls()


def _install_fake_transformers() -> None:
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    sys.modules["transformers"] = module


def test_chunking_budget_and_overlap() -> None:
    print("\nЧанкинг: бюджет токенов и перекрытие")
    _install_fake_transformers()
    from georag.parse.chunking import _chunk_plain_text

    paragraphs = ["слово " * 120, "текст " * 120, "абзац " * 120]
    parsed = ParsedDoc(
        doc_id="plain",
        source_path="plain.pdf",
        parser="pymupdf4llm",
        status=OK,
        page_count=1,
        pages_text={1: "\n\n".join(p.strip() for p in paragraphs)},
    )

    s = Settings(max_tokens=200, overlap_tokens=0)
    chunks = _chunk_plain_text(parsed, s)
    check("документ порезан на несколько чанков", len(chunks) >= 2, f"{len(chunks)} чанков")
    check("бюджет токенов соблюдён", all(c.n_tokens <= s.max_tokens for c in chunks),
          f"максимум {max(c.n_tokens for c in chunks)}")
    check("страница сохранена для цитирования", all(c.pages == [1] for c in chunks))

    s_ov = Settings(max_tokens=200, overlap_tokens=30)
    chunks_ov = _chunk_plain_text(parsed, s_ov)
    grew = [c for c in chunks_ov[1:] if len(c.embed_text) > len(c.text)]
    check("перекрытие добавлено в embed_text", len(grew) == len(chunks_ov) - 1)
    check(
        "текст для цитирования не изменён",
        all(c.text == c.embed_text.split("\n\n", 1)[-1] for c in chunks_ov[1:]),
    )
    check(
        "итоговый размер не вылезает за max_tokens",
        all(c.n_tokens <= s_ov.max_tokens for c in chunks_ov),
        f"максимум {max(c.n_tokens for c in chunks_ov)}",
    )


# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end(pdf: Path) -> None:
    print(f"\nПайплайн целиком на реальном файле: {pdf.name}")
    from georag.parse.pipeline import DoclingWorker, StepLogger, process_document

    out = Path("/tmp/georag-smoke")
    s = Settings(
        out_dir=out / "parsed",
        log_dir=out / "logs",
        golden_dir=out / "golden",
        manual_dir=out / "manual",
        doc_timeout_sec=60,
    )
    s.ensure_dirs()

    worker = DoclingWorker(s)
    logger = StepLogger(s.log_dir)
    try:
        result, _parsed, _chunks = process_document(pdf, s, worker, logger, make_chunks=False)
    finally:
        worker.close()

    check("документ обработан", result.status in {"ok", "partial"}, f"статус {result.status}")
    check("сработал запасной парсер", result.parser == "pymupdf4llm", result.parser)
    check("текст извлечён", (s.out_dir / f"{pdf.stem}.md").stat().st_size > 1000)
    check("страницы посчитаны", result.pages > 0, f"{result.pages} страниц")
    check("JSONL-лог написан", logger.path.exists() and logger.path.stat().st_size > 0)
    check(
        "в логе есть попытка docling до fallback",
        any(a.startswith("docling") for a in result.attempts),
        ", ".join(result.attempts),
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    test_garbage_ratio()
    test_morphology()
    test_validation_flow()
    test_chunking_budget_and_overlap()

    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if pdf_arg and Path(pdf_arg).exists():
        test_pipeline_end_to_end(Path(pdf_arg))
    else:
        print("\n(PDF не передан — сквозной прогон пропущен)")

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
