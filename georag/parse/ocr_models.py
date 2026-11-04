"""Модели распознавания сканов (RapidOCR) — в папке проекта, чтобы OCR шёл без интернета.

Без этой папки RapidOCR при первом скане скачивает модели сам (в пакет rapidocr внутри
.venv). Закрытая БЗ работает без интернета и переносится на другой диск — ей модели
нужны заранее, рядом с проектом:

    python -m georag.parse.ocr_models        положить модели в .local/models/rapidocr

Команда берёт модели из пакета rapidocr, если они там уже есть (после первого скана с
интернетом), а нет — скачивает. Три файла, около 13 МБ: поиск строк, поворот строки и
распознавание букв (кириллица; читает и латиницу). Те же, что RapidOCR выбирает сам для
ocr_lang = cyrillic: текст распознаётся одинаково (проверено 04.11.2026).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .config import Settings


def files(lang: str) -> dict[str, str]:
    """Имена файлов моделей: поиск строк (det), поворот (cls), распознавание (rec)."""
    return {
        "det": "ch_PP-OCRv5_det_mobile.onnx",
        "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "rec": f"{lang}_PP-OCRv5_rec_mobile.onnx",
    }


def local_paths(settings: Settings) -> dict[str, Path] | None:
    """Пути к моделям в папке проекта (Settings.ocr_models); не все на месте — None:
    тогда RapidOCR берёт модели сам (и скачивает, если их нет)."""
    paths = {k: settings.ocr_models / name for k, name in files(settings.ocr_lang).items()}
    return paths if all(p.is_file() for p in paths.values()) else None


def _package_dir() -> Path:
    import rapidocr

    return Path(rapidocr.__file__).parent / "models"


def _download(lang: str) -> None:
    """Модели — в пакет rapidocr (он скачивает их сам, создавая распознаватель)."""
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec(lang),
        }
    )


def prepare(settings: Settings) -> dict[str, Path]:
    """Модели — в Settings.ocr_models: из пакета rapidocr, нет там — скачать."""
    wanted = files(settings.ocr_lang)
    source = _package_dir()
    if not all((source / name).is_file() for name in wanted.values()):
        _download(settings.ocr_lang)
    settings.ocr_models.mkdir(parents=True, exist_ok=True)
    for name in wanted.values():
        target = settings.ocr_models / name
        if not target.is_file():
            shutil.copy2(source / name, target)
    paths = local_paths(settings)
    if paths is None:
        raise FileNotFoundError(f"моделей OCR нет в {settings.ocr_models}")
    return paths


def main(argv: list[str] | None = None) -> int:
    del argv
    settings = Settings()
    try:
        paths = prepare(settings)
    except Exception as exc:  # noqa: BLE001 — нет интернета и моделей в пакете
        print(f"Модели OCR не подготовлены: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    size = sum(p.stat().st_size for p in paths.values()) / 1e6
    print(f"Модели OCR ({size:.0f} МБ) — в {settings.ocr_models}: OCR пойдёт без интернета.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
