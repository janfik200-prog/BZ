"""Генерация из corpus/corpus.json: эталоны и описание корпуса.

Данные — в corpus.json, его и правьте, когда добавляете статью. Здесь только
генерация: python corpus/build_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "golden"


def load() -> list[dict]:
    return json.loads((CORPUS / "corpus.json").read_text(encoding="utf-8"))


def write_golden(articles: list[dict]) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for art in articles:
        golden = {
            "title": art["title"],
            "lang": art["lang"],
            "kind": "статья",
            "pages": art["pages"],
            "tables": art["tables"],
            "sections": art["sections"],
            "entities": art["entities"],
            "notes": f"{art['probe']}. Что проверяет: {art['tests']}",
            "source": {
                "journal": art["journal"],
                "year": art["year"],
                "doi": art["doi"],
                "url": art["url"],
            },
        }
        (GOLDEN / f"{art['id']}.json").write_text(
            json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def write_readme(articles: list[dict]) -> None:
    lines = [
        "# Тестовый корпус: выделение рудных узлов",
        "",
        "10 статей по теме: 8 русских из КиберЛенинки и 2 английские из MDPI. Подобраны",
        "под разные типы вёрстки — скан, тяжёлая графика, плотные таблицы, две колонки.",
        "Нужны, чтобы мерить точность парсинга: эталоны к ним лежат в `golden/`.",
        "",
        "```",
        "python corpus/download_corpus.py",
        "python -m georag.cli --input data/pdf --device cuda",
        "```",
        "",
        "Данные корпуса — в `corpus.json`. Добавили статью туда — запустите",
        "`python corpus/build_corpus.py`, и эталоны с описанием обновятся; скачивание (`download_corpus.py`) берёт список прямо из corpus.json.",
        "",
        "| Файл | Статья | Год | Стр. | Табл. | Что проверяет |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for art in articles:
        short = art["title"][:70] + ("…" if len(art["title"]) > 70 else "")
        lines.append(
            f"| `{art['id']}.pdf` | [{short}]({art['url']}) | {art['year']} | "
            f"{art['pages'] or '?'} | {'?' if art['tables'] is None else art['tables']} | "
            f"{art['tests'].split('—')[0].strip()} |"
        )
    lines += [
        "",
        "## Про эталоны",
        "",
        "Разделы («Введение», «Методы») есть только в одной русской статье из восьми и в",
        "обеих английских: русская геологическая периодика обычно идёт сплошным текстом",
        "с УДК и ключевыми словами. Поэтому структурная проверка для неё почти всегда",
        "пустая, а вес переносится на сущности — они взяты из авторских ключевых слов.",
        "",
        "Число таблиц стоит там, где его можно обосновать; где нельзя — `null`, и проверка",
        "пропускается: неверный эталон хуже отсутствующего. Числа страниц взяты из самих",
        "PDF, кроме двух файлов со сжатыми object stream.",
    ]
    (CORPUS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    articles = load()
    write_golden(articles)
    write_readme(articles)
    print(f"эталонов: {len(articles)} → {GOLDEN}")
    print(f"описание → {CORPUS / 'README.md'}")
