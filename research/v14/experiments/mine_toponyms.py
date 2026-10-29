"""Кандидаты в справочник территорий (WP05): топонимы корпуса без модели-строителя.

Источник — все фрагменты исследовательской копии базы (GEORAG_DSN, по умолчанию
georag_v14). Два способа: spaCy (LOC/GPE/FAC) и шаблоны «X deposit / ore field /
district / belt / shield …», «месторождение X», «X рудное поле». Выход — частоты по
корпусу и по статьям; справочник собирается из них вручную (regional_registry.csv),
до формирования выборки пар, и замораживается хэшем.

    .venv/Scripts/python.exe research/v14/experiments/mine_toponyms.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
OUT = ROOT / "research" / "v14" / "data" / "toponym_candidates.csv"

EN = re.compile(
    r"\b((?:[A-Z][\w'’\-]+\s){1,4})(deposits?|ore\s+field|ore\s+district|district|belt|shield|"
    r"craton|province|region|area|inlier|terrane|block|basin|zone|shear\s+zone|massif|"
    r"cluster|prospect|mine|ridge|mountains|highland|uplift|fault|complex|group|supergroup)\b"
)
RU = re.compile(
    r"\b(месторождени\w*|рудн\w+\s+пол\w+|рудн\w+\s+район\w*|рудн\w+\s+узл?\w*|"
    r"рудопроявлени\w*|россып\w*|щит\w*|кряж\w*|нагорь\w*|пояс\w*|зон\w+|провинци\w+)\s+"
    r"((?:[А-ЯЁ][\w\-]+\s?){1,3})"
)
RU_ADJ = re.compile(
    r"\b((?:[А-ЯЁ][а-яё\-]+(?:ск|цк|ов|ев|ин)\w*\s){1,3})"
    r"(месторождени\w*|рудн\w+\s+пол\w+|рудн\w+\s+район\w*|рудн\w+\s+узл?\w*|щит\w*|кряж\w*|"
    r"нагорь\w*|пояс\w*|зон\w+|провинци\w+|платформ\w+|складчат\w+\s+\w+|прогиб\w*)"
)


def main() -> None:
    try:
        import spacy

        nlp = {"en": spacy.load("en_core_web_sm"), "ru": spacy.load("ru_core_news_sm")}
    except Exception as exc:  # noqa: BLE001
        print("spaCy недоступен:", exc, file=sys.stderr)
        nlp = {}
    from georag.graph.entities import language

    freq: Counter[str] = Counter()
    docs: defaultdict[str, set[str]] = defaultdict(set)
    how: defaultdict[str, set[str]] = defaultdict(set)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, text FROM chunks ORDER BY doc_id, ord")
        rows = cur.fetchall()
        cur.execute("SELECT doc_id, title FROM documents")
        rows += [(d, t) for d, t in cur.fetchall()]
    for i, (doc_id, text) in enumerate(rows):
        text = " ".join((text or "").split())
        for m in EN.finditer(text):
            name = (m.group(1) + m.group(2)).strip()
            freq[name] += 1
            docs[name].add(doc_id)
            how[name].add("pattern_en")
        for rx in (RU, RU_ADJ):
            for m in rx.finditer(text):
                name = " ".join(m.groups()).strip()
                freq[name] += 1
                docs[name].add(doc_id)
                how[name].add("pattern_ru")
        lang = language(text)
        for code in ("ru", "en") if lang == "mixed" else (lang,):
            if code not in nlp:
                continue
            for ent in nlp[code](text[:4000]).ents:
                if ent.label_ in ("LOC", "GPE", "FAC") and len(ent.text) >= 3:
                    name = ent.text.strip()
                    freq[name] += 1
                    docs[name].add(doc_id)
                    how[name].add("spacy_" + ent.label_)
        if i % 500 == 0:
            print(f"{i}/{len(rows)}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "mentions", "n_docs", "found_by", "doc_ids"])
        for name, n in freq.most_common():
            if len(docs[name]) >= 1 and n >= 2:
                w.writerow([name, n, len(docs[name]), ";".join(sorted(how[name])), ";".join(sorted(docs[name]))[:2000]])
    print("written", OUT)


if __name__ == "__main__":
    main()
