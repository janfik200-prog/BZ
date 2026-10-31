"""Структура корпуса для оценки словарного фильтра (задача 1 ТЗ v17).

Фрагменты базы georag_v14 раскладываются по признакам, которые раньше пересекались
незаметно: дубликат (MinHash, dedup.py), подпись или легенда (captions.py), короче 200
символов (условие выборки v16 в labeling_sheet.py, в коде системы такого правила нет).
Пересечения считаются по каждой комбинации признаков, поэтому фрагмент не исключается
дважды. Результат кешируется: подсчёт терминов по 4 272 фрагментам идёт несколько минут.

    corpus_structure.py → .cache/chunks.json, results/corpus_structure.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

CACHE = HERE / ".cache" / "chunks.json"
RES = HERE / "results"
SHORT = 200  # условие отбора v16: len(text) >= 200


def bucket(t: int) -> str:
    return "5+" if t >= 5 else str(t)


def load() -> list[dict]:
    """Фрагменты с числом терминов и признаками; при первом запуске — из базы (labeling_sheet.chunk_terms)."""
    if CACHE.exists():
        return json.loads(CACHE.read_text("utf-8"))
    import psycopg

    from labeling_sheet import DSN, chunk_terms

    with psycopg.connect(DSN, autocommit=True) as conn:
        chunks = chunk_terms(conn)
    for c in chunks:
        c["length"] = len(c["text"])
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(chunks, ensure_ascii=False), "utf-8")
    return chunks


def main() -> None:
    chunks = load()
    combo = Counter((c["duplicate"], c["caption"], c["length"] < SHORT) for c in chunks)
    domain = [c for c in chunks if not c["duplicate"] and not c["caption"]]  # область фильтра
    frame = [c for c in domain if c["length"] >= SHORT]  # из чего отбирала выборка v16
    short = [c for c in domain if c["length"] < SHORT]
    out = {
        "total": len(chunks),
        "duplicates": sum(c["duplicate"] for c in chunks),
        "captions": sum(c["caption"] for c in chunks),
        "short_lt_200": sum(c["length"] < SHORT for c in chunks),
        "combinations": [{"duplicate": d, "caption": cp, "short": s, "chunks": n}
                         for (d, cp, s), n in sorted(combo.items())],
        "excluded_any": sum(c["duplicate"] or c["caption"] for c in chunks),
        "filter_domain": len(domain),
        "sampling_frame_v16": len(frame),
        "short_in_domain": len(short),
        "domain_by_terms": dict(Counter(bucket(c["terms"]) for c in domain)),
        "frame_by_terms": dict(Counter(bucket(c["terms"]) for c in frame)),
        "short_by_terms": dict(Counter(bucket(c["terms"]) for c in short)),
        "short_tokens": sum(c["n_tokens"] for c in short),
        "domain_tokens": sum(c["n_tokens"] for c in domain),
        "short_with_graph_facts": sum(c["facts_graph"] > 0 for c in short),
        "short_with_stored_facts": sum(c["facts_stored"] > 0 for c in short),
    }
    assert sum(x["chunks"] for x in out["combinations"]) == out["total"]
    assert out["sampling_frame_v16"] + out["short_in_domain"] == out["filter_domain"]
    RES.mkdir(exist_ok=True)
    (RES / "corpus_structure.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
