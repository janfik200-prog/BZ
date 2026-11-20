"""Правила графа для Qwen3-8B: варианты задания на одних и тех же фрагментах, сверка с ручной
разметкой v18.

Фрагменты — 100 фрагментов опыта v18 (база georag_v14 только читается), разметка — лист
research/v18/Разметка_сравнение_main_проверено.xlsx: 271 связь, у каждой «верно 1/0»
(разметка автора). Путь тот же, что у graph: первый этап (entities.Finder) → facts.messages →
модель-строитель → facts.check → 4 вопроса (confirm.fact_votes), в граф — ≥ 3 из 4.

Сверка: подтверждённый факт с теми же концами, что размеченная связь того же фрагмента
(ключи имён, одно содержит другое), получает её метку; концы наоборот у направленной
связи — ошибка направления. Остальные подтверждённые — «не размечены».

    python research/v19/rules_8b.py run <вариант> [--rules ФАЙЛ] [--no-list]
    python research/v19/rules_8b.py table                  → results/table.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.graph import confirm, entities, facts  # noqa: E402
from georag.llm import BUILDER_HOST, BUILDER_MODEL, LLM  # noqa: E402
from georag.text import name_key  # noqa: E402

V14 = "postgresql://georag:georag@127.0.0.1:5433/georag_v14"
V18 = ROOT / "research" / "v18"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SYMMETRIC = {"связано с", "сопровождается", "ассоциирует с", "сменяется"}


def labels() -> dict[int, list[dict]]:
    from openpyxl import load_workbook

    key = json.loads((V18 / "results" / "sheet_main_key.json").read_text("utf-8"))
    ws = load_workbook(V18 / "Разметка_сравнение_main_проверено.xlsx", read_only=True)["Разметка"]
    out: dict[int, list[dict]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[7] is None:
            continue
        cid = key[str(row[0])]["chunk_id"]
        out.setdefault(cid, []).append({"src": str(row[2]), "relation": str(row[3]),
                                        "dst": str(row[4]), "correct": int(row[7])})
    return out


def chunks(ids: list[int]) -> dict[int, str]:
    with psycopg.connect(V14) as conn:
        rows = conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()
    return {i: t or "" for i, t in rows}


def same(a: str, b: str) -> bool:
    ka, kb = name_key(a), name_key(b)
    return bool(ka and kb) and (ka in kb or kb in ka)


def run(name: str, rules_path: Path, use_list: bool, strict: bool = False) -> None:
    lab = labels()
    meta = json.loads((V18 / "results" / "extraction_main_meta.json").read_text("utf-8"))
    ids = meta["chunks"]
    texts = chunks(ids)
    rules = facts.load_rules(rules_path)
    finder = entities.Finder()
    llm = LLM(model=BUILDER_MODEL, host=BUILDER_HOST, timeout=240, temperature=0.0)

    def one(cid: int) -> dict:
        text = texts.get(cid, "")
        found = finder.find(text)
        listed = found.entities if use_list else None
        system, user = facts.messages(rules, text, listed)
        try:
            answer = llm.chat_json(system, user, max_tokens=facts.ANSWER_TOKENS, repair=True,
                                   reserve=facts.ANSWER_RESERVE)
        except Exception as exc:  # noqa: BLE001
            return {"chunk": cid, "error": str(exc)[:200]}
        raw = len(answer.get("факты") or []) if isinstance(answer, dict) else 0
        checked = facts.check(answer, text, strict=strict)
        rows = [{"src": f.src, "relation": f.relation, "dst": f.dst, "quote": f.quote,
                 "votes": confirm.fact_votes(llm, f)} for f in checked.facts]
        return {"chunk": cid, "raw": raw, "rejected": len(checked.rejected), "facts": rows}

    with ThreadPoolExecutor(12) as pool:
        results = list(pool.map(one, ids))
    (OUT / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), "utf-8")
    summary = score(results, lab)
    summary.update({"variant": name, "rules": str(rules_path), "list": use_list, "strict": strict})
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                              "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def score(results: list[dict], lab: dict[int, list[dict]]) -> dict:
    raw = rejected = checked = confirmed = matched = correct = flips = 0
    unlabeled = 0
    found_correct: set[tuple] = set()
    for r in results:
        if "error" in r:
            continue
        raw += r["raw"]
        rejected += r["rejected"]
        checked += len(r["facts"])
        for f in r["facts"]:
            if f["votes"] < 3:
                continue
            confirmed += 1
            hit = None
            for i, L in enumerate(lab.get(r["chunk"], [])):
                if same(f["src"], L["src"]) and same(f["dst"], L["dst"]):
                    hit = ("fwd", i, L)
                    break
                if same(f["src"], L["dst"]) and same(f["dst"], L["src"]):
                    hit = ("rev", i, L)
            if hit is None:
                unlabeled += 1
                continue
            way, i, L = hit
            if way == "rev" and L["relation"] not in SYMMETRIC and f["relation"] not in SYMMETRIC:
                flips += 1
                matched += 1
                continue
            matched += 1
            correct += L["correct"]
            if L["correct"]:
                found_correct.add((r["chunk"], i))
    total_correct = sum(L["correct"] for v in lab.values() for L in v)
    errors = sum(1 for r in results if "error" in r)
    return {
        "fragments": len(results) - errors, "errors": errors,
        "raw_facts": raw, "rejected_by_code": rejected, "checked": checked,
        "confirmed": confirmed, "confirmed_share": round(confirmed / max(1, checked), 3),
        "matched_labeled": matched, "correct": correct,
        "precision_labeled": round(correct / max(1, matched), 3),
        "direction_errors": flips, "unlabeled": unlabeled,
        "recall_labeled_correct": round(len(found_correct) / max(1, total_correct), 3),
        "labeled_correct_total": total_correct,
    }


CLOSED = {"входит в", "приурочено к", "контролирует", "сопровождается", "связано с",
          "расположено в", "выявлено методом", "изучено по данным", "содержит", "пересекает",
          "сменяется", "замещает", "образовано при", "является признаком"}
_JUNK = re.compile(r"\d|°|fig|рис|table|табл|см\.|km|км|м|%|this study"
                   r"|the model|данн(ые|ых)|результат", re.IGNORECASE)


def quality(name: str) -> dict:
    """По всем подтверждённым связям, без разметки: связь из закрытого списка (с пометкой
    «предположительно» — тоже), мусорный конец (число, рисунок, «данные»), фактов на фрагмент."""
    rows = [json.loads(x) for x in (OUT / f"{name}.jsonl").read_text("utf-8").splitlines()]
    conf = [f for r in rows for f in r.get("facts", []) if f["votes"] >= 3]
    rel = [f["relation"].replace("предположительно ", "") for f in conf]
    closed = sum(r in CLOSED for r in rel)
    junk = sum(bool(_JUNK.search(f["src"]) or _JUNK.search(f["dst"])) for f in conf)
    frags = sum(1 for r in rows if "error" not in r)
    return {"confirmed": len(conf), "closed_list_share": round(closed / max(1, len(conf)), 3),
            "junk_end_share": round(junk / max(1, len(conf)), 3),
            "confirmed_per_fragment": round(len(conf) / max(1, frags), 2)}


def table() -> None:
    rows = [json.loads(p.read_text("utf-8")) for p in sorted(OUT.glob("*.summary.json"))]
    for r in rows:
        r.update(quality(r["variant"]))
    (OUT / "table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    cols = ["variant", "raw_facts", "confirmed", "precision_labeled", "direction_errors",
            "recall_labeled_correct", "closed_list_share", "junk_end_share", "confirmed_per_fragment"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c)) for c in cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "table"])
    p.add_argument("name", nargs="?")
    p.add_argument("--rules", default=str(ROOT / "config" / "graph-rules.qwen3-8b.txt"))
    p.add_argument("--no-list", action="store_true")
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.name, Path(a.rules), not a.no_list, a.strict)
    else:
        table()
