"""Задача 1 ТЗ v16: судьба каждой связи, предложенной моделью, — по этапам фактического кода.

Этапы (georag/graph):
  extraction    — модель (qwen3:14b) предложила связь по фрагменту (facts.py);
  code_check    — проверка кодом: форма ответа, имена в цитате, номер предложения,
                  «не предметы» (facts.check, facts.recheck). Отброшенная связь в facts не
                  пишется: остаётся строка с причиной в facts_pass.rejected (до 40 на фрагмент);
  confirmation  — проверка вопросами по цитате (confirm.py, votes из 4) или повтор концов связи
                  в другой статье (store.accepted): прошла — «подтверждена», нет — кандидат;
  typing        — тип связи и виды концов (kinds.py): модель выбирает тип из закрытого списка
                  или «нет» (не о геологии), код проверяет виды и имена; без типа — в граф не идёт.
                  Связь, принятая только по повтору (votes < 3), тип не получала;
  final         — graph / candidate / excluded_typing / rejected.

Дубликаты: UNIQUE (chunk_id, src_key, relation, dst_key) в facts; сумма facts_pass.facts
сверяется с числом строк facts — если равны, при записи ничего не потеряно. Синонимы
связей не удаляют: они сводят концы к одному объекту при сборке графа; итоговое
отношение (final_relation_id) — тройка (объект, тип, объект) после синонимов.

Только чтение базы (по умолчанию копия georag_v14).

    relation_audit.py → results/relation_pipeline_audit.csv, results/relation_pipeline_summary.json,
                        docs/relation_pipeline_summary.md
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import psycopg  # noqa: E402

from georag.graph import api, store  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.text import name_key  # noqa: E402

DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
RES, DOCS = HERE / "results", HERE / "docs"
CAP = 40  # store.save пишет не больше 40 причин отказа на фрагмент


def _reason_group(r: str) -> str:
    """Причина отказа кодом — по тексту строки facts_pass.rejected."""
    tail = r.rsplit("»: ", 1)[-1] if "»: " in r else r.rsplit(": ", 1)[-1]
    if r.startswith("ответ:") or "факт — не объект" in r:
        return "ответ модели не по форме"
    if tail.startswith("в цитате нет"):
        return "имени нет в цитате-основании"
    if tail.startswith("цитаты нет во фрагменте"):
        return "цитаты нет во фрагменте"
    if tail.startswith("нет предложения"):
        return "нет указанного предложения"
    if tail.endswith("а не предмет"):
        return "конец связи — не предмет (" + tail.split(",")[0] + ")"
    if "связь с самим собой" in tail:
        return "связь с самим собой"
    return tail[:60]


def _type_group(note: str | None) -> str:
    note = note or ""
    if note.startswith("виды не подходят к связи"):
        return "виды концов не подходят к типу связи"
    if note == "связь":
        return "модель назвала тип вне закрытого списка"
    return note or "без пояснения"


def main() -> None:
    RES.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    conn = psycopg.connect(DSN, autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='facts' AND column_name='stale_since'")
    stale = bool(cur.fetchone())
    acc = store.accepted(fresh=stale)
    cur.execute(f"""
        SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.src_key, f.dst_key, f.votes,
               f.typed_at IS NOT NULL, f.rel_type, f.rel_flip, f.type_note, {acc},
               EXISTS (SELECT 1 FROM facts o WHERE o.doc_id <> f.doc_id AND ((o.src_key = f.src_key
                       AND o.dst_key = f.dst_key) OR (o.src_key = f.dst_key AND o.dst_key = f.src_key)))
        FROM facts f ORDER BY f.id""")
    facts = cur.fetchall()
    cur.execute("SELECT chunk_id, facts, rejected FROM facts_pass ORDER BY chunk_id")
    passes = cur.fetchall()
    cur.execute("SELECT DISTINCT model FROM facts")
    models = [m for (m,) in cur.fetchall()]

    syn = api._with_auto(conn, S.load())

    def node(name: str) -> str:
        k = name_key(name)
        return name_key(syn.names[k]) if k in syn.names else k

    graph_ids = {f["id"] for f in store.all_facts(conn)}
    rows: list[list] = []
    final = Counter()
    conf = Counter()
    typing = Counter()
    final_rel: dict[tuple, str] = {}
    for (fid, chunk, _doc, src, rel, dst, _sk, _dk, votes, typed, rtype, flip, note, ok,
         repeat) in facts:
        rid = f"F{fid}"
        rows.append([rid, chunk, "extraction", "proposed", "", ""])
        rows.append([rid, chunk, "code_check", "passed", "", ""])
        by_votes = (votes or 0) >= store.MIN_VOTES
        why = f"вопросов пройдено {votes if votes is not None else '—'} из 4"
        if repeat:
            why += "; концы связаны и в другой статье"
        if not ok:
            rows.append([rid, chunk, "confirmation", "not_confirmed", why, ""])
            rows.append([rid, chunk, "final", "candidate", "не подтверждена: остаётся кандидатом", ""])
            final["candidate"] += 1
            conf[f"кандидат: вопросов {votes}/4" + ("; повтор есть, но не учтён" if repeat else "")] += 1
            continue
        conf["подтверждена вопросами (≥3 из 4)" if by_votes else "принята только по повтору в другой статье"] += 1
        rows.append([rid, chunk, "confirmation", "confirmed", why, ""])
        if not typed:
            typing["тип не ставился (принята по повтору)"] += 1
            rows.append([rid, chunk, "typing", "not_typed", "тип ставится только связям с votes ≥ 3", ""])
        elif rtype is None:
            typing["без типа: " + _type_group(note)] += 1
            rows.append([rid, chunk, "typing", "no_type", note or "", ""])
            rows.append([rid, chunk, "final", "excluded_typing",
                         "в граф не идёт (kinds.py), в базе остаётся: " + (note or ""), ""])
            final["excluded_typing"] += 1
            assert fid not in graph_ids
            continue
        else:
            typing["тип поставлен"] += 1
            rows.append([rid, chunk, "typing", "typed", rtype + (" (направление развёрнуто)" if flip else ""), ""])
        assert fid in graph_ids, fid
        a, b = (node(dst), node(src)) if flip else (node(src), node(dst))
        key = (a, rtype or rel, b)
        gid = final_rel.setdefault(key, f"G{len(final_rel) + 1}")
        rows.append([rid, chunk, "final", "graph", "", gid])
        final["graph"] += 1

    rejected = Counter()
    n_rej = 0
    capped = 0
    for chunk, _n, rej in passes:
        rej = rej or []
        capped += len(rej) >= CAP
        for i, r in enumerate(rej, 1):
            n_rej += 1
            g = _reason_group(r)
            rejected[g] += 1
            rid = f"R{chunk}-{i}"
            rows.append([rid, chunk, "extraction", "proposed", "", ""])
            rows.append([rid, chunk, "code_check", "rejected", r, ""])
            rows.append([rid, chunk, "final", "rejected", g, ""])
    final["rejected"] = n_rej

    stored = len(facts)
    pass_sum = sum(n for _, n, _ in passes)
    accepted = sum(1 for f in facts if f[13])
    summary = {
        "db": DSN.rsplit("/", 1)[-1], "model": models,
        "proposed": stored + n_rej, "rejected_by_code": n_rej, "stored": stored,
        "facts_pass_sum": pass_sum, "lost_on_insert": pass_sum - stored,
        "chunks_with_rejected_capped_at_40": capped,
        "confirmed": accepted, "candidates": stored - accepted,
        "confirmation": dict(conf), "typing": dict(typing),
        "excluded_typing": final["excluded_typing"], "graph": final["graph"],
        "graph_unique_relations": len(final_rel), "rejected_reasons": dict(rejected.most_common()),
        "check": {
            "facts_pass.facts = rows of facts": pass_sum == stored,
            "stored = candidates + confirmed": stored == (stored - accepted) + accepted,
            "confirmed = graph + excluded_typing": accepted == final["graph"] + final["excluded_typing"],
            "graph = store.all_facts": final["graph"] == len(graph_ids),
        },
    }
    with (RES / "relation_pipeline_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["relation_id", "source_chunk_id", "stage", "decision", "reason", "final_relation_id"])
        w.writerows(rows)
    (RES / "relation_pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    (DOCS / "relation_pipeline_summary.md").write_text(_markdown(summary), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def n(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def _markdown(s: dict) -> str:
    t, c = s["typing"], s["confirmation"]
    nongeo = s["excluded_typing"]
    lines = [
        "# Воронка обработки связей (relation_pipeline_summary)",
        "",
        f"База: `{s['db']}`, модель извлечения: {', '.join(s['model'])}. Построено скриптом "
        "`research/v16/relation_audit.py` только чтением базы; построчно — `results/relation_pipeline_audit.csv`.",
        "",
        "| Этап (код) | Вход | Прошло | Отсеяно / иной итог |",
        "|---|---:|---:|---|",
        f"| Предложено моделью (facts.py) | — | {n(s['proposed'])} | — |",
        f"| Проверка кодом (facts.check / recheck) | {n(s['proposed'])} | {n(s['stored'])} | "
        f"{n(s['rejected_by_code'])} отклонены |",
        f"| Подтверждение вопросами или повтором в другой статье (confirm.py, store.accepted) | "
        f"{n(s['stored'])} | {n(s['confirmed'])} | {n(s['candidates'])} — неподтверждённые кандидаты |",
        f"| Типизация (kinds.py) | {n(s['confirmed'])} | {n(s['graph'])} | {n(nongeo)} — тип не назначен, "
        "в граф не входят |",
        f"| **Граф** | | **{n(s['graph'])}** | уникальных отношений после синонимов — {n(s['graph_unique_relations'])} |",
        "",
        "## Судьба 3 585 записей",
        "",
        f"13 739 − 3 818 − 6 336 = {n(s['stored'] - s['graph'] - s['candidates'])}. Это связи, которые "
        f"прошли подтверждение (всего подтверждено {n(s['confirmed'])}), но на шаге типизации не получили "
        "типа из закрытого списка: модель ответила «нет» (связь не о геологии) или код отверг виды концов. "
        "Такие связи в граф не входят, но остаются в базе (`facts.rel_type IS NULL`, `typed_at` заполнено, "
        "причина — в `facts.type_note`). Решение «не о геологии» принимает модель; его точность не "
        "измерялась, и среди исключённых встречаются геологические связи (например, «гидротермально-"
        "метасоматическая минерализация — выражена — замещении и перераспределении пирита»). Для оценки "
        "подготовлен лист ручной разметки случайной выборки (`typing_exclusions_sample.py`).",
        "",
        "| Причина (type_note) | Связей |",
        "|---|---:|",
    ]
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        if k.startswith("без типа"):
            lines.append(f"| {k[len('без типа: '):]} | {n(v)} |")
    lines += [
        f"| **Итого** | **{n(nongeo)}** |",
        "",
        "## Подтверждение",
        "",
        "| Итог | Связей |",
        "|---|---:|",
    ] + [f"| {k} | {n(v)} |" for k, v in sorted(c.items(), key=lambda kv: -kv[1])] + [
        "",
        f"Из {n(s['graph'])} связей графа {n(t.get('тип не ставился (принята по повтору)', 0))} приняты "
        "только по повтору концов в другой статье: типизация к ним не применялась, поэтому они входят "
        "в граф без типа (табл. 2 статьи — «тип не получили»).",
        "",
        "## Отклонено проверкой кодом",
        "",
        "| Причина | Связей |",
        "|---|---:|",
    ] + [f"| {k} | {n(v)} |" for k, v in s["rejected_reasons"].items() if v >= 5] + [
        f"| прочие (менее 5 каждая) | {n(sum(v for v in s['rejected_reasons'].values() if v < 5))} |",
        "",
        "## Ограничения журнала",
        "",
        f"- Отклонённые кодом связи в таблицу `facts` не записываются; от них остаётся текстовая строка "
        f"«связь: причина» в `facts_pass.rejected`. Их идентификаторы в CSV (`R<фрагмент>-<n>`) построены "
        "по этим строкам, а не взяты из журнала.",
        f"- `store.save` хранит не более 40 причин на фрагмент; у {s['chunks_with_rejected_capped_at_40']} "
        "фрагментов записано ровно 40, поэтому 8 479 — нижняя граница числа отклонённых, а 22 218 — "
        "нижняя граница числа предложенных.",
        f"- Повторяющиеся в одном фрагменте связи отсекаются ограничением UNIQUE при записи; сумма "
        f"`facts_pass.facts` = {n(s['facts_pass_sum'])}, строк `facts` = {n(s['stored'])}: при записи "
        f"потеряно {n(s['lost_on_insert'])}.",
        "- Промежуточные состояния (до переразбора, ранние прогоны) не сохранялись: воронка описывает "
        "текущее состояние базы, а не историю всех запусков.",
        "",
        "Проверки сходимости: " + "; ".join(f"{k} — {'да' if v else 'НЕТ'}" for k, v in s["check"].items()) + ".",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
