"""Опыт v18: двухэтапное извлечение связей (А) против прямого извлечения в закрытый список (Б).

Вопрос рецензента: подтверждает ли эксперимент, что двухэтапная схема лучше прямой
типизации на одних и тех же предложениях. Ранняя проба (40 фрагментов, журнал
.local/logs/graph-v2-20261014 не сохранился) дала 12 фактов против 79 — здесь то же
сравнение повторяется на большей выборке, с сохранёнными результатами и ручной разметкой.

Условия одинаковы для обоих вариантов:
  * те же фрагменты (случайная выборка с фиксированным зерном), та же модель
    (Qwen3-14B, сервер GEORAG_LLM), температура 0;
  * те же правила выписывания (config/graph-rules.txt), без списка предметов первого этапа;
  * та же проверка кодом (facts.check: оба имени в предложении, имя — предмет);
  * то же подтверждение вопросами по цитате (confirm.votes, порог MIN_VOTES из 4);
  * та же проверка видов концов и названий (kinds.judge).

Различие одно.
  А (как в системе): модель пишет связь словами статьи → проверка → подтверждение →
    отдельным запросом тип из закрытого списка и виды концов (kinds.prompt).
  Б (прямо): правило 3 заменено — связь сразу одна из 14 типов, и к каждому факту виды
    концов → проверка → подтверждение → kinds.judge по видам, названным моделью.

База georag_v14 только читается.

    python research/v18/extraction_compare.py run --n 100
    python research/v18/extraction_compare.py sheet      # лист разметки (вслепую)
    python research/v18/extraction_compare.py score      # итог по размеченному листу
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from georag.graph import confirm as C  # noqa: E402
from georag.graph import facts as F  # noqa: E402
from georag.graph import kinds as K  # noqa: E402
from georag.graph.store import MIN_VOTES  # noqa: E402
from georag.llm import LLM  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DSN = "postgresql://georag:georag@127.0.0.1:5433/georag_v14"
SEED = 18
WORKERS = 4
MIN_CHARS = 200

FORMAT_A = 'Верни JSON: {"факты": [[п, "от", "связь", "к"], ...]}'
FORMAT_B = 'Верни JSON: {"факты": [[п, "от", "связь", "к", "вид от", "вид к"], ...]}'


# ---------------------------------------------------------------- правила варианта Б
def rules_direct(rules: str) -> str:
    """Правила системы, где правило 3 (связь словами статьи) заменено закрытым списком."""
    start = rules.index("3. «связь»")
    end = rules.index("4. «п»")
    relations = "\n".join(
        f"   - «{r}» ({', '.join(sorted(s))} → {', '.join(sorted(d))}): {K._MEANING[r]}"
        for r, (s, d) in K.RELATIONS.items()
    )
    rule3 = (
        "3. «связь» — строго одна из этого списка, слово в слово:\n"
        f"{relations}\n"
        "   Других связей не бывает. Если ни одна не подходит или факт не о геологии "
        "(рисунок, таблица, цвет на карте, расчёт, свойство) — факт не пиши. Связь "
        "читается от «от» к «к»: «Билляхская зона» — «входит в» — «Анабарский щит».\n"
        "   К каждому факту допиши «вид от» и «вид к» — вид каждого предмета одним словом: "
        "объект (с названием: месторождение, рудный район, щит, пояс, свита, комплекс), "
        "структура (разлом, зона сдвига, линеамент, складка), порода (порода, толща, "
        "интрузия, дайка, жила, руда), минерал, элемент (химический элемент, полезный "
        "компонент), оруденение (оруденение, минерализация, тип месторождений), процесс "
        "(метасоматоз, изменение пород, магматизм, метаморфизм, этап), признак (аномалия, "
        "ореол, геохимическая ассоциация), метод (метод, данные, съёмка).\n\n"
    )
    return rules[:start] + rule3 + rules[end:]


# ---------------------------------------------------------------- выборка
def sample(conn: Any, n: int) -> list[tuple[int, str, str]]:
    """Случайные фрагменты, где модель основного прогона предложила хоть одну связь
    (принятую или отклонённую): фрагмент о чём-то говорит. Отбор не зависит от того,
    что вышло на выходе варианта А, и одинаков для обоих вариантов."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.doc_id, c.text FROM chunks c JOIN facts_pass p ON p.chunk_id = c.id "
            "WHERE (p.facts > 0 OR cardinality(p.rejected) > 0) AND length(c.text) >= %s "
            "ORDER BY c.id",
            (MIN_CHARS,),
        )
        rows = cur.fetchall()
    random.Random(SEED).shuffle(rows)
    return sorted(rows[:n])


# ---------------------------------------------------------------- один фрагмент
def _kinds_map(answer: Any) -> dict[tuple[str, str], tuple[str, str]]:
    out: dict[tuple[str, str], tuple[str, str]] = {}
    items = answer.get("факты") if isinstance(answer, dict) else None
    for item in items or []:
        if isinstance(item, list) and len(item) >= 6:
            src, dst = F._clean_name(item[1]), F._clean_name(item[3])
            out[(F.name_key(src), F.name_key(dst))] = (str(item[4]), str(item[5]))
    return out


def _proposed(answer: Any) -> int:
    items = answer.get("факты") if isinstance(answer, dict) else None
    return len(items) if isinstance(items, list) else 0


def _bare(relation: str) -> str:
    return relation.removeprefix(F.HEDGE).strip() if hasattr(F, "HEDGE") else relation


def extract(llm: LLM, rules: str, text: str, direct: bool) -> tuple[Any, F.Checked]:
    system, user = F.messages(rules, text)
    if direct:
        user = user.replace(FORMAT_A, FORMAT_B)
    answer = llm.chat_json(
        system, user, temperature=0, max_tokens=F.ANSWER_TOKENS, repair=True,
        reserve=F.ANSWER_RESERVE,
    )
    return answer, F.check(answer, text)


def type_batch(llm: LLM, rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Типизация варианта А: как kinds.run, пачками по K.BATCH."""
    out = []
    for i in range(0, len(rows), K.BATCH):
        batch = rows[i : i + K.BATCH]
        try:
            answer = llm.chat_json(
                K.SYSTEM, K.prompt(batch), temperature=0, max_tokens=K.ANSWER_TOKENS
            )
        except Exception:  # noqa: BLE001 — нет ответа: факты пачки без типа
            answer = {}
        answer = answer if isinstance(answer, dict) else {}
        for j, (_id, src, _rel, dst, _q) in enumerate(batch, start=1):
            out.append(K.judge(answer.get(str(j)), src, dst))
    return out


def process(llm: LLM, rules: dict[str, str], chunk: tuple[int, str, str], mode: str) -> dict:
    chunk_id, doc_id, text = chunk
    direct = mode == "B"
    rec: dict[str, Any] = {"chunk_id": chunk_id, "doc_id": doc_id, "mode": mode}
    started = time.monotonic()
    try:
        answer, checked = extract(llm, rules[mode], text, direct)
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {exc}", proposed=0, checked=0, facts=[])
        return rec
    kinds_by = _kinds_map(answer) if direct else {}
    rec["proposed"] = _proposed(answer)
    rec["checked"] = len(checked.facts)
    rec["rejected"] = checked.rejected
    facts = []
    for f in checked.facts:
        v = C.votes(llm, f.src, f.relation, f.dst, f.quote)
        facts.append({"src": f.src, "relation": f.relation, "dst": f.dst, "quote": f.quote,
                      "votes": v})
    confirmed = [x for x in facts if x["votes"] >= MIN_VOTES]
    if direct:
        for x in confirmed:
            sk, dk = kinds_by.get((F.name_key(x["src"]), F.name_key(x["dst"])), ("", ""))
            ans = {"А": sk, "Б": dk, "связь": _bare(x["relation"]), "от": "А"}
            rel, flip, ks, kd, note = K.judge(ans, x["src"], x["dst"])
            x.update(rel_type=rel, rel_flip=flip, src_kind=ks, dst_kind=kd, type_note=note)
    elif confirmed:
        rows = [(i, x["src"], x["relation"], x["dst"], x["quote"]) for i, x in enumerate(confirmed)]
        for x, (rel, flip, ks, kd, note) in zip(confirmed, type_batch(llm, rows)):
            x.update(rel_type=rel, rel_flip=flip, src_kind=ks, dst_kind=kd, type_note=note)
    rec["facts"] = facts
    rec["seconds"] = round(time.monotonic() - started, 1)
    return rec


# ---------------------------------------------------------------- команды
def cmd_run(args: argparse.Namespace) -> None:
    import psycopg

    OUT.mkdir(parents=True, exist_ok=True)
    llm = LLM()
    base = F.load_rules()
    rules = {"A": base, "B": rules_direct(base)}
    (OUT / "rules_B.txt").write_text(rules["B"], encoding="utf-8")
    with psycopg.connect(DSN) as conn:
        chunks = sample(conn, args.n)
    path = OUT / f"extraction_{args.tag}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["chunk_id"], r["mode"]))
    jobs = [(c, m) for c in chunks for m in ("A", "B") if (c[0], m) not in done]
    print(f"фрагментов {len(chunks)}, заданий {len(jobs)} (уже есть {len(done)})", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(WORKERS) as pool, path.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(process, llm, rules, c, m) for c, m in jobs]
        for i, fut in enumerate(futures, 1):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(futures):
                pace = (time.monotonic() - started) / i
                print(f"  {i}/{len(futures)} · {pace:.1f} с на задание", flush=True)
    meta = {"seed": SEED, "n": args.n, "min_chars": MIN_CHARS, "min_votes": MIN_VOTES,
            "model": llm.model, "chunks": [c[0] for c in chunks]}
    (OUT / f"extraction_{args.tag}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    summarize(path)


def load(path: Path) -> list[dict]:
    recs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[(r["chunk_id"], r["mode"])] = r  # последняя запись по заданию
    return list(recs.values())


def summarize(path: Path) -> dict:
    recs = load(path)
    out: dict[str, Any] = {}
    for mode in ("A", "B"):
        rs = [r for r in recs if r["mode"] == mode]
        facts = [f for r in rs for f in r.get("facts", [])]
        conf = [f for f in facts if f["votes"] >= MIN_VOTES]
        graph = [f for f in conf if f.get("rel_type")]
        out[mode] = {
            "fragments": len(rs),
            "errors": sum("error" in r for r in rs),
            "proposed": sum(r.get("proposed", 0) for r in rs),
            "passed_check": len(facts),
            "confirmed": len(conf),
            "in_graph": len(graph),
            "fragments_with_graph": sum(
                any(f["votes"] >= MIN_VOTES and f.get("rel_type") for f in r.get("facts", []))
                for r in rs),
            "types_used": len({f["rel_type"] for f in graph}),
            "type_notes": Counter(f.get("type_note") for f in conf if not f.get("rel_type")),
            "rejected_reasons": Counter(
                s.split(": ", 1)[-1] for r in rs for s in r.get("rejected", [])).most_common(8),
        }
    path.with_name(path.stem + "_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for mode, s in out.items():
        print(f"{mode}: предложено {s['proposed']}, прошло код {s['passed_check']}, "
              f"подтверждено {s['confirmed']}, в граф {s['in_graph']} "
              f"(фрагментов с связями {s['fragments_with_graph']}, типов {s['types_used']}, "
              f"ошибок модели {s['errors']})")
    return out


def _edge(f: dict) -> tuple[str, str, str]:
    a, b = F.name_key(f["src"]), F.name_key(f["dst"])
    if f.get("rel_flip"):
        a, b = b, a
    return a, f["rel_type"], b


def cmd_sheet(args: argparse.Namespace) -> None:
    """Все связи графа обоих вариантов в одном листе, перемешано, без пометки варианта.
    Одинаковая связь (фрагмент, концы, тип) из А и Б — одна строка."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    recs = load(OUT / f"extraction_{args.tag}.jsonl")
    rows: dict[tuple, dict] = {}
    for r in recs:
        for f in r.get("facts", []):
            if f["votes"] < MIN_VOTES or not f.get("rel_type"):
                continue
            key = (r["chunk_id"], *_edge(f))
            row = rows.setdefault(key, {"chunk_id": r["chunk_id"], "fact": f, "modes": set()})
            row["modes"].add(r["mode"])
    items = list(rows.values())
    random.Random(SEED + 1).shuffle(items)
    wb = Workbook()
    ws = wb.active
    ws.title = "Разметка"
    head = ["№", "Предложение из статьи", "От", "Связь", "К", "Вид от", "Вид к",
            "Верно? (1/0)", "Оговорка автора передана? (да/нет/не нужна)", "Комментарий"]
    ws.append(head)
    key = {}
    for i, it in enumerate(items, 1):
        f = it["fact"]
        src, dst, sk, dk = f["src"], f["dst"], f["src_kind"], f["dst_kind"]
        if f.get("rel_flip"):
            src, dst, sk, dk = dst, src, dk, sk
        hedge = "предположительно " if f["relation"].startswith(F.HEDGE) else ""
        ws.append([i, " ".join(f["quote"].split()), src, hedge + f["rel_type"], dst, sk, dk,
                   "", "", ""])
        key[i] = {"chunk_id": it["chunk_id"], "modes": sorted(it["modes"])}
    widths = [5, 80, 28, 22, 28, 12, 12, 10, 16, 30]
    for col, w in zip("ABCDEFGHIJ", widths):
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    path = HERE / f"Разметка_сравнение_{args.tag}.xlsx"
    wb.save(path)
    (OUT / f"sheet_{args.tag}_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    both = sum(len(k["modes"]) == 2 for k in key.values())
    print(f"строк {len(items)} (общих для А и Б {both}) → {path}")


def cmd_score(args: argparse.Namespace) -> None:
    from openpyxl import load_workbook

    key = json.loads((OUT / f"sheet_{args.tag}_key.json").read_text(encoding="utf-8"))
    path = Path(args.file) if args.file else HERE / f"Разметка_сравнение_{args.tag}.xlsx"
    ws = load_workbook(path).active
    marks = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[7] in (None, ""):
            continue
        marks[str(row[0])] = (int(row[7]), str(row[8] or "").strip().lower())
    res: dict[str, Any] = {"labeled": len(marks), "rows": len(key)}
    correct_union = {n for n, (ok, _) in marks.items() if ok == 1}
    for mode in ("A", "B"):
        mine = [n for n, k in key.items() if mode in k["modes"] and n in marks]
        good = [n for n in mine if marks[n][0] == 1]
        lost = [n for n in mine if marks[n][1] == "нет"]
        res[mode] = {
            "labeled": len(mine), "correct": len(good),
            "precision": round(len(good) / len(mine), 3) if mine else None,
            "share_of_all_correct": round(len(good) / len(correct_union), 3)
            if correct_union else None,
            "hedge_lost": len(lost),
        }
    res["correct_total"] = len(correct_union)
    res["correct_only_A"] = sum(1 for n in correct_union if key[n]["modes"] == ["A"])
    res["correct_only_B"] = sum(1 for n in correct_union if key[n]["modes"] == ["B"])
    res["correct_both"] = sum(1 for n in correct_union if len(key[n]["modes"]) == 2)
    (OUT / f"score_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--tag", default="main")
    s = sub.add_parser("sheet")
    s.add_argument("--tag", default="main")
    c = sub.add_parser("score")
    c.add_argument("--tag", default="main")
    c.add_argument("--file")
    sub.add_parser("summary").add_argument("--tag", default="main")
    args = p.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "sheet":
        cmd_sheet(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        summarize(OUT / f"extraction_{args.tag}.jsonl")


if __name__ == "__main__":
    main()
