"""Проверка прослеживаемости, прозрачности и обоснованности графа и ответов — одним запуском.

База только читается (сессия read-only), граф не меняется. Подходит и для старого графа,
построенного до проверок кода: у каждой связи в таблице facts есть своя цитата, и все
проверки ниже делаются по ней заново.

1. Прослеживаемость — можно ли дойти от связи до источника:
   цитата есть / найдена в тексте фрагмента (дословно, после нормализации пробелов,
   по частям) / у фрагмента есть статья, страница, раздел; у статьи DOI или ссылка.
2. Территория — к какому месту относится связь и откуда это взято
   (georag/evidence/scope.py: цитата, соседние предложения, раздел, название статьи).
3. Прозрачность — что решала модель и что отбросил код: какой моделью построен граф,
   сколько связей в графе и сколько кандидатов, сколько связей подтверждено вопросами
   при построении (facts.votes), почему код отбрасывал ответы модели (facts_pass).
4. Обоснованность — подтверждает ли цитата связь (проверяется заново, по цитате):
   код: оба конца названы в цитате; нет отрицания и «ранее считалось»; не строка легенды;
        конец — не число; направление по порядку слов; оговорка автора не потеряна;
   модель: четыре вопроса к цитате (graph/confirm.py), подтверждена при ≥ 3 из 4.
   «Связь не из закрытого списка 14» — правило формы, а не обоснованности: считается
   отдельно и в вердикт не входит (старый граф писал связь словами статьи).
5. Насколько верить проверке 4 — та же проверка на 120 связях, размеченных вручную
   (gold_120.csv: 104 верные, 16 ошибок): сколько ошибок она ловит, сколько верных
   отбрасывает. Базы не нужно: связь и цитата лежат в файле.
6. Ответы чат-бота (--chat N): доля предложений со ссылкой [N], ссылки на несуществующий
   фрагмент, сколько утверждений убрала сверка ссылок, сколько отказов «в базе нет».

Запуск (из корня проекта):
    python research/trace_audit/trace_audit.py                     # всё, кроме чат-бота
    python research/trace_audit/trace_audit.py --sample 1500       # вопросы — по выборке связей
    python research/trace_audit/trace_audit.py --no-llm            # только код, без модели
    python research/trace_audit/trace_audit.py --chat 30           # плюс 30 вопросов чат-боту

Модель для вопросов — модель-строитель (GEORAG_BUILDER, по умолчанию Qwen3-8B на :8101);
другая — --host и --model. Для --chat нужна модель чат-бота (Qwen3-14B на :8100): обе
сразу на 16 ГБ не помещаются, поэтому удобно два запуска — сначала граф с 8B, потом
--only-chat с 14B. Ответы модели складываются в results/votes_cache.jsonl: прерванный
запуск продолжает с места остановки.

Результат: results/summary.json (все числа), results/report.md (то же словами),
results/facts.csv (вердикт по каждой связи), results/gold_check.csv, results/chat.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
RES = HERE / "results"

import psycopg  # noqa: E402

from georag.graph import confirm, store  # noqa: E402
from georag.graph import facts as FX  # noqa: E402
from georag.index.db import DEFAULT_DSN  # noqa: E402
from georag.text import locate_quote, mentions  # noqa: E402

MIN_VOTES = store.MIN_VOTES


# --------------------------------------------------------------------------- общее
def ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def pct(k: int, n: int) -> float | None:
    return round(100 * k / n, 1) if n else None


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95-процентный интервал доли (Уилсон): для выборки, а не всего графа."""
    if not n:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def share(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "pct": pct(k, n)}


def read_only(dsn: str) -> str:
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}options=-c%20default_transaction_read_only%3Don"


def columns(conn: Any, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,)
    ).fetchall()
    return {r[0] for r in rows}


# --------------------------------------------------------------------------- данные
def load_facts(conn: Any) -> list[dict[str, Any]]:
    """Все связи таблицы facts (и в графе, и кандидаты) со статьёй и фрагментом."""
    fc = columns(conn, "facts")
    opt = {name: (f"f.{name}" if name in fc else "NULL") for name in
           ("votes", "model", "rel_type", "rel_flip", "typed_at")}
    rows = conn.execute(
        f"""
        SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.quote,
               {opt['votes']}, {opt['model']}, {opt['rel_type']}, {opt['rel_flip']},
               {opt['typed_at']}, c.text, c.headings, c.pages, d.title, d.doi, d.url, d.year
        FROM facts f JOIN chunks c ON c.id = f.chunk_id JOIN documents d ON d.doc_id = f.doc_id
        ORDER BY f.id
        """
    ).fetchall()
    keys = ("id", "chunk_id", "doc_id", "src", "relation", "dst", "quote", "votes", "model",
            "rel_type", "rel_flip", "typed_at", "chunk_text", "headings", "pages", "title",
            "doi", "url", "year")
    out = [dict(zip(keys, r, strict=True)) for r in rows]
    in_graph = {f["id"] for f in store.all_facts(conn)}
    for f in out:
        f["in_graph"] = f["id"] in in_graph
    return out


# --------------------------------------------------------------------------- 1. прослеживаемость
def locate(quote: str, text: str) -> str:
    """Где нашлась цитата в тексте фрагмента."""
    if not ws(quote):
        return "нет цитаты"
    if quote in text:
        return "дословно"
    if ws(quote) in ws(text):
        return "после нормализации пробелов"
    if locate_quote(quote, text):
        return "по частям (многоточие, переносы)"
    return "не найдена"


def trace_block(fs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(fs)
    where = Counter(f["locate"] for f in fs)
    found = sum(v for k, v in where.items() if k not in ("нет цитаты", "не найдена"))
    full = sum(1 for f in fs if f["locate"] not in ("нет цитаты", "не найдена")
               and f["pages"] and f["title"])
    return {
        "связей": n,
        "цитата найдена в тексте фрагмента": share(found, n),
        "как найдена": dict(where.most_common()),
        "у фрагмента есть страница": share(sum(1 for f in fs if f["pages"]), n),
        "у фрагмента есть раздел": share(sum(1 for f in fs if f["headings"]), n),
        "у статьи есть DOI или ссылка": share(sum(1 for f in fs if f["doi"] or f["url"]), n),
        "полная цепочка (цитата найдена + статья + страница)": share(full, n),
        "статей": len({f["doc_id"] for f in fs}),
        "фрагментов": len({f["chunk_id"] for f in fs}),
    }


# --------------------------------------------------------------------------- 2. территория
def territory(fs: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from georag.evidence import gate
        from georag.evidence.scope import resolve_scope
        reg = gate.registry()
    except Exception as exc:  # noqa: BLE001 — нет справочника территорий
        return {"ошибка": f"справочник территорий не загрузился: {exc}"[:300]}
    names = {"EXPLICIT": "из цитаты", "ANAPHORA": "из соседних предложений",
             "SECTION": "из заголовка раздела", "TITLE": "из названия статьи"}
    for f in fs:
        sc = resolve_scope(f["quote"], reg, chunk_text=ws(f["chunk_text"]),
                           headings=f["headings"] or [], title=f["title"] or "")
        f["territory"] = "|".join(sorted(sc.region_ids)) if sc.resolved else ""
        f["territory_from"] = names.get(sc.derivation, sc.derivation) if sc.resolved else "не определена"
    n = len(fs)
    got = sum(1 for f in fs if f["territory"])
    return {"связей": n, "территория определена": share(got, n),
            "откуда взята": dict(Counter(f["territory_from"] for f in fs).most_common())}


# --------------------------------------------------------------------------- 3. прозрачность
def transparency(conn: Any, fs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(fs)
    out: dict[str, Any] = {
        "связей в таблице": n,
        "в графе (подтверждены вопросами ≥ 3 из 4 или повторены в другой статье)":
            share(sum(f["in_graph"] for f in fs), n),
        "кандидаты (в базе, в граф не идут)": sum(not f["in_graph"] for f in fs),
        "модель, построившая граф": dict(Counter(f["model"] or "не записана" for f in fs).most_common()),
        "подтверждение вопросами при построении (facts.votes)":
            dict(sorted(Counter("не проверялась" if f["votes"] is None else f"{f['votes']} из 4"
                                for f in fs).items())),
    }
    typed = [f for f in fs if f["typed_at"] is not None]
    if typed:
        out["тип связи поставлен (kinds.py)"] = share(sum(1 for f in typed if f["rel_type"]), len(typed))
        out["тип «не о геологии» — в граф не идёт"] = sum(1 for f in typed if not f["rel_type"])
    if "rejected" in columns(conn, "facts_pass"):
        passed = conn.execute("SELECT count(*), sum(facts), sum(cardinality(rejected)) "
                              "FROM facts_pass").fetchone()
        out["фрагментов прочитано моделью"] = passed[0]
        out["ответов модели отброшено кодом при построении"] = passed[2] or 0
        out["причины отказа кода (при построении)"] = {
            k: c for k, c, _ in store.rejected_reasons(conn, examples=0)[:15]}
        if "skipped" in columns(conn, "facts_pass"):
            out["фрагменты, не отданные модели (почему)"] = dict(conn.execute(
                "SELECT skipped, count(*) FROM facts_pass WHERE skipped IS NOT NULL "
                "GROUP BY 1 ORDER BY 2 DESC").fetchall())
    return out


# --------------------------------------------------------------------------- 4. обоснованность
def base_relation(f: dict[str, Any]) -> tuple[str, bool]:
    """Связь без «предположительно», сведённая к закрытому списку, если получится.
    Старый граф писал связь словами статьи, а тип ставил отдельно (rel_type)."""
    rel = ws(f["relation"]).lower()
    hedge = rel.startswith(f"{FX.HEDGE} ")
    rel = rel.removeprefix(f"{FX.HEDGE} ")
    norm, _ = FX.normalize_relation(rel)
    if norm not in FX.CLOSED_RELATIONS and f.get("rel_type") in FX.CLOSED_RELATIONS:
        norm = f["rel_type"]
    return norm, hedge


def code_check(f: dict[str, Any]) -> list[str]:
    """Причины, по которым цитата не подтверждает связь (по правилам кода). Пусто — нет."""
    src, dst, quote = ws(f["src"]), ws(f["dst"]), ws(f["quote"])
    rel, hedge = base_relation(f)
    why: list[str] = []
    if not quote:
        return ["нет цитаты"]
    if not mentions(src, quote):
        why.append("в цитате нет конца «от»")
    if not mentions(dst, quote):
        why.append("в цитате нет конца «к»")
    if rel.startswith("не ") or " не " in f" {rel} ":
        why.append("связь — отрицание")
    own = " ".join(x for x in FX.split_sentences(quote) if mentions(dst, x)) or quote
    neg = FX._NEGATION_BY_RELATION.get(rel)
    if not neg and rel.split():
        neg = r"\bне\s+" + re.escape(rel.split()[0][:5])  # связь словами статьи
    if (neg and re.search(neg, own, re.IGNORECASE)) or FX._ABSENCE.search(own):
        why.append("в цитате отрицание")
    if FX._OUTDATED.search(own):
        why.append("«ранее считалось»")
    if FX._CAPTION.match(quote) or len(FX._LEGEND.findall(quote)) >= 2:
        why.append("строка легенды или подписи")
    if FX._MEASURE.search(src) or FX._MEASURE.search(dst):
        why.append("конец — число или угол")
    a, b = FX.strict_direction(src, rel, dst, quote)
    if (a, b) != (src, dst):
        why.append("направление обратно порядку слов")
    if not hedge and FX.hedged(rel, quote) != rel:
        why.append("оговорка автора потеряна")
    return why


def ask_all(items: list[dict[str, Any]], host: str, model: str, cache_path: Path,
            threads: int) -> None:
    """Четыре вопроса к цитате для каждой связи (graph/confirm.py) → it['llm_votes']."""
    from georag.llm import LLM

    cache: dict[str, int] = {}
    if cache_path.exists():
        for line in cache_path.read_text("utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["votes"]
    def key(it: dict[str, Any]) -> str:
        return json.dumps([model, ws(it["src"]), ws(it["relation"]), ws(it["dst"]), ws(it["quote"])],
                          ensure_ascii=False)
    todo = [it for it in items if key(it) not in cache]
    llm = LLM(model=model, host=host, timeout=180, temperature=0.0)
    done, t0 = 0, time.time()
    with cache_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(threads) as pool:
        def one(it: dict[str, Any]) -> tuple[dict[str, Any], int]:
            rel = ws(it["relation"])
            return it, confirm.votes(llm, ws(it["src"]), rel, ws(it["dst"]), ws(it["quote"]))
        for it, v in pool.map(one, todo):
            cache[key(it)] = v
            out.write(json.dumps({"key": key(it), "votes": v}, ensure_ascii=False) + "\n")
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                print(f"  вопросы: {done}/{len(todo)}, {rate:.1f} связи/с, "
                      f"осталось ≈ {(len(todo) - done) / rate / 60:.0f} мин", flush=True)
    for it in items:
        it["llm_votes"] = cache.get(key(it))


def verdict(it: dict[str, Any], use_llm: bool) -> str:
    if it["code_reasons"]:
        return "не подтверждена кодом"
    if use_llm:
        v = it.get("llm_votes")
        if v is None:
            return "не проверялась моделью"
        return "подтверждена" if v >= MIN_VOTES else "не подтверждена вопросами"
    return "прошла проверки кода"


def grounding_block(fs: list[dict[str, Any]], use_llm: bool) -> dict[str, Any]:
    checked = [f for f in fs if f.get("verdict") and f["verdict"] != "не проверялась моделью"]
    n = len(checked)
    ok_code = sum(1 for f in checked if not f["code_reasons"])
    reasons = Counter(r for f in checked for r in f["code_reasons"])
    out: dict[str, Any] = {
        "проверено связей": n,
        "прошли проверки кода": share(ok_code, n),
        "причины (у связи их может быть несколько)": dict(reasons.most_common()),
        "связь не из закрытого списка 14 (правило формы, в вердикт не входит)":
            share(sum(1 for f in checked if base_relation(f)[0] not in FX.CLOSED_RELATIONS), n),
    }
    if use_llm:
        conf = sum(1 for f in checked if f["verdict"] == "подтверждена")
        out["подтверждена цитатой (код + ≥ 3 из 4 вопросов)"] = share(conf, n)
        out["95 % интервал, %"] = wilson(conf, n)
        out["ответы на вопросы, совпавших из 4"] = dict(sorted(Counter(
            f"{f['llm_votes']} из 4" for f in checked if f.get("llm_votes") is not None).items()))
    out["вердикты"] = dict(Counter(f["verdict"] for f in checked).most_common())
    g = [f for f in checked if f["in_graph"]]
    if g and use_llm:
        out["только связи графа: подтверждена цитатой"] = share(
            sum(1 for f in g if f["verdict"] == "подтверждена"), len(g))
    return out


# --------------------------------------------------------------------------- 5. разметка 120
def gold_block(use_llm: bool, host: str, model: str, threads: int) -> dict[str, Any]:
    path = HERE / "gold_120.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        r["rel_type"] = None
        r["code_reasons"] = code_check(r)
    if use_llm:
        ask_all(rows, host, model, RES / "votes_cache.jsonl", threads)
    for r in rows:
        r["verdict"] = verdict(r, use_llm)
        r["flagged"] = r["verdict"] not in ("подтверждена", "прошла проверки кода")
    wrong = [r for r in rows if r["correct"] == "0"]
    right = [r for r in rows if r["correct"] == "1"]
    caught = sum(r["flagged"] for r in wrong)
    lost = sum(r["flagged"] for r in right)
    passed = [r for r in rows if not r["flagged"]]
    with (RES / "gold_check.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "верна по разметке", "категория ошибки", "вердикт проверки",
                    "причины кода", "совпало из 4", "src", "relation", "dst", "quote"])
        for r in rows:
            w.writerow([r["no"], r["correct"], r["category"], r["verdict"],
                        "; ".join(r["code_reasons"]), r.get("llm_votes"),
                        r["src"], r["relation"], r["dst"], r["quote"]])
    by_cat: dict[str, list[int]] = {}
    for r in wrong:
        cat = (r["category"] or "без категории").split(";")[0].strip()
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += r["flagged"]
        by_cat[cat][1] += 1
    return {
        "связей размечено": len(rows),
        "верных по разметке": len(right),
        "ошибок по разметке": len(wrong),
        "ошибок поймано проверкой": share(caught, len(wrong)),
        "верных отброшено проверкой": share(lost, len(right)),
        "среди прошедших проверку верных": share(sum(r["correct"] == "1" for r in passed), len(passed)),
        "95 % интервал (верных среди прошедших), %":
            wilson(sum(r["correct"] == "1" for r in passed), len(passed)),
        "поймано по видам ошибок": {k: f"{v[0]} из {v[1]}" for k, v in by_cat.items()},
        "без проверки верных было": share(len(right), len(rows)),
    }


# --------------------------------------------------------------------------- 6. чат-бот
CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def cited(sentence: str) -> list[int]:
    nums: list[int] = []
    for group in CITE.findall(sentence):
        for part in re.split(r"\s*,\s*", group):
            if re.match(r"^\d+\s*[–-]\s*\d+$", part):
                a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part))
                nums += list(range(a, b + 1))
            elif part.strip().isdigit():
                nums.append(int(part))
    return nums


def chat_block(limit: int) -> dict[str, Any]:
    import yaml

    from georag.chat import answer as A
    from georag.chat.settings import Settings
    from georag.index import db
    from georag.index.embed import Embedder

    data = yaml.safe_load((ROOT / "config" / "eval-questions.yaml").read_text("utf-8"))
    qs = [x["вопрос"] for x in data["вопросы"] if x.get("ожидание") == "база"][:limit]
    emb = Embedder()
    rows = []
    with db.connect() as conn:
        for q in qs:
            ev: dict[str, Any] = {"question": q, "before": [], "final": None, "sources": [],
                                  "mode": None, "unsupported": 0, "error": ""}
            try:
                for e in A.answer(conn, emb, q, [], Settings()):
                    t = e.get("type")
                    if t == "token":
                        ev["before"].append(e["text"])
                    elif t == "sources":
                        ev["sources"] = [s["n"] for s in e["sources"]]
                    elif t == "revised":
                        ev["final"] = e.get("text")
                    elif t == "done":
                        ev["mode"] = e.get("mode")
                        ev["unsupported"] = int(e.get("unsupported") or 0)
            except Exception as exc:  # noqa: BLE001
                ev["error"] = str(exc)[:200]
            text = ev["final"] if ev["final"] is not None else "".join(ev["before"])
            sents = [s for s in FX.split_sentences(text) if len(s) > 25]
            refusal = ev["mode"] == "нет в базе" or text.strip().startswith("В базе знаний нет")
            with_link = [s for s in sents if cited(s)]
            bad = [n for s in sents for n in cited(s) if n not in ev["sources"]]
            rows.append({"question": q, "mode": ev["mode"], "refusal": refusal,
                         "sentences": len(sents), "with_link": len(with_link),
                         "links": sum(len(cited(s)) for s in sents), "bad_links": len(bad),
                         "removed_by_check": ev["unsupported"], "error": ev["error"]})
            print(f"  чат: {q[:60]!r} — предложений {len(sents)}, со ссылкой {len(with_link)}, "
                  f"убрано сверкой {ev['unsupported']}", flush=True)
    with (RES / "chat.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    answered = [r for r in rows if not r["refusal"] and not r["error"]]
    sents = sum(r["sentences"] for r in answered)
    links = sum(r["links"] for r in answered)
    return {
        "вопросов": len(rows),
        "отказов «в базе нет»": share(sum(r["refusal"] for r in rows), len(rows)),
        "ошибок": sum(1 for r in rows if r["error"]),
        "предложений в ответах": sents,
        "предложений со ссылкой на фрагмент": share(sum(r["with_link"] for r in answered), sents),
        "ссылок на несуществующий фрагмент": share(sum(r["bad_links"] for r in answered), links),
        "утверждений убрано сверкой ссылок": sum(r["removed_by_check"] for r in rows),
        "ответов, где сверка что-то убрала": share(
            sum(1 for r in rows if r["removed_by_check"]), len(rows)),
    }


# --------------------------------------------------------------------------- отчёт
def write_report(summary: dict[str, Any]) -> None:
    lines = ["# Прослеживаемость, прозрачность, обоснованность", "",
             f"База: {summary['база']}; запуск {summary['время']}; "
             f"вопросы к цитатам: {summary['модель вопросов'] or 'не задавались'}.", ""]

    def dump(obj: Any, depth: int = 0) -> None:
        pad = "  " * depth
        for k, v in obj.items():
            if isinstance(v, dict) and not {"k", "n"} <= set(v):
                lines.append(f"{pad}- **{k}**:")
                dump(v, depth + 1)
            elif isinstance(v, dict):
                lines.append(f"{pad}- {k}: {v['k']} из {v['n']} ({v['pct']} %)")
            else:
                lines.append(f"{pad}- {k}: {v}")

    for title, key in [("1. Прослеживаемость: все связи", "прослеживаемость"),
                       ("1а. Прослеживаемость: только связи графа", "прослеживаемость (граф)"),
                       ("2. Территория", "территория"),
                       ("3. Прозрачность: что решала модель, что отбросил код", "прозрачность"),
                       ("4. Обоснованность: подтверждает ли цитата связь", "обоснованность"),
                       ("5. Насколько верить проверке 4: 120 связей с ручной разметкой", "проверка на разметке"),
                       ("6. Ответы чат-бота", "чат-бот")]:
        if key in summary:
            lines += [f"## {title}", ""]
            dump(summary[key])
            lines.append("")
    (RES / "report.md").write_text("\n".join(lines), "utf-8")


def main() -> None:
    from georag.llm import BUILDER_HOST, BUILDER_MODEL

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", default=os.environ.get("GEORAG_DSN", DEFAULT_DSN))
    ap.add_argument("--host", default=BUILDER_HOST, help="модель для четырёх вопросов")
    ap.add_argument("--model", default=BUILDER_MODEL)
    ap.add_argument("--no-llm", action="store_true", help="без вопросов модели, только код")
    ap.add_argument("--sample", type=int, default=0,
                    help="вопросы модели — по случайной выборке N связей (0 — все)")
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--chat", type=int, default=0, help="сколько вопросов задать чат-боту")
    ap.add_argument("--only-chat", action="store_true", help="только чат-бот, граф не трогать")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    use_llm = not args.no_llm
    summary: dict[str, Any] = {"база": re.sub(r"//[^@]*@", "//", args.dsn),
                               "время": time.strftime("%Y-%m-%d %H:%M"),
                               "модель вопросов": f"{args.model} ({args.host})" if use_llm else None}
    old = RES / "summary.json"
    if args.only_chat and old.exists():
        summary = {**json.loads(old.read_text("utf-8")), **{"время": summary["время"]}}

    if not args.only_chat:
        print("Читаю граф…", flush=True)
        with psycopg.connect(read_only(args.dsn)) as conn:
            fs = load_facts(conn)
            print(f"  связей {len(fs)}, в графе {sum(f['in_graph'] for f in fs)}", flush=True)
            for f in fs:
                f["locate"] = locate(f["quote"], f["chunk_text"])
            summary["прослеживаемость"] = trace_block(fs)
            summary["прослеживаемость (граф)"] = trace_block([f for f in fs if f["in_graph"]])
            print("Территория…", flush=True)
            summary["территория"] = territory(fs)
            summary["прозрачность"] = transparency(conn, fs)
        print("Проверка по цитате: код…", flush=True)
        for f in fs:
            f["code_reasons"] = code_check(f)
        pool = fs
        if use_llm:
            pool = [f for f in fs if not f["code_reasons"]]
            if args.sample and args.sample < len(pool):
                pool = random.Random(args.seed).sample(pool, args.sample)
            print(f"Проверка по цитате: вопросы модели, связей {len(pool)}…", flush=True)
            ask_all(pool, args.host, args.model, RES / "votes_cache.jsonl", args.threads)
        asked = {id(f) for f in pool}
        for f in fs:
            if use_llm and not f["code_reasons"] and id(f) not in asked:
                f["verdict"] = "не проверялась моделью"
            else:
                f["verdict"] = verdict(f, use_llm)
        if use_llm and args.sample:
            # Доля по выборке: связи, отброшенные кодом, входят в неё в той же пропорции.
            k = len(pool) / max(1, sum(1 for f in fs if not f["code_reasons"]))
            rnd = random.Random(args.seed + 1)
            for f in fs:
                if f["code_reasons"] and rnd.random() >= k:
                    f["verdict"] = "не проверялась моделью"
        summary["обоснованность"] = grounding_block(fs, use_llm)
        if use_llm and args.sample:
            summary["обоснованность"]["выборка"] = (
                f"вопросы — по {len(pool)} случайным связям из прошедших код, отброшенные кодом "
                f"взяты в той же доле (seed {args.seed})")
        with (RES / "facts.csv").open("w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(["id", "в графе", "вердикт", "причины кода", "совпало из 4", "цитата найдена",
                        "территория", "территория откуда", "src", "relation", "dst", "quote",
                        "статья", "страницы"])
            for f in fs:
                w.writerow([f["id"], int(f["in_graph"]), f["verdict"], "; ".join(f["code_reasons"]),
                            f.get("llm_votes"), f["locate"], f.get("territory", ""),
                            f.get("territory_from", ""), f["src"], f["relation"], f["dst"],
                            ws(f["quote"]), f["title"], ",".join(map(str, f["pages"] or []))])
        print("Проверка на 120 размеченных связях…", flush=True)
        summary["проверка на разметке"] = gold_block(use_llm, args.host, args.model, args.threads)

    if args.chat:
        print(f"Чат-бот: {args.chat} вопросов…", flush=True)
        summary["чат-бот"] = chat_block(args.chat)

    old.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), "utf-8")
    write_report(summary)
    print((RES / "report.md").read_text("utf-8"))


if __name__ == "__main__":
    main()
