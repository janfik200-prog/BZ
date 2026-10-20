"""Оценка базы знаний и чат-бота по набору вопросов с ожидаемым результатом.

    python -m georag.evaluation                     поиск и ответы (нужна модель)
    python -m georag.evaluation --only-search       только поиск
    python -m georag.evaluation --only s01,t01      отдельные вопросы

Вопросы — config/eval-questions.yaml. Для каждого проверяется две вещи.

**Поиск.** Нашлись ли близкие фрагменты (те же, что получит чат-бот) и есть
ли в них ключевые слова вопроса. Для контрольных вопросов не по теме —
наоборот: не нашлось ничего.

**Ответ.** Отвечал ли бот по базе, когда должен, и сказал ли «в базе нет»,
когда должен. Длина — ответ развёрнутый, а не одна фраза. Ссылки: сколько
утверждений со ссылкой на фрагмент и нет ли ссылок на несуществующие.
Ключевые слова — в самом ответе.

Модель не оценивает сама себя: всё считается кодом по тексту. Поэтому оценка
грубая — она не скажет, верен ли ответ по существу. Зато она одинаковая от
прогона к прогону, и по ней видно, что изменение сделало лучше или хуже:
отчёт сравнивается с прошлым прогоном.

**Ключевые слова.** Строка в «ключевые» — одна мысль, варианты через «|», и
засчитывается любой: «гематит|hematite» — ответ по-русски, статья по-английски.
Слово ищется в любой форме и с начала слова: у русского снимается окончание
(«кварцевые жилы» находит «кварцевыми жилами»), у английского — множественное
число («valleys» находит «valley», «anomalies» — «anomaly»). Аббревиатура и
символ элемента — только целым словом: ROC не находится в «rock», Au — в «author».
Перевода нет, но имя собственное одним словом узнаётся и другим алфавитом или
другой транслитерацией («Kular» находит «район Кулар», «Koldar» — «массив Кольдар»;
text.translit_key) — кроме начала составного имени («Куларо-Нерский»). Обычное слово —
только как написано: русское написание термина дают вариантом.

**Узкий вопрос** — «кратко: да» в файле вопросов (так помечает свои вопросы
graph/questions.py): спрашивается один факт, и верный ответ может быть одной
фразой («рутил [1]»). Порог длины MIN_CHARS для него не действует — вместо него
ответ должен сослаться хотя бы на один фрагмент. Без пометки вопрос обзорный, и
ответ в одну фразу — отписка.

**Номер из вопроса.** «[13]» в самом вопросе («согласно источнику [13]») ответ
может повторить; это не ссылка на выдуманный фрагмент и ошибкой не считается.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .chat import answer as chat
from .chat.checks import citations
from .chat.dataset import collect_dataset, graph_candidates
from .chat.plan import plan_question
from .chat.retrieval import judge_fragments, run
from .chat.settings import Settings
from .paths import CONFIG, LOGS
from .text import keyword_hits

DEFAULT_QUESTIONS = CONFIG / "eval-questions.yaml"
DEFAULT_LOGS = LOGS
IN_BASE = "база"
NOT_IN_BASE = "без базы"

# Пороги «прошёл / не прошёл». Подобраны так, чтобы хороший ответ проходил
# с запасом, а отписка — нет. Менять — здесь.
MIN_KEYWORDS = 0.5  # доля ключевых мыслей, найденных в тексте
MIN_COVERAGE = 0.6  # доля утверждений ответа со ссылкой
MIN_CHARS = 150  # ответ по базе — не одна фраза (кроме узких вопросов, «кратко: да»)
MIN_DOCS = 0.5  # доля нужных статей (поле «статьи»), найденных поиском


class QuestionsError(ValueError):
    """Файл вопросов заполнен так, что по нему нельзя работать."""


@dataclass
class Question:
    id: str
    text: str
    expect: str
    keywords: list[list[str]] = field(default_factory=list)
    concept: str = ""
    territory: str = ""
    method: str = ""
    # Какие статьи поиск обязан найти (doc_id, как в базе). Ключевые слова говорят
    # только, что в найденном есть нужные слова; статьи — что найдено именно то.
    docs: list[str] = field(default_factory=list)
    # Узкий вопрос об одном факте («кратко: да»): короткий ответ — не отписка.
    short: bool = False


def _yes(value: Any) -> bool:
    """«да» в файле вопросов: да, yes, true (YAML сам читает true как True)."""
    return value is True or str(value or "").strip().lower() in {"да", "yes", "true"}


def load_questions(path: Path | str = DEFAULT_QUESTIONS) -> list[Question]:
    import yaml

    path = Path(path)
    if not path.exists():
        raise QuestionsError(f"нет файла вопросов: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    raw = data.get("вопросы")
    if not isinstance(raw, list) or not raw:
        raise QuestionsError(f"{path.name}: раздел «вопросы» пуст")
    out, seen = [], set()
    for item in raw:
        qid = str(item.get("id") or "").strip()
        text = str(item.get("вопрос") or "").strip()
        expect = str(item.get("ожидание") or IN_BASE).strip()
        if not qid or not text:
            raise QuestionsError(f"у вопроса нет id или текста: {item!r}")
        if qid in seen:
            raise QuestionsError(f"id «{qid}» повторяется")
        if expect not in (IN_BASE, NOT_IN_BASE):
            raise QuestionsError(f"{qid}: ожидание «{expect}» — допустимо «база» или «без базы»")
        seen.add(qid)
        groups = [
            [w.strip() for w in str(line).split("|") if w.strip()]
            for line in (item.get("ключевые") or [])
        ]
        docs = item.get("статьи") or []
        out.append(
            Question(
                qid,
                text,
                expect,
                [g for g in groups if g],
                str(item.get("понятие") or ""),
                str(item.get("территория") or ""),
                str(item.get("метод") or ""),
                [
                    str(d).strip()
                    for d in (docs if isinstance(docs, list) else [docs])
                    if str(d).strip()
                ],
                _yes(item.get("кратко")),
            )
        )
    return out


# Ключевые слова ищет text.keyword_hits (правила — «Ключевые слова» вверху): тем же
# сопоставлением генератор вопросов (graph/questions.py) проверяет свои ключи.


def share(found: int, total: int) -> float | None:
    return round(found / total, 2) if total else None


# --------------------------------------------------------------------------- #
#  Прогон одного вопроса
# --------------------------------------------------------------------------- #
def check_search(q: Question, sources: list[dict[str, Any]]) -> dict[str, Any]:
    text = " ".join(f"{s.get('title', '')} {s.get('text', '')}" for s in sources)
    found, missing = keyword_hits(q.keywords, text)
    recall = share(found, len(q.keywords))
    got = {s.get("doc_id") for s in sources}
    docs_missing = [d for d in q.docs if d not in got]
    docs_recall = share(len(q.docs) - len(docs_missing), len(q.docs))
    if q.expect == IN_BASE:
        ok = (
            bool(sources)
            and (recall is None or recall >= MIN_KEYWORDS)
            and (docs_recall is None or docs_recall >= MIN_DOCS)
        )
    else:
        ok = not sources
    return {
        "found": len(sources),
        "keywords": recall,
        "missing": missing,
        "ok": ok,
        "docs": docs_recall,
        "docs_missing": docs_missing,
        "sources": [
            {
                "n": s["n"],
                "chunk_id": s.get("chunk_id"),  # нашёлся ли фрагмент с фактом (поле «фрагмент»)
                "title": s.get("title", "")[:90],
                "year": s.get("year"),
                "similarity": s.get("similarity"),
                "found_by": s.get("found_by"),
            }
            for s in sources
        ],
    }


def check_answer(q: Question, events: list[dict[str, Any]]) -> dict[str, Any]:
    text = ""
    for event in events:
        if event["type"] in ("revised", "general"):
            text = event.get("text", "") if event["type"] == "revised" else ""
        elif event["type"] == "token":
            text += event["text"]
    done = next((e for e in events if e["type"] == "done"), {})
    error = next((e["error"] for e in events if e["type"] == "error"), None)
    mode = done.get("mode")
    cited, claims = done.get("coverage") or (0, 0)
    found, missing = keyword_hits(q.keywords, text)
    recall = share(found, len(q.keywords))
    coverage = share(cited, claims)
    # Номер «[n]» из самого вопроса ответ повторяет — это не выдуманный фрагмент.
    asked = set(citations(q.text, 0)[1])
    unknown = [n for n in done.get("unknown", []) if n not in asked]
    problems = []
    if error:
        problems.append(f"ошибка: {error}")
    elif q.expect == IN_BASE:
        if mode != chat_mode(IN_BASE):
            problems.append("ответил без базы, хотя должен был по статьям")
        else:
            if not done.get("used"):
                problems.append("нет ссылки на фрагмент")
            if not q.short and len(text) < MIN_CHARS:
                problems.append(f"короткий ответ: {len(text)} знаков")
            if coverage is not None and coverage < MIN_COVERAGE:
                problems.append(f"мало ссылок: {cited} из {claims} утверждений")
            if unknown:
                problems.append(f"ссылки на несуществующие фрагменты {unknown}")
            if recall is not None and recall < MIN_KEYWORDS:
                problems.append("в ответе нет ключевого: " + ", ".join(missing))
    elif mode != chat_mode(NOT_IN_BASE):
        problems.append("ответил по базе на вопрос не по теме")
    return {
        "mode": mode,
        "chars": len(text),
        "used": done.get("used", []),
        "unknown": unknown,
        "coverage": coverage,
        "claims": [cited, claims],
        "keywords": recall,
        "missing": missing,
        "seconds": done.get("seconds"),
        "ok": not problems,
        "problems": problems,
        "text": text,
    }


def chat_mode(expect: str) -> str:
    """Как называет режим сам чат-бот (поле mode в событии done)."""
    return "база" if expect == IN_BASE else "без базы"


def run_question(
    conn: Any,
    embedder: Any,
    q: Question,
    settings: Settings,
    *,
    with_answer: bool,
    planner: Any = None,
    llm: Any = None,
    judge: Any = judge_fragments,
    datasets: Any = collect_dataset,
    graph_hits: Any = graph_candidates,
) -> dict[str, Any]:
    """Один вопрос: поиск — тот же, что у чат-бота (разбор, отбор моделью, датасет),
    и по найденному — ответ. Ищется один раз и для оценки поиска, и для ответа.
    datasets=None и graph_hits=None — без графа (--no-graph)."""
    planner = planner or plan_question
    started = time.monotonic()
    prepared = run(
        chat.prepare(conn, embedder, q.text, [], settings, planner, judge, datasets, graph_hits)
    )
    found = prepared.retrieval
    result = {
        "id": q.id,
        "question": q.text,
        "expect": q.expect,
        "concept": q.concept,
        "territory": q.territory,
        "method": q.method,
        "queries": prepared.plan.queries + prepared.plan.extra,
        "rounds": found.rounds,
        "judged": found.judged,
        "search": check_search(q, prepared.sources),
        "search_seconds": round(time.monotonic() - started, 1),
    }
    if with_answer:
        events = list(
            chat.answer(
                conn,
                embedder,
                q.text,
                [],
                settings,
                llm=llm or chat.stream_llm,
                prepared=prepared,
            )
        )
        result["answer"] = check_answer(q, events)
    return result


# --------------------------------------------------------------------------- #
#  Сводка и отчёт
# --------------------------------------------------------------------------- #
def _avg(values: list[Any]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    base = [r for r in results if r["expect"] == IN_BASE]
    control = [r for r in results if r["expect"] == NOT_IN_BASE]
    out = {
        "questions": len(results),
        "search_ok": share(sum(r["search"]["ok"] for r in results), len(results)),
        "search_found": share(sum(bool(r["search"]["found"]) for r in base), len(base)),
        "search_keywords": _avg([r["search"]["keywords"] for r in base]),
        "search_docs": _avg([r["search"].get("docs") for r in base]),
        "control_clean": share(sum(r["search"]["ok"] for r in control), len(control)),
    }
    answered = [r for r in results if "answer" in r]
    if answered:
        base_a = [r for r in answered if r["expect"] == IN_BASE]
        control_a = [r for r in answered if r["expect"] == NOT_IN_BASE]
        in_base = [r for r in base_a if r["answer"]["mode"] == "база"]
        out.update(
            {
                "answer_ok": share(sum(r["answer"]["ok"] for r in answered), len(answered)),
                "answer_in_base": share(len(in_base), len(base_a)),
                "answer_chars": _avg([r["answer"]["chars"] for r in in_base]),
                "answer_coverage": _avg([r["answer"]["coverage"] for r in in_base]),
                "answer_keywords": _avg([r["answer"]["keywords"] for r in in_base]),
                "answer_unknown": sum(bool(r["answer"]["unknown"]) for r in answered),
                "control_general": share(
                    sum(r["answer"]["mode"] == "без базы" for r in control_a), len(control_a)
                ),
                "answer_seconds": _avg([r["answer"]["seconds"] for r in answered]),
            }
        )
    return out


LABELS = [
    ("search_ok", "вопросов прошло по поиску", "доля"),
    ("search_found", "по теме: нашлись близкие фрагменты", "доля"),
    ("search_keywords", "по теме: ключевого в найденном", "доля"),
    ("search_docs", "по теме: нужных статей найдено (поле «статьи»)", "доля"),
    ("control_clean", "не по теме: поиск ничего не подсунул", "доля"),
    ("answer_ok", "вопросов прошло по ответу", "доля"),
    ("answer_in_base", "по теме: ответ по статьям базы", "доля"),
    ("answer_chars", "средняя длина ответа, знаков", "число"),
    ("answer_coverage", "утверждений со ссылкой на фрагмент", "доля"),
    ("answer_keywords", "ключевого в самом ответе", "доля"),
    ("answer_unknown", "ответов со ссылкой на несуществующий фрагмент", "число"),
    ("control_general", "не по теме: бот сказал «в базе нет»", "доля"),
    ("answer_seconds", "среднее время ответа, с", "число"),
]


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    return f"{value:.0%}" if kind == "доля" else f"{value:g}"


def _delta(now: Any, before: Any, kind: str) -> str:
    if now is None or before is None or now == before:
        return ""
    diff = now - before
    text = f"{diff:+.0%}" if kind == "доля" else f"{diff:+g}"
    return f" ({text} к прошлому)"


def by_group(results: list[dict[str, Any]], key: str) -> list[tuple[str, int, float | None]]:
    """Средняя доля ключевого в найденном — по понятию, территории или методу."""
    groups: dict[str, list[Any]] = {}
    for r in results:
        if r.get(key):
            groups.setdefault(r[key], []).append(r["search"]["keywords"])
    return sorted(
        ((name, len(v), _avg(v)) for name, v in groups.items()),
        key=lambda x: (x[2] if x[2] is not None else -1),
    )


def report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    previous: dict[str, Any] | None,
    meta: dict[str, Any],
) -> str:
    lines = [f"# Оценка базы знаний — {meta['when']}", ""]
    lines.append(
        f"Вопросов: {summary['questions']} · в базе: {meta['documents']} статей, "
        f"{meta['chunks']} фрагментов · модель: {meta['model']}"
        + ("" if meta["with_answer"] else " · только поиск")
    )
    if not meta.get("planner"):
        lines.append("")
        lines.append(
            "Модель не отвечала — поиск шёл только по самому вопросу и отбирал по "
            "порогу близости, без модели. С моделью найдётся больше и точнее."
        )
    lines += ["", "## Итог", "", "| показатель | значение |", "|---|---|"]
    for key, label, kind in LABELS:
        if key in summary:
            before = (previous or {}).get(key)
            lines.append(
                f"| {label} | {_fmt(summary[key], kind)}{_delta(summary[key], before, kind)} |"
            )

    weak = [g for g in by_group(results, "concept") if g[2] is not None and g[2] < MIN_KEYWORDS]
    if weak:
        lines += [
            "",
            "## Где базе не хватает статей",
            "",
            "Понятия, по которым в найденном меньше половины ключевого. Добрать статьи: "
            "`python georag.py all --topics topics-check.yaml`.",
            "",
        ]
        lines += [f"- {name} — {v:.0%} (вопросов: {n})" for name, n, v in weak]

    lines += [
        "",
        "## По вопросам",
        "",
        "| id | вопрос | найдено | ключевое в найденном | ответ | длина | ссылки | итог |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["search"]
        a = r.get("answer")
        verdict_ok = s["ok"] and (a is None or a["ok"])
        cells = [r["id"], r["question"][:70], str(s["found"]), _fmt(s["keywords"], "доля")]
        if a:
            cells += [
                a["mode"] or "ошибка",
                str(a["chars"]),
                f"{a['claims'][0]}/{a['claims'][1]}" if a["claims"][1] else "—",
            ]
        else:
            cells += ["—", "—", "—"]
        cells.append("да" if verdict_ok else "нет")
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")

    failed = [r for r in results if not (r["search"]["ok"] and r.get("answer", {"ok": True})["ok"])]
    if failed:
        lines += ["", "## Что не так", ""]
        for r in failed:
            lines.append(f"**{r['id']}. {r['question']}**")
            s = r["search"]
            if not s["ok"]:
                if r["expect"] == IN_BASE and not s["found"]:
                    lines.append(
                        "- поиск: близких фрагментов нет — в базе нет статей по теме "
                        "или порог слишком строгий"
                    )
                elif r["expect"] == IN_BASE:
                    if s["missing"]:
                        lines.append(
                            "- поиск: в найденном нет ключевого — " + ", ".join(s["missing"])
                        )
                    if s.get("docs_missing"):
                        lines.append(
                            "- поиск: не найдены нужные статьи — " + ", ".join(s["docs_missing"])
                        )
                else:
                    lines.append(f"- поиск: на вопрос не по теме нашлось {s['found']} фрагментов")
            for problem in r.get("answer", {}).get("problems", []):
                lines.append(f"- ответ: {problem}")
            lines.append("- искал: " + " · ".join(f"«{x}»" for x in r["queries"]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def previous_run(log_dir: Path, before: str) -> dict[str, Any] | None:
    """Сводка прошлого прогона — из последнего eval-*.json в папке логов."""
    runs = sorted(p for p in log_dir.glob("eval-*.json") if p.stem < f"eval-{before}")
    for path in reversed(runs):
        try:
            summary: dict[str, Any] | None = json.loads(path.read_text(encoding="utf-8")).get(
                "summary"
            )
            return summary
        except (OSError, json.JSONDecodeError):
            continue
    return None


# --------------------------------------------------------------------------- #
#  Запуск
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    from .common import add_db_args, add_embedder_args, add_llm_args
    from .index import db
    from .index.embed import build_embedder

    parser = argparse.ArgumentParser(description="Оценка базы знаний и чат-бота")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--only-search", action="store_true", help="без ответов модели")
    parser.add_argument("--only", default="", help="id вопросов через запятую")
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="без графа: ни датасетов «что известно о…», ни фрагментов о предмете из графа",
    )
    parser.add_argument(
        "--facts",
        action="store_true",
        help="с поиском по векторам фактов графа (фрагменты, которые привели факты)",
    )
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    add_db_args(parser)
    add_embedder_args(parser)
    add_llm_args(parser)
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.questions)
    except QuestionsError as exc:
        print(f"Вопросы не читаются: {exc}", file=sys.stderr)
        return 1
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        questions = [q for q in questions if q.id in wanted]
        if not questions:
            print(f"Нет вопросов с id {', '.join(sorted(wanted))}", file=sys.stderr)
            return 1

    settings = Settings(model=args.model, host=args.llm_host, fact_search=args.facts)
    status = chat.llm_status(settings.host, settings.model)
    with_answer = not args.only_search
    if with_answer and not status["ok"]:
        print(f"{status['error']}\nОценю только поиск (--only-search).", file=sys.stderr)
        with_answer = False
    planner = plan_question if status["ok"] else (lambda *_: [])
    judge = judge_fragments if status["ok"] else None

    print("Загружаю модель эмбеддингов…", file=sys.stderr)
    embedder = build_embedder(device=args.device)
    results = []
    with db.connect(args.dsn) as conn:
        info = db.stats(conn)
        for i, q in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {q.id}: {q.text[:70]}", file=sys.stderr)
            graph = {"datasets": None, "graph_hits": None} if args.no_graph else {}
            r = run_question(
                conn,
                embedder,
                q,
                settings,
                with_answer=with_answer,
                planner=planner,
                judge=judge,
                **graph,
            )
            results.append(r)
            s, a = r["search"], r.get("answer")
            line = (
                f"    поиск: {s['found']} фрагм., кругов {r['rounds']}, "
                f"ключевого {_fmt(s['keywords'], 'доля')}"
            )
            if a:
                line += f"; ответ: {a['mode']}, {a['chars']} зн." + (
                    "" if a["ok"] else " — " + "; ".join(a["problems"])
                )
            print(line, file=sys.stderr)

    when = datetime.now().strftime("%Y%m%d-%H%M")
    args.logs.mkdir(parents=True, exist_ok=True)
    previous = previous_run(args.logs, when)
    summary = summarize(results)
    meta = {
        "when": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "documents": info["documents"],
        "chunks": info["chunks"],
        "model": settings.model,
        "with_answer": with_answer,
        "planner": status["ok"],
        "graph": not args.no_graph,
        "facts": settings.fact_search,
    }
    md = report(results, summary, previous, meta)
    (args.logs / f"eval-{when}.md").write_text(md, encoding="utf-8")
    (args.logs / f"eval-{when}.json").write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "results": results}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )

    print()
    for key, label, kind in LABELS:
        if key in summary:
            print(
                f"  {label}: {_fmt(summary[key], kind)}"
                f"{_delta(summary[key], (previous or {}).get(key), kind)}"
            )
    print(f"\nОтчёт: {args.logs / f'eval-{when}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
