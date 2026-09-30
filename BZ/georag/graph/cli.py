"""Граф знаний: факты из статей, которые выписывает Qwen3 и проверяет код.

    python -m georag.graph.cli build             Qwen3 проходит новые фрагменты
    python -m georag.graph.cli build --redo      всё заново (после правки правил)
    python -m georag.graph.cli build --limit 30  только 30 фрагментов — проба
    python -m georag.graph.cli report            что нашла модель и где ошибалась
    python -m georag.graph.cli dataset           сущности, у которых больше всего фактов
    python -m georag.graph.cli dataset --name "Анабарский щит"   датасет и файл CSV
    python -m georag.graph.cli synonyms          предложения в словарь синонимов
    python -m georag.graph.cli questions         вопросы оценки из фактов (config/eval-graph.yaml)
    python -m georag.graph.cli gaps              темы для добычи по пробелам (config/topics-graph.yaml)

Факт — «от — связь — к» с дословной цитатой (подробно — в facts.py). Разные
написания одного и того же сводит словарь config/synonyms.yaml (synonyms.py):
его правка действует сразу, модель заново проходить не нужно.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common import add_db_args, add_embedder_args, add_llm_args
from ..index import db
from . import api, facts, store
from .synonyms import PROPOSALS_PATH, SynonymsError, judge_pairs, load, proposals_text, suggest

ROOT_DIR = Path(__file__).resolve().parents[2]


def _llm(args):
    from ..llm import Ollama

    return Ollama(model=args.model, host=args.ollama_host, timeout=args.timeout)


def _build(conn, args) -> int:
    llm = _llm(args)
    status = llm.status()
    if not status["ok"]:
        print(f"Qwen3 недоступна: {status['error']}\nГраф строит модель — запустите Ollama и "
              "снова python georag.py graph.", file=sys.stderr)
        return 1
    try:
        facts.load_rules(args.rules)
    except (OSError, ValueError) as exc:
        print(f"Правила для модели не читаются: {exc}", file=sys.stderr)
        return 1
    if args.redo:
        store.clear(conn)
        conn.commit()
        print("Прошлые факты убраны — модель проходит всё заново.")
    left = len(store.todo(conn))
    if not left:
        print("Новых фрагментов нет — модель уже прошла всё "
              "(заново: python georag.py graph --redo).")
        return 0
    todo = min(left, args.limit) if args.limit else left
    print(f"Факты выписывает {args.model} по правилам config/graph-rules.txt, каждый проверяет "
          f"код. Фрагментов: {todo}. Можно прервать Ctrl+C — продолжится с этого места.")
    stats = facts.run(conn, llm, limit=args.limit, rules_path=args.rules)
    print(f"\nПройдено фрагментов: {stats['done']} из {stats['todo']}. Фактов принято: "
          f"{stats['facts']}; отброшено проверкой: {stats['rejected']}."
          + (f" Модель не ответила: {stats['errors']}." if stats["errors"] else ""))
    print("Что нашла модель и где ошибалась: python georag.py graph --new\n"
          "Свести разные написания одного и того же: python georag.py synonyms")
    return 2 if stats["stopped"] else 0


def _report(conn, syn) -> int:
    data = api.report(conn, syn)
    p = data["pass"]
    if not p["passed"]:
        print("Модель ещё не выписывала факты: python georag.py graph (нужна Ollama).")
        return 0
    print(f"\nМодель прошла фрагментов: {p['passed']} из {p['chunks']}; фактов: {p['facts']}; "
          f"отброшено проверкой: {p['rejected']}. Сущностей: {data['entities_total']}.")
    print(f"\n{'сущность':<46}{'статей':>7}{'фактов':>8}")
    for name, docs, count, known, spellings in data["entities"]:
        mark = "  (словарь)" if known else ""
        extra = f"  ← {', '.join(spellings)}" if spellings else ""
        print(f"  {name[:44]:<44}{docs:>7}{count:>8}{mark}{extra}"[:160])
    print("\nСвязи (сколько фактов):")
    for relation, count in data["relations"]:
        print(f"  {count:>5}  {relation}")
    if data["rejected"]:
        print("\nПочему код отбрасывал факты (чаще всего):")
        for reason, count in data["rejected"]:
            print(f"  {count:>5}  {reason[:90]}")
    print("\nРазные написания одного и того же сводит config\\synonyms.yaml; предложения "
          "для него: python georag.py synonyms")
    return 0


def _dataset(conn, syn, args) -> int:
    if not args.name:
        data = api.datasets_response(conn, syn, limit=40)
        if not data["datasets"]:
            print("Фактов пока нет: python georag.py graph (нужна Ollama).")
            return 0
        print(f"\nСущностей с фактами: {data['total']}. Больше всего фактов у:\n")
        print(f"{'сущность':<50}{'статей':>7}{'фактов':>8}")
        for r in data["datasets"]:
            print(f"{r['name'][:48]:<50}{r['documents']:>7}{r['facts']:>8}")
        print('\nФакты о любой из них: python georag.py facts "Анабарский щит"')
        return 0
    try:
        data = api.dataset_response(conn, syn, {"name": args.name})
    except api.RequestError as exc:
        near = ", ".join(exc.known[:15])
        print(f"{exc.message}: «{exc.detail}». Больше всего фактов у: {near}", file=sys.stderr)
        return 1
    inside = f" (вместе с: {', '.join(data['includes'])})" if data["includes"] else ""
    print(f"\nФакты: {data['name']}{inside} — статей {data['documents']}, "
          f"фактов {len(data['facts'])}\n")
    for r in data["facts"]:
        line = (f"  {r['about']} — {r['relation']} — {r['other']}" if r["direction"] == "→"
                else f"  {r['other']} — {r['relation']} — {r['about']}")
        print(f"{line[:110]:<112}статей {r['documents']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in data["name"])[:80]
    path = out / f"факты-{safe}.csv"
    path.write_text(api.dataset_csv(data)[1:], encoding="utf-8-sig")
    print(f"\nТаблица для Excel: {path}")
    return 0


def _synonyms(conn, syn, args) -> int:
    from ..index.embed import build_embedder

    g = api.graph(conn, syn)
    if not g.entities:
        print("Фактов пока нет: python georag.py graph (нужна Ollama).")
        return 0
    llm = _llm(args)
    status = llm.status()
    if not status["ok"]:
        print(f"Qwen3 недоступна: {status['error']}", file=sys.stderr)
        return 1
    print("Загружаю модель векторов…")
    embedder = build_embedder(args.embedder, device=args.device)
    names = {name: (name, e["facts"]) for name, e in g.entities.items()}
    relations = Counter_relations(g)
    print(f"Ищу похожие среди {len(names)} имён и {len(relations)} связей; Qwen3 решает, "
          "одно ли это…")
    name_groups = suggest(names, syn, embedder.encode, lambda p: judge_pairs(llm, p),
                          threshold=args.threshold)
    rel_groups = suggest(relations, _relations_as_names(syn), embedder.encode,
                         lambda p: judge_pairs(llm, p), threshold=max(args.threshold, 0.85))
    PROPOSALS_PATH.write_text(proposals_text(name_groups, rel_groups), encoding="utf-8")
    print(f"\nПредложений: имён {len(name_groups)}, связей {len(rel_groups)}.\n"
          f"Файл: {PROPOSALS_PATH}\nПросмотрите Блокнотом и перенесите нужное в "
          f"config\\synonyms.yaml — действует сразу, модель заново не нужна.")
    return 0


def _questions(conn, syn, args) -> int:
    from . import questions

    g = api.graph(conn, syn)
    if not g.links:
        print("Фактов пока нет: python georag.py graph (нужна Ollama).")
        return 0
    llm = None
    if not args.no_llm:
        llm = _llm(args)
        status = llm.status()
        if not status["ok"]:
            print(f"Qwen3 недоступна ({status['error']}) — вопросы по шаблону.")
            llm = None
    print(f"Составляю вопросы из фактов графа ({'Qwen3 переформулирует' if llm else 'по шаблону'})…")
    items = questions.build(g, llm, limit=args.limit or 30)
    out = Path(args.out_questions)
    out.write_text(questions.to_yaml(items), encoding="utf-8")
    docs = len({d for item in items for d in item["статьи"]})
    print(f"\nВопросов: {len(items)} по {docs} статьям. Файл: {out}\n"
          f"Оценка по ним: python georag.py eval --questions {out.relative_to(ROOT_DIR)}")
    return 0


def _gaps(conn, syn, args) -> int:
    from . import gaps

    rows = gaps.find(api.graph(conn, syn), syn, limit=args.limit or 25)
    out = Path(args.out_topics)
    out.write_text(gaps.to_yaml(rows), encoding="utf-8")
    if not rows:
        print("Пробелов не нашлось: всё из словаря есть в фактах, и подтверждено не одной статьёй.")
        return 0
    print(f"\nПробелов: {len(rows)}\n")
    for row in rows:
        print(f"  {row['name'][:50]:<52}{row['why']}")
    print(f"\nТемы для добычи: {out}\nДобыть: python georag.py all --topics {out.name}")
    return 0


def Counter_relations(g) -> dict[str, tuple[str, int]]:  # noqa: N802 — читается как таблица
    count: dict[str, int] = {}
    for link in g.links.values():
        count[link.relation] = count.get(link.relation, 0) + len(link.facts)
    return {r: (r, n) for r, n in count.items()}


def _relations_as_names(syn):
    """Для предложений по связям — словарь, где «имена» это связи."""
    from .synonyms import Synonyms

    groups = {canon: [] for canon in set(syn.relations.values())}
    return Synonyms(names={}, groups=groups)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Граф знаний ГеоRAG: факты из статей")
    parser.add_argument("command", choices=["build", "report", "dataset", "synonyms",
                                            "questions", "gaps"])
    parser.add_argument("--no-llm", action="store_true",
                        help="вопросы по шаблону, без модели (questions)")
    parser.add_argument("--out-questions", default=str(ROOT_DIR / "config" / "eval-graph.yaml"))
    parser.add_argument("--out-topics", default=str(ROOT_DIR / "config" / "topics-graph.yaml"))
    parser.add_argument("--name", default=None, help="чей датасет (dataset)")
    parser.add_argument("--territory", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default="data/datasets", help="куда класть таблицу датасета")
    parser.add_argument("--redo", action="store_true", help="модель проходит всё заново")
    parser.add_argument("--limit", type=int, default=None, help="сколько фрагментов пройти")
    parser.add_argument("--rules", default=None, help="другой файл правил для модели")
    parser.add_argument("--synonyms", default=None, help="другой словарь синонимов")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="насколько похожи имена, чтобы спросить модель (synonyms)")
    parser.add_argument("--timeout", type=int, default=300, help="сколько ждать модель, с")
    add_db_args(parser)
    add_embedder_args(parser)
    add_llm_args(parser)
    args = parser.parse_args(argv)
    args.name = args.name or args.territory

    try:
        syn = load(args.synonyms)
    except SynonymsError as exc:
        print(f"Словарь синонимов: {exc}", file=sys.stderr)
        return 1
    for warning in syn.warnings:
        print(f"Словарь синонимов, предупреждение: {warning}")

    try:
        with db.connect(args.dsn) as conn:
            store.init(conn)
            if args.command == "build":
                return _build(conn, args)
            if args.command == "report":
                return _report(conn, syn)
            if args.command == "dataset":
                return _dataset(conn, syn, args)
            if args.command == "questions":
                return _questions(conn, syn, args)
            if args.command == "gaps":
                return _gaps(conn, syn, args)
            return _synonyms(conn, syn, args)
    except KeyboardInterrupt:
        print("\nПрервано. Сделанное сохранено; та же команда продолжит с этого места.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
