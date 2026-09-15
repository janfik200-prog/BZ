"""Построение графа сущностей по тому, что уже лежит в базе.

    python -m georag.graph.cli build          перестроить граф целиком (правила)
    python -m georag.graph.cli build --llm    добавить разбор моделью: имена в любом
                                             порядке слов и названные связи
    python -m georag.graph.cli top            что чаще всего упоминается
    python -m georag.graph.cli near "Анабарский щит"    с чем встречается вместе

Сеть и модели не нужны: граф строится по текстам, которые уже в базе.
Перестроение идёт с чистого листа — правила извлечения могли измениться,
и старые сущности иначе не убрать.
"""

from __future__ import annotations

import argparse
import sys

from ..index import db
from . import store
from .extract import extract
from .llm_extract import extract_with_llm


def build(conn, verbose: bool = True, llm=None) -> dict:
    store.init(conn)
    store.clear(conn)

    rows = store.chunks_for_graph(conn)
    if not rows:
        print("В базе нет чанков — сначала ingest", file=sys.stderr)
        return {}

    seen_docs = set()
    for i, (chunk_id, doc_id, text) in enumerate(rows, start=1):
        for entity in extract(text or ""):
            store.save(conn, entity, doc_id, chunk_id)

        # Модель — вторым проходом поверх правил. Сущности схлопываются по общему
        # ключу, а связи с названным типом правилам вообще недоступны.
        if llm is not None:
            entities, relations = extract_with_llm(llm, text or "")
            for entity in entities:
                store.save(conn, entity, doc_id, chunk_id)
            for relation in relations:
                store.save_relation(conn, relation, doc_id, chunk_id)

        seen_docs.add(doc_id)
        if verbose and (i % 10 == 0 or llm is not None):
            print(f"  обработано фрагментов: {i}/{len(rows)}", end="\r")
    conn.commit()
    if verbose:
        print(" " * 50, end="\r")

    info = store.counts(conn)
    info["chunks"] = len(rows)
    info["documents"] = len(seen_docs)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Граф сущностей ГеоRAG")
    parser.add_argument("command", choices=["build", "top", "near"])
    parser.add_argument("name", nargs="?", help="название сущности для команды near")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--kind", choices=["объект", "ископаемое", "метод"], default=None)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--llm", action="store_true", help="дополнить разбор моделью")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args(argv)

    try:
        with db.connect(args.dsn) as conn:
            if args.command == "build":
                llm = None
                if args.llm:
                    from ..acquire.llm import OllamaLLM

                    llm = OllamaLLM(model=args.model, host=args.ollama_host)
                    print(f"Разбор правилами + модель {args.model}. Это небыстро:")
                    print("примерно две-три секунды на фрагмент.")
                info = build(conn, llm=llm)
                if not info:
                    return 1
                kinds = ", ".join(f"{k}: {v}" for k, v in sorted(info["by_kind"].items()))
                print(
                    f"Просмотрено {info['chunks']} фрагментов в {info['documents']} статьях.\n"
                    f"Сущностей: {info['entities']} ({kinds}); упоминаний: {info['mentions']}"
                    + (f"; названных связей: {info['relations']}" if info.get("relations") else "")
                )
                return 0

            store.init(conn)

            if args.command == "top":
                rows = store.top_entities(conn, args.kind, args.limit)
                if not rows:
                    print("Граф пуст. Постройте его: ... graph.cli build")
                    return 0
                print(f"{'сущность':<44}{'вид':<12}{'статей':>7}{'упоминаний':>12}")
                print("-" * 76)
                for row in rows:
                    print(f"{row['name'][:42]:<44}{row['kind']:<12}{row['docs']:>7}{row['mentions']:>12}")
                return 0

            # near
            if not args.name:
                print('Нужно название: ... graph.cli near "Анабарский щит"', file=sys.stderr)
                return 1
            match = next(
                (e for e in store.top_entities(conn, limit=5000)
                 if e["name"].lower() == args.name.lower()
                 or args.name.lower() in e["name"].lower()),
                None,
            )
            if not match:
                print(f"«{args.name}» в графе нет. Посмотрите список: ... graph.cli top",
                      file=sys.stderr)
                return 1

            print(f"\n{match['name']} — упоминается в {match['docs']} статьях\n" + "=" * 70)
            named = store.named_relations(conn, match["key"])
            if named:
                print("\nСвязи из текста:")
                for row in named:
                    arrow = "→" if row["direction"] == "из" else "←"
                    print(f"  {arrow} {row['type']:<18} {row['name'][:40]:<42}"
                          f"в {row['docs']} статьях")

            print("\nВстречается вместе с:")
            for row in store.related(conn, match["key"], args.limit):
                print(f"  {row['name'][:44]:<46}{row['kind']:<12}в {row['together']} статьях")
            print("\nСтатьи:")
            for doc in store.entity_documents(conn, match["key"]):
                print(f"  · {doc['title'][:66]} ({doc['year'] or '—'}) — упоминаний {doc['hits']}")
                print(f"    {doc['url']}")
            return 0

    except ImportError as exc:
        print(f"Не хватает библиотеки: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна или ответила ошибкой: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
