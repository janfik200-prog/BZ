"""Граф базы знаний: разметка фрагментов по словарю и просмотр результата.

    python -m georag.graph.cli build            разметить всё заново (без модели)
    python -m georag.graph.cli build --llm      то же + проверка моделью
    python -m georag.graph.cli verify           только проверка моделью, с места остановки
    python -m georag.graph.cli verify --limit 50    проверить первые 50 — пилот
    python -m georag.graph.cli concepts         сводка по понятиям
    python -m georag.graph.cli evidence "гидротермальные изменения" --territory "Билляхская зона"
    python -m georag.graph.cli top              какие названия чаще всего встречаются
    python -m georag.graph.cli entity "Анабарский щит"   в каких статьях

Что строится, по порядку:

1. названия из статей — правилами по опорным словам («…ский рудный узел»);
2. метки фрагментов — какие территории и методы словаря в них названы;
3. доказательства понятий — какие фрагменты описывают понятия словаря,
   с цитатой (подробно — в concepts.py).

Словарь — vocabulary.yaml в корне проекта. После его правки — снова build.
Решения модели при перестроении сохраняются.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from ..index import db
from . import concepts as tagging
from . import store
from .extract import extract
from .vocabulary import VocabularyError, load


def build(
    conn,
    embedder,
    vocab,
    verbose: bool = True,
    min_similarity: float = tagging.MIN_SIMILARITY,
    min_similarity_terms: float = tagging.MIN_SIMILARITY_TERMS,
    max_per_concept: int = tagging.MAX_PER_CONCEPT,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    store.init(conn)

    rows = store.chunks_for_graph(conn)
    if not rows:
        print("В базе нет фрагментов — сначала ingest", file=sys.stderr)
        return {}

    # 1. Названия из статей — обзор того, что упоминается в корпусе.
    log("Названия из статей (правила)…")
    store.clear_entities(conn)
    for chunk_id, doc_id, text in rows:
        for entity in extract(text or ""):
            store.save(conn, entity, doc_id, chunk_id)

    # 2. Метки: территории и методы словаря во фрагментах.
    log("Территории и методы из словаря…")
    tags = [(chunk_id, kind, name)
            for chunk_id, _, text in rows for kind, name in vocab.tags(text or "")]
    store.replace_tags(conn, tags)

    # 3. Доказательства понятий.
    log(f"Понятия: ищу фрагменты (похожесть от {min_similarity}, "
        f"со словом словаря — от {min_similarity_terms})…")
    candidates, report = tagging.find_candidates(
        conn, embedder, vocab, rows,
        min_similarity=min_similarity,
        min_similarity_terms=min_similarity_terms,
        max_per_concept=max_per_concept,
        log=log,
    )
    sync = store.sync_evidence(conn, candidates, list(vocab.concepts))
    store.set_meta(
        conn,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        vocabulary_sha=vocab.sha,
        min_similarity=min_similarity,
        min_similarity_terms=min_similarity_terms,
        embedder=getattr(embedder, "name", "?"),
    )
    conn.commit()

    info = store.counts(conn)
    info.update(chunks=len(rows), documents=len({r[1] for r in rows}),
                candidates=len(candidates), report=report, **sync)
    return info


def _llm(args):
    from ..acquire.llm import OllamaLLM

    return OllamaLLM(model=args.model, host=args.ollama_host)


def _print_summary(conn, vocab) -> None:
    summary = store.concept_summary(conn)
    print(f"\n{'понятие':<42}{'фрагм.':>7}{'статей':>8}{'подтв.':>8}{'не пров.':>10}{'откл.':>7}")
    print("-" * 82)
    for code in vocab.concepts:
        s = summary.get(code, {})
        print(f"{code[:40]:<42}{s.get('fragments', 0):>7}{s.get('documents', 0):>8}"
              f"{s.get('confirmed', 0):>8}{s.get('unchecked', 0):>10}{s.get('rejected', 0):>7}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Граф базы знаний ГеоRAG")
    parser.add_argument("command",
                        choices=["build", "verify", "concepts", "evidence", "top", "entity"])
    parser.add_argument("name", nargs="?", help="понятие для evidence или название для entity")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--vocabulary", default=None, help="другой файл словаря")
    parser.add_argument("--llm", action="store_true", help="после разметки проверить моделью")
    parser.add_argument("--limit", type=int, default=None,
                        help="verify: сколько проверить; evidence/top: сколько показать")
    parser.add_argument("--min-similarity", type=float, default=tagging.MIN_SIMILARITY)
    parser.add_argument("--min-similarity-terms", type=float,
                        default=tagging.MIN_SIMILARITY_TERMS)
    parser.add_argument("--max-per-concept", type=int, default=tagging.MAX_PER_CONCEPT)
    parser.add_argument("--territory", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--checked-only", action="store_true",
                        help="evidence: только подтверждённое моделью")
    parser.add_argument("--kind", choices=["объект", "ископаемое", "метод"], default=None)
    parser.add_argument("--embedder", choices=["local", "ollama"], default="local")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args(argv)

    try:
        vocab = load(args.vocabulary)
    except VocabularyError as exc:
        print(f"Словарь: {exc}", file=sys.stderr)
        return 1
    for warning in vocab.warnings:
        print(f"Словарь, предупреждение: {warning}")

    try:
        with db.connect(args.dsn) as conn:
            store.init(conn)

            if args.command == "build":
                from ..index.embed import build_embedder

                print("Загружаю модель эмбеддингов…")
                embedder = build_embedder(args.embedder, device=args.device)
                info = build(conn, embedder, vocab,
                             min_similarity=args.min_similarity,
                             min_similarity_terms=args.min_similarity_terms,
                             max_per_concept=args.max_per_concept)
                if not info:
                    return 1
                kinds = ", ".join(f"{k}: {v}" for k, v in sorted(info["by_kind"].items()))
                print(
                    f"\nПросмотрено {info['chunks']} фрагментов в {info['documents']} статьях.\n"
                    f"Названий: {info['entities']} ({kinds}); меток территорий и методов: "
                    f"{info['tags']}.\n"
                    f"Доказательств понятий: {info['evidence']}"
                    + (f" (убрано устаревших: {info['dropped_stale']})"
                       if info.get("dropped_stale") else "")
                )
                if args.llm:
                    print(f"\nПроверка моделью {args.model} — секунды на фрагмент. "
                          "Можно прервать Ctrl+C и продолжить командой verify.")
                    tagging.verify(conn, _llm(args), vocab, limit=args.limit)
                _print_summary(conn, vocab)
                return 0

            if args.command == "verify":
                left = len(store.unchecked(conn, None))
                if not left:
                    print("Проверять нечего: всё уже просмотрено моделью "
                          "(или разметка не построена — сначала build).")
                    return 0
                todo = min(left, args.limit) if args.limit else left
                print(f"Не проверено: {left}. Проверяю {todo} моделью {args.model}. "
                      "Прервать — Ctrl+C, продолжится с этого места.")
                stats = tagging.verify(conn, _llm(args), vocab, limit=args.limit)
                print(f"\nПодтверждено: {stats['confirmed']}, отклонено: {stats['rejected']}"
                      + (f" (из них без цитаты в тексте: {stats['no_quote']})"
                         if stats["no_quote"] else "")
                      + (f", ошибок модели: {stats['errors']}" if stats["errors"] else ""))
                _print_summary(conn, vocab)
                return 2 if stats["stopped"] else 0

            if args.command == "concepts":
                meta = store.meta(conn)
                if not meta.get("built_at"):
                    print("Разметка не построена: ... graph.cli build")
                    return 0
                print(f"Разметка от {meta['built_at']}, словарь {meta.get('vocabulary_sha')}"
                      + (" — словарь с тех пор изменён, пора build"
                         if meta.get("vocabulary_sha") != vocab.sha else ""))
                _print_summary(conn, vocab)
                return 0

            if args.command == "evidence":
                concept = vocab.concept(args.name or "")
                if concept is None:
                    print(f"Понятия «{args.name}» в словаре нет. Есть: "
                          + "; ".join(vocab.concepts), file=sys.stderr)
                    return 1
                territory = vocab.territory(args.territory) if args.territory else None
                method = vocab.method(args.method) if args.method else None
                if args.territory and territory is None:
                    print(f"Территории «{args.territory}» в словаре нет. Есть: "
                          + "; ".join(vocab.territories), file=sys.stderr)
                    return 1
                if args.method and method is None:
                    print(f"Метода «{args.method}» в словаре нет. Есть: "
                          + "; ".join(vocab.methods), file=sys.stderr)
                    return 1
                rows, totals = store.evidence_for(
                    conn, concept.code,
                    territory=territory.name if territory else None,
                    method=method.name if method else None,
                    checked_only=args.checked_only, limit=args.limit or 10)
                print(f"\n{concept.code} — фрагментов: {totals['fragments']} в "
                      f"{totals['documents']} статьях, показано {len(rows)}\n" + "=" * 78)
                for i, r in enumerate(rows, 1):
                    page = f", с. {r['page']}" if r["page"] else ""
                    print(f"\n{i}. {r['title'][:70]} ({r['year'] or '—'}{page})")
                    print(f"   {r['verdict']} · нашёл: {r['found_by']}"
                          + (f" · похожесть {r['similarity']:.2f}" if r["similarity"] else "")
                          + (f" · слова: {', '.join(r['terms'])}" if r["terms"] else ""))
                    print(f"   «{r['quote']}»")
                    where = r["territories"] + r["methods"]
                    if where:
                        print(f"   во фрагменте: {', '.join(where)}")
                    print(f"   {r['doi'] and 'https://doi.org/' + r['doi'] or r['url']}")
                return 0

            if args.command == "top":
                rows = store.top_entities(conn, args.kind, args.limit or 25)
                if not rows:
                    print("Названий нет. Постройте разметку: ... graph.cli build")
                    return 0
                print(f"{'название':<44}{'вид':<12}{'статей':>7}{'упоминаний':>12}")
                print("-" * 76)
                for row in rows:
                    print(f"{row['name'][:42]:<44}{row['kind']:<12}{row['docs']:>7}"
                          f"{row['mentions']:>12}")
                return 0

            # entity
            if not args.name:
                print('Нужно название: ... graph.cli entity "Анабарский щит"', file=sys.stderr)
                return 1
            match = next(
                (e for e in store.top_entities(conn, limit=5000)
                 if e["name"].lower() == args.name.lower()
                 or args.name.lower() in e["name"].lower()),
                None,
            )
            if not match:
                print(f"«{args.name}» среди названий нет. Список: ... graph.cli top",
                      file=sys.stderr)
                return 1
            print(f"\n{match['name']} — упоминается в {match['docs']} статьях\n" + "=" * 70)
            for doc in store.entity_documents(conn, match["key"]):
                print(f"  · {doc['title'][:66]} ({doc['year'] or '—'}) — упоминаний {doc['hits']}")
                print(f"    {doc['url']}")
            return 0

    except KeyboardInterrupt:
        print("\nПрервано. Сделанное сохранено; verify продолжит с этого места.")
        return 130
    except ImportError as exc:
        print(f"Не хватает библиотеки: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна или ответила ошибкой: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
