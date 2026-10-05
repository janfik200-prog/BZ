"""CLI этапа добычи.

    python -m georag.acquire.cli "выделение рудных узлов"
    python -m georag.acquire.cli --topics config/topics.yaml            # весь список тем за прогон

Темы в модель, обратно — запросы, поиск не запускается:

    python -m georag.acquire.cli --queries-only                  # темы с клавиатуры
    python -m georag.acquire.cli "тема" --queries-only --save-queries queries.yaml
    python -m georag.acquire.cli --topics config/topics.yaml --queries-file queries.yaml

    python -m georag.acquire.cli "прогноз оруденения" --dry-run  # поиск и отбор, без скачивания
    python -m georag.acquire.cli "рудные узлы" --no-llm          # отбор эвристикой, без модели
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..common import add_llm_args
from ..parse.config import Settings
from .heuristic import HeuristicFilter
from .llm import OllamaLLM
from .models import (
    ACQUIRED,
    FETCH_FAILED,
    NO_FULLTEXT,
    NOT_FETCHED,
    PARSE_FAILED,
    REJECTED,
    UNREVIEWED,
)
from .pipeline import AcquireConfig, acquire

STATUS_LABELS = {
    ACQUIRED: "добыта",
    NO_FULLTEXT: "нет открытого PDF",
    NOT_FETCHED: "отобрана, не качали",
    FETCH_FAILED: "не скачалась",
    PARSE_FAILED: "не разобралась",
    REJECTED: "мимо темы",
    UNREVIEWED: "нужно решение",
}


def load_topics(path: Path) -> list[str]:
    """Список тем из файла: topics: [..] в YAML либо просто строки по одной."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(text) or {}
        items = raw.get("topics") if isinstance(raw, dict) else raw
    elif path.suffix == ".json":
        raw = json.loads(text)
        items = raw.get("topics") if isinstance(raw, dict) else raw
    else:
        items = [line.strip() for line in text.splitlines()]

    topics = [
        str(t).strip() for t in (items or []) if str(t).strip() and not str(t).startswith("#")
    ]
    if not topics:
        raise ValueError(f"в {path} нет ни одной темы")
    return topics


def load_queries(path: Path) -> dict[str, list[str]]:
    """Готовые запросы по темам: {тема: [запросы]} — из YAML или JSON."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("queries") if isinstance(raw, dict) and "queries" in raw else raw
    if not isinstance(data, dict):
        raise ValueError(f"в {path} ожидается «тема: [запросы]»")
    return {
        str(topic): [str(q).strip() for q in (queries or []) if str(q).strip()]
        for topic, queries in data.items()
    }


def save_queries(path: Path, prepared: dict[str, list[str]]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Запросы, которые ваша модель составила по темам.\n"
        "# Правьте свободно: при запуске с --queries-file они берутся отсюда,\n"
        "# и модель на этом шаге больше не дёргается.\n\n"
    )
    body = yaml.safe_dump({"queries": prepared}, allow_unicode=True, sort_keys=False)
    path.write_text(header + body, encoding="utf-8")


def read_topics_from_stdin() -> list[str]:
    """Темы построчно с клавиатуры: пустая строка заканчивает ввод."""
    print("Пишите темы по одной, каждая с новой строки. Пустая строка — закончить.\n")
    topics = []
    while True:
        try:
            line = input("тема> ").strip()
        except EOFError:
            break
        if not line:
            break
        topics.append(line)
    return topics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Добыча статей для базы знаний ГеоRAG")
    p.add_argument("topic", nargs="?", help="тема поиска, например «выделение рудных узлов»")
    p.add_argument("--topics", type=Path, default=None, help="файл со списком тем")
    p.add_argument(
        "--dois",
        type=Path,
        default=None,
        help="файл со списком DOI — статьи без поиска и отбора (config/articles.yaml)",
    )
    p.add_argument(
        "--doi", action="append", default=[], help="одна статья по DOI (можно несколько раз)"
    )
    p.add_argument(
        "--sources", type=Path, default=Path("config/sources.yaml"), help="каталог источников"
    )
    p.add_argument("--out", type=Path, default=Path("data/acquired"))
    p.add_argument("--queries", type=int, default=5, help="сколько запросов просить у модели")
    p.add_argument("--per-query", type=int, default=25, help="сколько результатов на запрос")
    p.add_argument("--max-docs", type=int, default=20, help="сколько статей разбирать за тему")
    p.add_argument("--dry-run", action="store_true", help="только поиск и фильтрация")
    p.add_argument(
        "--queries-only",
        action="store_true",
        help="отдать темы модели и показать запросы, без поиска. Без темы — ввод с клавиатуры",
    )
    p.add_argument(
        "--save-queries", type=Path, default=None, help="куда записать полученные запросы"
    )
    p.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="взять готовые запросы отсюда, не обращаясь к модели",
    )
    p.add_argument("--no-llm", action="store_true", help="отбор эвристикой, без модели")
    p.add_argument(
        "--no-fallback", action="store_true", help="не подстраховывать модель эвристикой"
    )
    p.add_argument("--force", action="store_true", help="игнорировать индекс уже добытого")
    p.add_argument("--no-chunks", action="store_true", help="не резать на чанки")
    add_llm_args(p)
    p.add_argument("--mailto", default="", help="e-mail для polite pool OpenAlex и Crossref")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=None)
    p.add_argument(
        "--no-ocr",
        action="store_true",
        help="не распознавать картинки: быстрее в разы, годится для обычных статей",
    )
    p.add_argument(
        "--tables",
        choices=["accurate", "fast"],
        default=None,
        help="разбор таблиц: accurate точнее, fast заметно быстрее",
    )
    p.add_argument("--max-tokens", type=int, default=None, help="размер чанка в токенах")
    return p


def _print_report(report: Any, show_table: bool = True) -> None:
    print("\n" + "=" * 78)
    print(f"Тема: «{report.topic}» | запросов: {len(report.queries)}")
    for query in report.queries:
        print(f"  · {query}")
    print(
        f"\nНайдено {report.found}, новых {report.after_dedup}, уже было {report.already_known}; "
        f"по теме {report.relevant}, мимо {report.rejected}, без решения {report.unreviewed}"
    )

    if show_table and report.records:
        print("\n" + f"{'статья':<50}{'статус':<20}{'кто отобрал':<14}{'чанки':>5}")
        print("-" * 78)
        for rec in report.records:
            title = rec.title[:48] + ("…" if len(rec.title) > 48 else "")
            print(
                f"{title:<50}{STATUS_LABELS.get(rec.status, rec.status):<20}"
                f"{(rec.filtered_by or '-')[:12]:<14}{rec.chunks or '':>5}"
            )

    print("-" * 78)
    print(
        f"Добыто и разобрано: {len(report.acquired)} | чанков: {report.total_chunks} | "
        f"время: {report.duration_sec:.0f} с"
    )

    problems = [r for r in report.records if r.status in (FETCH_FAILED, PARSE_FAILED, UNREVIEWED)]
    if problems:
        print("\nНе получилось взять автоматически:")
        for rec in problems:
            print(f"  • [{STATUS_LABELS.get(rec.status, rec.status)}] {rec.title[:60]}")
            print(f"    {rec.url}")
            if rec.note:
                print(f"    {rec.note[:150]}")


def _by_doi(args: Any) -> int:
    """Статьи по DOI: без поиска и отбора моделью — их выбрал человек."""
    from .pipeline import acquire_dois, load_dois

    dois = list(args.doi)
    if args.dois:
        try:
            dois += load_dois(args.dois)
        except (OSError, ValueError) as exc:
            print(f"Не читается список DOI: {exc}", file=sys.stderr)
            return 1
    if not dois:
        print("Список DOI пуст", file=sys.stderr)
        return 1
    settings = Settings()
    if args.device:
        settings.device = args.device
    if args.no_ocr:
        settings.use_ocr = False
    cfg = AcquireConfig(
        out_dir=args.out,
        force=args.force,
        make_chunks=not args.no_chunks,
        mailto=args.mailto,
        dry_run=args.dry_run,
    )
    print(f"Статей по DOI: {len(dois)}")
    report, missing = acquire_dois(dois, settings, cfg)
    _print_report(report)
    if missing:
        print("\nНе найдено в OpenAlex:")
        for line in missing:
            print(f"  • {line}")
    print("\nДальше: python georag.py ingest, потом python georag.py graph")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dois or args.doi:
        return _by_doi(args)

    try:
        if args.topics:
            topics = load_topics(args.topics)
        elif args.topic:
            topics = [args.topic]
        elif args.queries_only:
            topics = read_topics_from_stdin()
        else:
            print("Укажите тему, --topics с файлом тем или --queries-only", file=sys.stderr)
            return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Не читается список тем: {exc}", file=sys.stderr)
        return 1

    if not topics:
        print("Ни одной темы не задано", file=sys.stderr)
        return 1

    settings = Settings()
    if args.device:
        settings.device = args.device
    if args.max_tokens:
        settings.max_tokens = args.max_tokens
    if args.no_ocr:
        settings.use_ocr = False
    if args.tables:
        settings.table_mode = args.tables

    cfg = AcquireConfig(
        sources_path=args.sources,
        out_dir=args.out,
        queries=args.queries,
        per_query=args.per_query,
        max_docs=args.max_docs,
        dry_run=args.dry_run,
        force=args.force,
        make_chunks=not args.no_chunks,
        mailto=args.mailto,
        fallback_filter=not args.no_fallback,
    )

    if args.queries_file:
        try:
            cfg.queries_map = load_queries(args.queries_file)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Не читаются готовые запросы: {exc}", file=sys.stderr)
            return 1

    llm = HeuristicFilter() if args.no_llm else OllamaLLM(model=args.model, host=args.ollama_host)

    # Режим «только запросы»: тема уходит в модель, обратно приходят запросы.
    # Поиск не запускается — это шаг, который можно посмотреть и поправить.
    if args.queries_only:
        fallback = HeuristicFilter()
        prepared: dict[str, list[str]] = {}
        for topic in topics:
            source = "готовые"
            if cfg.queries_map and topic in cfg.queries_map:
                queries = cfg.queries_map[topic]
            else:
                try:
                    queries = llm.queries(topic, args.queries)
                    source = llm.model
                except Exception as exc:  # noqa: BLE001
                    queries = fallback.queries(topic, args.queries)
                    source = f"эвристика ({type(exc).__name__})"
            prepared[topic] = queries
            print(f"\n«{topic}» → {source}")
            for query in queries:
                print(f"  · {query}")

        if args.save_queries:
            save_queries(args.save_queries, prepared)
            print(f"\nЗаписано в {args.save_queries}.")
            print(
                f"Искать по ним: python -m georag.acquire.cli --topics {args.topics or 'config/topics.yaml'} "
                f"--queries-file {args.save_queries}"
            )
        return 0

    print(f"Тем: {len(topics)} | отбор: {llm.model} | каталог: {args.sources}")

    reports = []
    for i, topic in enumerate(topics, start=1):
        print(f"\n########## [{i}/{len(topics)}] {topic}")
        try:
            reports.append(acquire(topic, llm, settings, cfg))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"Тема «{topic}» пропущена: {exc}", file=sys.stderr)

    if not reports:
        return 1

    for report in reports:
        _print_report(report, show_table=len(topics) == 1)

    if len(reports) > 1:
        print("\n" + "=" * 78)
        print("Итог по всем темам:")
        for report in reports:
            print(
                f"  {report.topic[:48]:<50} добыто {len(report.acquired):>3}, "
                f"чанков {report.total_chunks:>5}"
            )
        print(
            f"\nВсего добыто: {sum(len(r.acquired) for r in reports)}, "
            f"чанков: {sum(r.total_chunks for r in reports)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
