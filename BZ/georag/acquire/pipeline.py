"""Добыча: тема → запросы → поиск → фильтрация → разбор в памяти → ссылка и текст на диск.

PDF не сохраняется. На диск ложатся: ссылка с метаданными, разобранный текст,
структурный документ docling (чтобы перечанковать без выхода в сеть) и чанки.
Плюс sha256 полученных байт — чтобы позже увидеть, что статья на сайте подменилась.

Устойчивость: одна плохая статья не роняет прогон. Каждая получает статус
(добыта / нет открытого текста / не скачалась / не разобралась / отклонена /
нужен человек) и попадает в отчёт.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..parse.config import Settings
from ..parse.models import OK, PARTIAL, ParseInput
from ..parse.pipeline import DoclingWorker, StepLogger, process_document
from .fetch import fetch_pdf
from .heuristic import HeuristicFilter, has_domain_term
from .llm import filter_candidates
from .models import (
    ACQUIRED,
    FETCH_FAILED,
    NOT_FETCHED,
    NO_FULLTEXT,
    PARSE_FAILED,
    REJECTED,
    UNREVIEWED,
    AcquireReport,
    AcquiredRecord,
    Candidate,
)
from .providers import build_providers
from .sources import load_sources

INDEX_NAME = "index.jsonl"


@dataclass
class AcquireConfig:
    sources_path: Path = Path("config/sources.yaml")
    out_dir: Path = Path("data/acquired")
    queries: int = 5
    per_query: int = 25
    max_docs: int = 20
    batch_size: int = 8
    max_mb: int = 80
    fetch_timeout: int = 60
    dry_run: bool = False
    force: bool = False
    make_chunks: bool = True
    mailto: str = ""
    fallback_filter: bool = True   # если модель молчит, решает эвристика, а не человек
    prefer_fetchable: bool = True  # сначала статьи, у которых есть полный текст
    max_fetch_attempts: int = 4    # сколько копий полного текста пробовать подряд
    max_candidates: int = 120      # предел на отбор: ночной прогон должен заканчиваться
    # До модели — дешёвый доменный фильтр: без единого слова рудной тематики в
    # названии и аннотации статья отбрасывается сразу. Модель не тратит на неё время,
    # и место в пределе отбора достаётся статьям, которые могут подойти.
    domain_gate: bool = True
    # Заранее подготовленные запросы по темам. Если для темы они есть, модель
    # на этом шаге не дёргается: думали один раз, ищем каждую ночь одинаково.
    queries_map: dict[str, list[str]] | None = None


# --------------------------------------------------------------------------- #
#  Индекс уже добытого — дедупликация между прогонами
# --------------------------------------------------------------------------- #
def load_index(out_dir: Path) -> dict[str, dict]:
    path = out_dir / INDEX_NAME
    if not path.exists():
        return {}
    index: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("doc_id"):
            index[record["doc_id"]] = record
    return index


def append_index(out_dir: Path, record: AcquiredRecord) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / INDEX_NAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.index_line(), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
#  Шаги
# --------------------------------------------------------------------------- #
def _queries_for(topic: str, llm, fallback, cfg: "AcquireConfig", logger) -> list[str]:
    """Запросы для темы: готовые из файла, иначе от модели, иначе от эвристики."""
    prepared = (cfg.queries_map or {}).get(topic)
    if prepared:
        logger.log("-", "queries", OK, 0.0, detail=f"готовые: {'; '.join(prepared)}")
        return list(prepared)

    step = time.monotonic()
    try:
        queries = llm.queries(topic, cfg.queries)
        logger.log("-", "queries", OK, time.monotonic() - step, detail="; ".join(queries))
        return queries
    except Exception as exc:  # noqa: BLE001 — модель недоступна, прогон продолжается
        queries = fallback.queries(topic, cfg.queries) if fallback else [topic]
        logger.log(
            "-",
            "queries",
            "failed",
            time.monotonic() - step,
            detail=f"откат на эвристику: {'; '.join(queries)}",
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        return queries


def _search_all(providers, queries, per_query, logger) -> list[Candidate]:
    found: list[Candidate] = []
    for query in queries:
        for provider in providers:
            started = time.monotonic()
            try:
                batch = provider.search(query, per_query)
                found.extend(batch)
                logger.log(
                    "-",
                    f"search:{provider.name}",
                    OK,
                    time.monotonic() - started,
                    detail=f"«{query}» → {len(batch)}",
                )
            except Exception as exc:  # noqa: BLE001 — источник мог ответить ошибкой
                logger.log(
                    "-",
                    f"search:{provider.name}",
                    "failed",
                    time.monotonic() - started,
                    # Причина — сразу на экран: без неё «failed» ничего не говорит.
                    detail=f"«{query}» — {str(exc)[:300] or type(exc).__name__}",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
    return found


def _dedup(candidates: list[Candidate], known: dict[str, dict], force: bool):
    unique: dict[str, Candidate] = {}
    already = 0
    for cand in candidates:
        doc_id = cand.doc_id
        if doc_id in unique:
            continue
        if not force and doc_id in known:
            already += 1
            continue
        unique[doc_id] = cand
    return list(unique.values()), already


def _write_doc(out_dir: Path, record: AcquiredRecord, parsed, chunks) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = record.doc_id

    payload = record.to_dict()
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if parsed is not None:
        (out_dir / f"{stem}.md").write_text(parsed.text, encoding="utf-8")
        if parsed.docling_doc:
            # Вместо PDF: из этого файла можно перечанковать, не скачивая статью заново.
            (out_dir / f"{stem}.docling.json").write_text(
                json.dumps(parsed.docling_doc, ensure_ascii=False), encoding="utf-8"
            )
    if chunks:
        (out_dir / f"{stem}.chunks.json").write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
#  Основной проход
# --------------------------------------------------------------------------- #
def acquire(topic: str, llm, settings: Settings, cfg: AcquireConfig) -> AcquireReport:
    started = time.monotonic()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(settings.log_dir, prefix="acquire")
    report = AcquireReport(topic=topic)
    fallback = HeuristicFilter() if cfg.fallback_filter else None

    # --- источники --------------------------------------------------------- #
    sources = load_sources(cfg.sources_path)
    providers, skipped = build_providers(sources)
    for note in skipped:
        logger.log("-", "source", "skipped", 0.0, detail=note)
    if not providers:
        raise RuntimeError(
            "нет ни одного источника со способом api: включите его в каталоге "
            f"{cfg.sources_path}"
        )

    # --- шаг 1-2: тема → запросы ------------------------------------------ #
    report.queries = _queries_for(topic, llm, fallback, cfg, logger)

    # --- шаг 3: поиск ------------------------------------------------------ #
    candidates = _search_all(providers, report.queries, cfg.per_query, logger)
    report.found = len(candidates)

    known = load_index(cfg.out_dir)
    candidates, report.already_known = _dedup(candidates, known, cfg.force)
    report.after_dedup = len(candidates)
    logger.log(
        "-",
        "dedup",
        OK,
        0.0,
        detail=f"найдено {report.found}, новых {report.after_dedup}, уже в базе {report.already_known}",
    )

    if not candidates:
        report.duration_sec = time.monotonic() - started
        return report

    gated: list[Candidate] = []
    if cfg.domain_gate:
        kept = []
        for cand in candidates:
            if has_domain_term(f"{cand.title} {cand.abstract}"):
                kept.append(cand)
            else:
                cand.relevant = False
                cand.reason = "ни одного термина рудной тематики в названии и аннотации"
                cand.filtered_by = "доменный фильтр"
                gated.append(cand)
        candidates = kept
        logger.log("-", "domain", OK, 0.0,
                   detail=f"без рудной тематики отброшено {len(gated)}, на отбор {len(candidates)}")

    # Порядок решает, на что уйдёт лимит max_docs. Сначала статьи, у которых
    # есть адрес полного текста, среди них — открытый доступ: у платных издателей
    # файл всё равно не отдастся, и попытка съест место в очереди. Внутри — свежие.
    if cfg.prefer_fetchable:
        candidates.sort(key=lambda c: (not c.fetch_urls, not c.is_oa, -(c.year or 0)))

    # Отбор моделью — самый долгий шаг, поэтому число кандидатов ограничено.
    if len(candidates) > cfg.max_candidates:
        dropped = len(candidates) - cfg.max_candidates
        candidates = candidates[: cfg.max_candidates]
        logger.log(
            "-", "cap", OK, 0.0,
            detail=f"на отбор взято {cfg.max_candidates}, отложено {dropped}",
        )

    # --- шаг 4: фильтрация аннотаций -------------------------------------- #
    step = time.monotonic()
    errors = filter_candidates(llm, topic, candidates, cfg.batch_size, fallback=fallback)
    relevant = [c for c in candidates if c.relevant is True]
    rejected = [c for c in candidates if c.relevant is False] + gated
    unreviewed = [c for c in candidates if c.relevant is None]
    report.relevant, report.rejected, report.unreviewed = (
        len(relevant),
        len(rejected),
        len(unreviewed),
    )
    logger.log(
        "-",
        "filter",
        OK if not errors else "partial",
        time.monotonic() - step,
        detail=f"по теме {len(relevant)}, мимо {len(rejected)}, без решения {len(unreviewed)}",
        errors=errors,
    )

    for cand in rejected:
        report.records.append(
            AcquiredRecord.from_candidate(cand, REJECTED, note=cand.reason)
        )
    for cand in unreviewed:
        report.records.append(
            AcquiredRecord.from_candidate(cand, UNREVIEWED, note=cand.reason)
        )

    if cfg.dry_run:
        for cand in relevant:
            report.records.append(
                AcquiredRecord.from_candidate(
                    cand, NOT_FETCHED, note="пробный прогон: полный текст не запрашивали"
                )
            )
        report.duration_sec = time.monotonic() - started
        _save_report(cfg.out_dir, logger.run_id, report)
        return report

    # --- шаг 5-7: получить в память, разобрать, проверить, порезать -------- #
    worker = DoclingWorker(settings)
    try:
        for i, cand in enumerate(relevant[: cfg.max_docs], start=1):
            print(f"\n[{i}/{min(len(relevant), cfg.max_docs)}] {cand.title[:70]}")
            record = _acquire_one(cand, settings, cfg, worker, logger)
            report.records.append(record)
            append_index(cfg.out_dir, record)
    finally:
        worker.close()

    report.duration_sec = time.monotonic() - started
    _save_report(cfg.out_dir, logger.run_id, report)
    print(f"\nЛог: {logger.path}")
    return report


def acquire_dois(dois: list[str], settings: Settings, cfg: AcquireConfig,
                 provider=None) -> tuple[AcquireReport, list[str]]:
    """Статьи по списку DOI: без поиска и без отбора — их выбрал человек.

    Метаданные и адреса полного текста — из OpenAlex, дальше всё как обычно:
    получить в память, разобрать, проверить, порезать. Второй результат —
    DOI, которых OpenAlex не знает или которые не удалось спросить.
    """
    from .providers.openalex import OpenAlexProvider, normalize_doi

    started = time.monotonic()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(settings.log_dir, prefix="acquire")
    report = AcquireReport(topic="статьи по списку DOI")
    provider = provider or OpenAlexProvider("openalex", {"mailto": cfg.mailto})
    missing: list[str] = []
    candidates: list[Candidate] = []
    for doi in dict.fromkeys(normalize_doi(d) for d in dois if d and d.strip()):
        try:
            cand = provider.by_doi(doi)
        except Exception as exc:  # noqa: BLE001 — одна статья не рушит список
            missing.append(f"{doi} — {type(exc).__name__}: {exc}")
            continue
        if cand is None:
            missing.append(f"{doi} — в OpenAlex такой статьи нет")
            continue
        cand.relevant, cand.reason, cand.filtered_by = True, "выбрана человеком", "человек"
        candidates.append(cand)
    report.found = len(candidates)
    candidates, report.already_known = _dedup(candidates, load_index(cfg.out_dir), cfg.force)
    report.after_dedup = report.relevant = len(candidates)
    logger.log("-", "dois", OK, 0.0,
               detail=f"найдено {report.found}, новых {report.after_dedup}, не найдено {len(missing)}")
    if candidates and not cfg.dry_run:
        worker = DoclingWorker(settings)
        try:
            for i, cand in enumerate(candidates, start=1):
                print(f"\n[{i}/{len(candidates)}] {cand.title[:70]}")
                record = _acquire_one(cand, settings, cfg, worker, logger)
                report.records.append(record)
                append_index(cfg.out_dir, record)
        finally:
            worker.close()
    report.duration_sec = time.monotonic() - started
    _save_report(cfg.out_dir, logger.run_id, report)
    return report, missing


def load_dois(path: Path) -> list[str]:
    """DOI из файла: YAML со списком «статьи: [{doi: …}]» или просто по одному в строке."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
        items = data.get("статьи") if isinstance(data, dict) else data
        out = []
        for item in items or []:
            doi = item.get("doi") if isinstance(item, dict) else item
            if doi:
                out.append(str(doi))
        return out
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _acquire_one(
    cand: Candidate, settings: Settings, cfg: AcquireConfig, worker, logger
) -> AcquiredRecord:
    urls = list(cand.fetch_urls)
    # Страница статьи — последняя попытка: сама она не PDF, но в её мета-тегах
    # почти всегда лежит прямой адрес файла (citation_pdf_url).
    if cand.landing_page_url and cand.landing_page_url not in urls:
        urls.append(cand.landing_page_url)
    if not urls:
        record = AcquiredRecord.from_candidate(
            cand, NO_FULLTEXT, note="открытого PDF нет, осталась только ссылка"
        )
        logger.log(cand.doc_id, "fetch", NO_FULLTEXT, 0.0, detail=cand.url)
        _write_doc(cfg.out_dir, record, None, None)
        return record

    # Пробуем копии полного текста по очереди: издатель может отдать проверку
    # вместо файла, а копия в репозитории — отдать файл.
    step = time.monotonic()
    user_agent = f"georag/0.1 mailto:{cfg.mailto}" if cfg.mailto else "georag/0.1"
    tried: list[str] = []
    problems: list[str] = []
    fetched = None
    for url in urls[: cfg.max_fetch_attempts]:
        tried.append(url)
        attempt = fetch_pdf(
            url,
            timeout=cfg.fetch_timeout,
            max_mb=cfg.max_mb,
            user_agent=user_agent,
            referer=cand.landing_page_url or "",
        )
        if attempt.ok:
            fetched = attempt
            break
        problems.append(f"{url} → {attempt.error}")

    if fetched is None:
        record = AcquiredRecord.from_candidate(
            cand,
            FETCH_FAILED,
            tried_urls=tried,
            note="; ".join(problems)[:500],
        )
        logger.log(
            cand.doc_id,
            "fetch",
            FETCH_FAILED,
            time.monotonic() - step,
            detail=(
                f"пробовали адресов: {len(tried)}"
                + (f"; последняя причина — {problems[-1].split(' → ', 1)[-1][:90]}" if problems else "")
            ),
            errors=problems,
        )
        _write_doc(cfg.out_dir, record, None, None)
        return record

    logger.log(
        cand.doc_id,
        "fetch",
        OK,
        time.monotonic() - step,
        detail=(
            f"{fetched.bytes_len // 1024} КБ, {fetched.content_type or 'тип не указан'}"
            + (f", со {len(tried)}-й попытки" if len(tried) > 1 else "")
        ),
    )

    inp = ParseInput(
        doc_id=cand.doc_id,
        data=fetched.data,
        display_name=f"{cand.doc_id}.pdf",
        origin=cand.url,          # вместо пути к файлу с документом живёт ссылка
    )
    doc_result, parsed, chunks = process_document(
        inp,
        settings,
        worker,
        logger,
        make_chunks=cfg.make_chunks,
        write_outputs=False,      # свои файлы пишем сами, PDF не сохраняем
    )

    status = ACQUIRED if doc_result.status in (OK, PARTIAL) else PARSE_FAILED
    record = AcquiredRecord.from_candidate(
        cand,
        status,
        final_url=fetched.final_url,
        sha256=fetched.sha256,
        bytes_len=fetched.bytes_len,
        fetched_at=fetched.fetched_at,
        tried_urls=tried,
        parser=doc_result.parser,
        accuracy=doc_result.accuracy,
        pages=doc_result.pages,
        sections=doc_result.sections,
        tables=doc_result.tables,
        chunks=len(chunks),
        duration_sec=doc_result.duration_sec,
        failed_checks=doc_result.failed_checks,
        attempts=doc_result.attempts,
        note="" if status == ACQUIRED else "не прошёл валидацию, нужен человек",
    )
    _write_doc(cfg.out_dir, record, parsed, chunks)
    return record


def _save_report(out_dir: Path, run_id: str, report: AcquireReport) -> Path:
    path = out_dir / f"acquire-report-{run_id}.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {path}")
    return path
