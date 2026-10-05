"""Структуры данных этапа добычи."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

# Статусы документа на этапе добычи
ACQUIRED = "acquired"  # найден, скачан в память, разобран, прошёл валидацию
NO_FULLTEXT = "no_fulltext"  # найден, но открытого PDF нет — только ссылка
FETCH_FAILED = "fetch_failed"  # ссылка есть, но файл не отдался или это не PDF
PARSE_FAILED = "parse_failed"  # скачали, но разобрать не смогли
REJECTED = "rejected"  # LLM посчитала статью нерелевантной теме
UNREVIEWED = "unreviewed"  # LLM не смогла решить — нужен человек
NOT_FETCHED = "not_fetched"  # пробный прогон: отобрана, но мы намеренно не качали


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


@dataclass
class Candidate:
    """Статья, найденная в источнике. Полного текста может и не быть."""

    source: str
    external_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    language: str | None = None
    landing_page_url: str | None = None
    # Все известные копии полного текста по порядку предпочтения: версия издателя,
    # репозитории, зеркала. Один заблокированный адрес не должен стоить нам статьи.
    pdf_urls: list[str] = field(default_factory=list)
    is_oa: bool = False
    oa_status: str | None = None
    license: str | None = None
    query: str = ""
    # решение фильтрации
    relevant: bool | None = None
    reason: str = ""
    filtered_by: str = ""  # llm | эвристика

    @property
    def fetch_urls(self) -> list[str]:
        """Адреса полного текста по порядку предпочтения, без повторов."""
        return list(dict.fromkeys(u for u in self.pdf_urls if u))

    @property
    def pdf_url(self) -> str | None:
        """Основной адрес полного текста — он же попадает в запись."""
        urls = self.fetch_urls
        return urls[0] if urls else None

    @property
    def doc_id(self) -> str:
        """Стабильный идентификатор: по нему идёт дедупликация между прогонами."""
        if self.doi:
            return _slug(self.doi)
        return _slug(f"{self.source}-{self.external_id}")

    @property
    def url(self) -> str:
        """Ссылка, которая остаётся вместо PDF."""
        if self.landing_page_url:
            return self.landing_page_url
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.pdf_url or ""

    def brief(self, abstract_chars: int = 700) -> str:
        head = f"{self.title} ({self.year or 'без года'}, {self.journal or 'источник не указан'})"
        body = re.sub(r"\s+", " ", self.abstract)[:abstract_chars]
        return f"{head}\n{body}" if body else head


@dataclass
class AcquiredRecord:
    """То, что остаётся на диске вместо PDF: ссылка, метаданные и разобранный текст."""

    doc_id: str
    status: str
    # откуда
    source: str = ""
    query: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    language: str | None = None
    url: str = ""
    pdf_url: str | None = None
    final_url: str = ""
    oa_status: str | None = None
    license: str | None = None
    # чем подтверждается, что мы разбирали именно этот файл
    sha256: str = ""
    bytes_len: int = 0
    fetched_at: str = ""
    # что получилось
    parser: str = ""
    accuracy: float = 0.0
    pages: int = 0
    sections: int = 0
    tables: int = 0
    chunks: int = 0
    duration_sec: float = 0.0
    validation: dict[str, Any] | None = None
    failed_checks: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    filtered_by: str = ""  # кто решил: модель или эвристика
    tried_urls: list[str] = field(default_factory=list)  # какие адреса полного текста пробовали
    note: str = ""

    @classmethod
    def from_candidate(cls, cand: Candidate, status: str, **kwargs: Any) -> AcquiredRecord:
        return cls(
            doc_id=cand.doc_id,
            status=status,
            source=cand.source,
            query=cand.query,
            title=cand.title,
            authors=cand.authors,
            year=cand.year,
            journal=cand.journal,
            doi=cand.doi,
            language=cand.language,
            url=cand.url,
            pdf_url=cand.pdf_url,
            oa_status=cand.oa_status,
            license=cand.license,
            filtered_by=cand.filtered_by,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def index_line(self) -> dict[str, Any]:
        """Компактная строка для index.jsonl — по ней работает дедупликация."""
        return {
            "doc_id": self.doc_id,
            "status": self.status,
            "doi": self.doi,
            "url": self.url,
            "title": self.title[:200],
            "year": self.year,
            "source": self.source,
            "sha256": self.sha256,
            "chunks": self.chunks,
            "seen_at": self.fetched_at or datetime.now(UTC).isoformat(timespec="seconds"),
        }


@dataclass
class AcquireReport:
    topic: str
    queries: list[str] = field(default_factory=list)
    found: int = 0
    after_dedup: int = 0
    already_known: int = 0
    relevant: int = 0
    rejected: int = 0
    unreviewed: int = 0
    records: list[AcquiredRecord] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def acquired(self) -> list[AcquiredRecord]:
        return [r for r in self.records if r.status == ACQUIRED]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "queries": self.queries,
            "found": self.found,
            "after_dedup": self.after_dedup,
            "already_known": self.already_known,
            "relevant": self.relevant,
            "rejected": self.rejected,
            "unreviewed": self.unreviewed,
            "acquired": len(self.acquired),
            "total_chunks": self.total_chunks,
            "duration_sec": round(self.duration_sec, 2),
            "records": [r.to_dict() for r in self.records],
        }
