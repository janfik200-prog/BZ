"""Структуры данных, которые ходят между шагами пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Статусы парсинга документа
OK = "ok"                    # распарсили и прошли валидацию
PARTIAL = "partial"          # распарсили, но парсер сам сообщил об ошибках
FAILED = "failed"            # парсер упал
TIMEOUT = "timeout"          # процесс убит по таймауту
MANUAL_REVIEW = "manual_review"  # не смогли — на ручную проверку


@dataclass
class ParseInput:
    """Что парсим: файл на диске или байты в памяти.

    Второй вариант нужен режиму добычи: статья скачивается в память, парсится
    и на диск ложится уже разобранный текст, а не PDF.
    """

    doc_id: str
    path: Path | None = None
    data: bytes | None = None
    display_name: str = ""
    origin: str = ""  # ссылка на статью — с ней документ живёт дальше вместо PDF

    @classmethod
    def of(cls, source: "Path | str | ParseInput") -> "ParseInput":
        if isinstance(source, ParseInput):
            return source
        path = Path(source)
        return cls(doc_id=path.stem, path=path, display_name=path.name, origin=str(path))

    @property
    def name(self) -> str:
        return self.display_name or f"{self.doc_id}.pdf"


@dataclass
class ParsedDoc:
    """Результат работы одного парсера над одним PDF."""

    doc_id: str
    source_path: str
    parser: str                      # docling | docling+ocr | pymupdf4llm | manual
    status: str                      # OK | PARTIAL | FAILED | TIMEOUT
    markdown: str = ""
    page_count: int = 0
    sections: list[str] = field(default_factory=list)
    table_count: int = 0
    pages_text: dict[int, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0
    # Сериализованный DoclingDocument — нужен для чанкинга. У fallback-парсеров его нет.
    docling_doc: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return self.markdown or "\n\n".join(self.pages_text.get(p, "") for p in sorted(self.pages_text))

    def to_dict(self, with_doc: bool = True) -> dict:
        d = asdict(self)
        d["pages_text"] = {str(k): v for k, v in self.pages_text.items()}
        if not with_doc:
            d.pop("docling_doc", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedDoc":
        d = dict(d)
        d["pages_text"] = {int(k): v for k, v in (d.get("pages_text") or {}).items()}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Check:
    name: str
    ok: bool
    critical: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    ok: bool = True
    checks: list[Check] = field(default_factory=list)
    # Подсказка пайплайну, что делать дальше: rerun_ocr | fallback | none
    suggestion: str = "none"

    def add(self, name: str, ok: bool, critical: bool, detail: str = "") -> None:
        self.checks.append(Check(name=name, ok=ok, critical=critical, detail=detail))
        if critical and not ok:
            self.ok = False

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "suggestion": self.suggestion,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class Chunk:
    """Чанк, готовый к эмбеддингу и к цитированию.

    text       — оригинал, его показываем пользователю и цитируем;
    embed_text — то, что уходит в модель эмбеддинга (с заголовками разделов сверху).
    """

    doc_id: str
    index: int
    text: str
    embed_text: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    n_tokens: int = 0
    has_table: bool = False
    source_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
