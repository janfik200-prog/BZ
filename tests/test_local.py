"""Свои статьи из папки (acquire/local.py): что куда переезжает. Разбор подменён —
проверяется очередь и папки, а не парсер (его проверяет test_pdfs.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from georag.acquire import local
from georag.acquire.models import ACQUIRED, PARSE_FAILED, AcquiredRecord
from georag.parse.config import Settings


class _Worker:
    def __init__(self, settings: Any) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(local, "DoclingWorker", _Worker)
    inbox, outbox, out = tmp_path / "input", tmp_path / "output", tmp_path / "acquired"
    inbox.mkdir()
    for name in ("a.pdf", "b.pdf", "битая.pdf"):
        (inbox / name).write_bytes(f"%PDF {name}".encode())
    (inbox / "notes.txt").write_text("не статья", encoding="utf-8")
    return {"inbox": inbox, "outbox": outbox, "out": out, "logs": tmp_path / "logs"}


def _fake_parse(calls: list[str], stop_after: int | None = None) -> Any:
    def parse_one(path: Path, settings: Any, worker: Any, logger: Any, out_dir: Path) -> Any:
        if stop_after is not None and len(calls) == stop_after:
            raise KeyboardInterrupt
        calls.append(path.name)
        sha = local.hashlib.sha256(path.read_bytes()).hexdigest()
        status = PARSE_FAILED if "битая" in path.name else ACQUIRED
        return AcquiredRecord(doc_id=local.local_doc_id(path, sha), status=status, title=path.stem)

    return parse_one


def _run(f: dict[str, Path]) -> Any:
    return local.process_folder(f["inbox"], f["outbox"], Settings(log_dir=f["logs"]), f["out"])


def test_parsed_move_to_output_failed_aside(
    folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(local, "parse_one", _fake_parse(calls))
    report = _run(folders)
    assert len(report.acquired) == 2
    assert sorted(p.name for p in folders["outbox"].iterdir()) == ["a.pdf", "b.pdf"]
    assert [p.name for p in (folders["inbox"] / local.FAILED_DIR).iterdir()] == ["битая.pdf"]
    assert sorted(p.name for p in folders["inbox"].iterdir() if p.is_file()) == ["notes.txt"]


def test_interrupted_run_keeps_the_rest_in_input(
    folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(local, "parse_one", _fake_parse(calls, stop_after=1))
    with pytest.raises(KeyboardInterrupt):
        _run(folders)
    assert [p.name for p in folders["outbox"].iterdir()] == ["a.pdf"]
    assert {"b.pdf", "битая.pdf"} <= {p.name for p in folders["inbox"].iterdir()}

    calls.clear()
    monkeypatch.setattr(local, "parse_one", _fake_parse(calls))
    _run(folders)
    assert calls == ["b.pdf", "битая.pdf"]  # продолжили с того, до чего не дошли


def test_parsed_but_not_moved_is_not_parsed_again(
    folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(local, "parse_one", _fake_parse(calls))
    _run(folders)
    # Сбой между разбором и переездом: тот же файл снова в input.
    (folders["outbox"] / "a.pdf").rename(folders["inbox"] / "a.pdf")
    calls.clear()
    report = _run(folders)
    assert calls == [] and report.already_known == 1
    assert (folders["outbox"] / "a.pdf").exists()


def test_same_name_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.pdf").write_bytes(b"old")
    (tmp_path / "a.pdf").write_bytes(b"new")
    moved = local._move(tmp_path / "a.pdf", tmp_path / "out")
    assert moved.name == "a (2).pdf"
    assert (tmp_path / "out" / "a.pdf").read_bytes() == b"old"


def test_doc_id_for_russian_name_is_unique() -> None:
    a = local.local_doc_id(Path("Статья.pdf"), "1" * 64)
    b = local.local_doc_id(Path("Отчёт.pdf"), "2" * 64)
    assert a != b and a.startswith("local-")
    assert (
        local.local_doc_id(Path("Gravity Model.pdf"), "f" * 64) == "local-gravity_model-ffffffffff"
    )
