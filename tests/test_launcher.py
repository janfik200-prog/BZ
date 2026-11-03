"""Запускатель georag.py: ожидание готовности базы после docker compose up.

Docker и часы подменены: проверки идут мгновенно, контейнеры не трогаются.
georag.py грузится по пути файла под другим именем — у него то же имя, что у пакета
georag. При импорте он меняет sys.pycache_prefix и переменные окружения: их
возвращаем как было, чтобы не задеть другие проверки."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
PS_DB = ("ps", "--all", "--format", "{{.State}} {{.Health}}", "db")

MonkeyPatch = pytest.MonkeyPatch
Capsys = pytest.CaptureFixture[str]


def _load_launcher() -> ModuleType:
    environ, prefix = dict(os.environ), sys.pycache_prefix
    spec = importlib.util.spec_from_file_location("georag_launcher", ROOT / "georag.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for key in set(os.environ) - set(environ):
            del os.environ[key]
        os.environ.update(environ)
        sys.pycache_prefix = prefix
    return module


launcher = _load_launcher()


class FakeTime:
    """Часы понарошку: sleep не ждёт, а двигает monotonic вперёд."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeDocker:
    """docker compose понарошку. states — что по очереди отвечает ps о контейнере базы
    («running healthy», «exited »…), последнее повторяется; ready — коды pg_isready
    по очереди (кончились — 1, «не готова»). Все вызовы запоминаются."""

    def __init__(self, states: list[str], ready: list[int] | None = None) -> None:
        self.states = list(states)
        self.ready = list(ready or [])
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str, quiet: bool = False) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args == PS_DB:
            out = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return subprocess.CompletedProcess(args, 0, out + "\n", "")
        if args[0] == "exec":
            return subprocess.CompletedProcess(args, self.ready.pop(0) if self.ready else 1)
        if args[-2:] == ("up", "-d") or args[0] == "--profile":
            return subprocess.CompletedProcess(args, 0)
        raise AssertionError(f"неожиданный вызов docker compose {args}")

    def count(self, word: str) -> int:
        return sum(1 for args in self.calls if word in args)


@pytest.fixture
def clock(monkeypatch: MonkeyPatch) -> FakeTime:
    fake = FakeTime()
    monkeypatch.setattr(launcher, "time", fake)
    return fake


def fake_docker(
    monkeypatch: MonkeyPatch, states: list[str], ready: list[int] | None = None
) -> FakeDocker:
    fake = FakeDocker(states, ready)
    monkeypatch.setattr(launcher, "_docker", fake)
    return fake


def fake_run(monkeypatch: MonkeyPatch, code: int = 0, out: str = "") -> list[list[str]]:
    """Подменить сам subprocess.run: проверяется и настоящий помощник _docker."""
    seen: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, code, out, "")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    return seen


# --------------------------------------------------------------------------- #
#  _wait_database
# --------------------------------------------------------------------------- #
def test_ready_at_once(monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys) -> None:
    docker = fake_docker(monkeypatch, ["running healthy"])
    assert launcher._wait_database() is True
    assert clock.sleeps == []
    assert docker.count("pg_isready") == 0  # healthy — pg_isready не нужен
    assert capsys.readouterr().out == ""  # ждать не пришлось — и говорить не о чем


def test_ready_after_several_polls(
    monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys
) -> None:
    docker = fake_docker(monkeypatch, ["running starting"] * 3 + ["running healthy"])
    assert launcher._wait_database() is True
    assert clock.sleeps == [1, 1, 1]
    out = capsys.readouterr().out
    assert out.count("Жду, пока база будет готова") == 1  # один раз, а не на каждый опрос
    assert "База готова (3 с)" in out
    assert docker.count("pg_isready") == 3  # пока healthcheck молчит — спрашиваем сами


def test_starting_but_pg_isready_answers(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    """Первая проверка healthcheck — через 10 с после старта: её не ждём, если
    pg_isready уже отвечает."""
    fake_docker(monkeypatch, ["running starting"], ready=[2, 0])
    assert launcher._wait_database() is True
    assert clock.sleeps == [1]


def test_timeout(monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys) -> None:
    fake_docker(monkeypatch, ["running starting"])
    assert launcher._wait_database() is False
    assert clock.sleeps == [1] * 60
    captured = capsys.readouterr()
    assert captured.out.count("Жду, пока база будет готова") == 1
    assert "не ответила за 60 с" in captured.err
    assert "docker logs georag-db" in captured.err


def test_timeout_custom_limit(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    fake_docker(monkeypatch, ["restarting "])
    assert launcher._wait_database(limit=5) is False
    assert clock.sleeps == [1] * 5


def test_docker_not_installed(monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys) -> None:
    def missing(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(launcher.subprocess, "run", missing)
    assert launcher._wait_database() is False
    assert clock.sleeps == []
    assert "Docker не найден" in capsys.readouterr().err


def test_docker_not_responding(monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys) -> None:
    fake_run(monkeypatch, code=1)
    assert launcher._wait_database() is False
    assert clock.sleeps == []
    assert "Docker не отвечает" in capsys.readouterr().err


def test_real_docker_command(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    """Что именно уходит в docker и как читается его ответ (с переводом строки)."""
    seen = fake_run(monkeypatch, out="running healthy\n")
    assert launcher._wait_database() is True
    assert seen == [["docker", "compose", *PS_DB]]


@pytest.mark.parametrize("state", ["exited", "dead"])
def test_container_stopped_while_waiting(
    monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys, state: str
) -> None:
    fake_docker(monkeypatch, ["running starting", f"{state} "])
    assert launcher._wait_database() is False
    assert clock.sleeps == [1]  # не ждём все 60 с
    err = capsys.readouterr().err
    assert f"База остановилась ({state})" in err
    assert "docker logs georag-db" in err


def test_no_container(monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys) -> None:
    fake_docker(monkeypatch, [""])
    assert launcher._wait_database() is False
    assert clock.sleeps == []
    assert "Контейнера базы нет" in capsys.readouterr().err


def test_no_healthcheck_uses_pg_isready(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    docker = fake_docker(monkeypatch, ["running "], ready=[2, 2, 0])
    assert launcher._wait_database() is True
    assert clock.sleeps == [1, 1]
    exec_call = next(args for args in docker.calls if args[0] == "exec")
    check = ("pg_isready", "-q", "-h", "127.0.0.1", "-U", "georag", "-d", "georag")
    assert exec_call == ("exec", "-T", "db", *check)


# --------------------------------------------------------------------------- #
#  _ensure_database и start — тот же интерфейс, ожидание вместо sleep(5)
# --------------------------------------------------------------------------- #
def test_ensure_database_running_and_healthy(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    docker = fake_docker(monkeypatch, ["running healthy"])
    assert launcher._ensure_database() is True
    assert docker.calls == [PS_DB]  # один вызов docker, как и раньше
    assert clock.sleeps == []


def test_ensure_database_starts_and_waits(
    monkeypatch: MonkeyPatch, clock: FakeTime, capsys: Capsys
) -> None:
    docker = fake_docker(monkeypatch, ["exited ", "running starting", "running healthy"])
    assert launcher._ensure_database() is True
    assert ("up", "-d") in docker.calls
    assert clock.sleeps == [1]  # ровно столько, сколько база поднималась, а не 5 с
    assert "База не запущена. Поднимаю" in capsys.readouterr().out


def test_ensure_database_running_but_not_ready(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    """База запущена (например, её только что поднял сам Docker Desktop), но ещё не
    принимает подключения — ждём, а не поднимаем заново."""
    docker = fake_docker(monkeypatch, ["running starting"], ready=[2, 0])
    assert launcher._ensure_database() is True
    assert ("up", "-d") not in docker.calls
    assert clock.sleeps == [1]


def test_ensure_database_docker_down(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    fake_run(monkeypatch, code=1)
    assert launcher._ensure_database() is False
    assert clock.sleeps == []


def test_start_waits_for_database_then_model(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    # Первое состояние — проверка «отвечает ли Docker» до up, дальше — ожидание базы.
    fake_docker(monkeypatch, ["exited ", "running starting", "running healthy"])
    monkeypatch.setattr(launcher, "_ensure_llm", lambda: True)
    assert launcher.cmd_start(launcher.parse(["start"])) == 0
    assert clock.sleeps == [1]


def test_start_database_failed(monkeypatch: MonkeyPatch, clock: FakeTime) -> None:
    fake_docker(monkeypatch, ["exited "])
    asked: list[str] = []

    def ensure_llm() -> bool:
        asked.append("llm")
        return True

    monkeypatch.setattr(launcher, "_ensure_llm", ensure_llm)
    assert launcher.cmd_start(launcher.parse(["start"])) == 1
    assert asked == []  # база не поднялась — полчаса ждать модель незачем


@pytest.mark.parametrize(
    ("dsn", "local"),
    [
        ("", True),  # по умолчанию — наш контейнер
        ("postgresql://georag:georag@127.0.0.1:5433/georag_copy", True),
        ("postgresql://georag:georag@localhost:5433/georag", True),
        ("postgresql://u:p@10.0.0.5:5432/georag", False),  # чужой сервер — Docker не трогаем
        ("postgresql://u:p@127.0.0.1:5999/georag", False),
    ],
)
def test_local_dsn(dsn: str, local: bool) -> None:
    assert launcher._local_dsn(dsn) is local


def test_add_blank_topic(capsys: Capsys) -> None:
    """Тема из одних пробелов — как пустая: добыча не начинается."""
    assert launcher.cmd_add(launcher.parse(["add", "   "])) == 1
    assert "Нужна тема" in capsys.readouterr().err


def test_dsn_and_queries_accepted() -> None:
    opts = launcher.parse(["stats", "--dsn", "postgresql://x@h:1/db", "--queries", "1"])
    assert (opts.dsn, opts.queries) == ("postgresql://x@h:1/db", 1)


@pytest.mark.parametrize(
    "argv, rest, dsn",
    [
        (["dict", "list", "--dsn", "postgresql://h:1/x"], ["dict", "list"], "postgresql://h:1/x"),
        (["--dsn=postgresql://h:1/x", "stats"], ["stats"], "postgresql://h:1/x"),
        (["dict", "find", "адуляр"], ["dict", "find", "адуляр"], "прежний"),
        (["stats", "--dsn"], ["stats"], "прежний"),  # без значения — адрес прежний
    ],
)
def test_take_dsn(monkeypatch: MonkeyPatch, argv: list[str], rest: list[str], dsn: str) -> None:
    monkeypatch.setenv("GEORAG_DSN", "прежний")
    assert launcher._take_dsn(argv) == rest
    assert os.environ["GEORAG_DSN"] == dsn


@pytest.mark.parametrize("command", ["dict", "titles", "evidence", "formula"])
def test_dsn_reaches_commands_with_own_args(monkeypatch: MonkeyPatch, command: str) -> None:
    """Команда со своими ключами получает их без --dsn (dict и titles его не знают и
    падали), а адрес — через GEORAG_DSN ещё до проверки базы: база на другом сервере —
    свой контейнер не поднимается."""
    monkeypatch.setenv("GEORAG_DSN", "")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(launcher, "_need_venv", lambda: None)
    monkeypatch.setattr(launcher, "_db_state", lambda: pytest.fail("Docker не нужен"))
    monkeypatch.setattr(launcher, "_py", lambda module, *args: seen.update(args=args) or 0)
    remote = "postgresql://georag:georag@10.0.0.5:5432/georag"
    assert launcher.main([command, "x", "--dsn", remote]) == 0
    assert seen["args"] == ("x",)
    assert os.environ["GEORAG_DSN"] == remote


def test_start_docker_not_responding(monkeypatch: MonkeyPatch, capsys: Capsys) -> None:
    """Docker не отвечает — start говорит по-русски, что делать, и up не зовёт."""
    seen = fake_run(monkeypatch, code=1)
    assert launcher.cmd_start(launcher.parse(["start"])) == 1
    assert "Запустите Docker Desktop" in capsys.readouterr().err
    assert not any("up" in cmd for cmd in seen)


# --------------------------------------------------------------------------- #
#  Модель-строитель (план графа v2, Б2): переключение на одной видеокарте
# --------------------------------------------------------------------------- #
def fake_models(
    monkeypatch: MonkeyPatch, analyst: bool, builder: bool = False, starts: bool = True
) -> list[str]:
    """Подменить docker и модели; вернуть журнал действий."""
    log: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_llm_up",
        lambda timeout=5, host=launcher.LLM: builder if host == launcher.BUILDER_HOST else analyst,
    )
    monkeypatch.setattr(launcher, "_docker", lambda *a, **k: log.append(" ".join(a)))
    monkeypatch.setattr(launcher, "_running", lambda service: analyst and service == "llm")
    monkeypatch.setattr(launcher, "_start_builder", lambda: log.append("start builder") or starts)
    monkeypatch.setattr(launcher, "_ensure_llm", lambda wait=True: log.append("start llm") or True)
    return log


def test_builder_already_up(monkeypatch: MonkeyPatch) -> None:
    log = fake_models(monkeypatch, analyst=True, builder=True)
    with launcher._builder(launcher.parse(["graph"])) as ready:
        assert ready
    assert log == []  # ничего не переключали


def test_builder_replaces_analyst_and_returns_it(monkeypatch: MonkeyPatch) -> None:
    log = fake_models(monkeypatch, analyst=True)
    with launcher._builder(launcher.parse(["graph", "--yes"])) as ready:
        assert ready
        assert log == ["--profile llm stop llm", "start builder"]
    assert log[2:] == ["--profile builder stop builder", "start llm"]


def test_builder_asks_before_stopping_chat(monkeypatch: MonkeyPatch, capsys: Capsys) -> None:
    log = fake_models(monkeypatch, analyst=True)
    monkeypatch.setattr(launcher.sys.stdin, "isatty", lambda: False)
    with launcher._builder(launcher.parse(["graph"])) as ready:
        assert not ready
    assert log == []  # чат-бот не тронут
    assert "--yes" in capsys.readouterr().err


def test_builder_without_analyst(monkeypatch: MonkeyPatch) -> None:
    log = fake_models(monkeypatch, analyst=False)
    with launcher._builder(launcher.parse(["graph"])) as ready:
        assert ready
    assert log == ["start builder", "--profile builder stop builder"]  # 14B не поднимаем


def test_analyst_returns_after_interrupt(monkeypatch: MonkeyPatch) -> None:
    log = fake_models(monkeypatch, analyst=True)
    with pytest.raises(KeyboardInterrupt), launcher._builder(launcher.parse(["graph", "--yes"])):
        raise KeyboardInterrupt
    assert log[-1] == "start llm"


def test_graph_runs_on_builder(monkeypatch: MonkeyPatch) -> None:
    fake_models(monkeypatch, analyst=False, builder=True)
    monkeypatch.setattr(launcher, "_ensure_database", lambda: True)
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(launcher, "_py", lambda module, *a: seen.append((module, *a)) or 0)
    assert launcher.cmd_graph(launcher.parse(["graph", "--limit", "5"])) == 0
    args = seen[0]
    assert args[:2] == ("georag.graph.cli", "build")
    assert launcher.BUILDER_MODEL in args and launcher.BUILDER_HOST in args
    assert args[args.index("--workers") + 1] == str(launcher.BUILDER_WORKERS)


def test_loading_analyst_counts_as_running(monkeypatch: MonkeyPatch) -> None:
    """Qwen3-14B ещё грузится (не отвечает), но видеопамять заняла: её тоже остановить."""
    log = fake_models(monkeypatch, analyst=False)
    monkeypatch.setattr(launcher, "_running", lambda service: service == "llm")
    with launcher._builder(launcher.parse(["graph", "--yes"])) as ready:
        assert ready
    assert log[0] == "--profile llm stop llm" and log[-1] == "start llm"


@pytest.mark.parametrize(("code", "embedded"), [(0, True), (2, False)])
def test_embed_runs_on_free_gpu(monkeypatch: MonkeyPatch, code: int, embedded: bool) -> None:
    """Имена, векторы и кластеры — после остановки строителя и до возврата Qwen3-14B:
    видеокарта в этот момент свободна. Прерванный graph (код 2) — без них."""
    log = fake_models(monkeypatch, analyst=True)
    monkeypatch.setattr(launcher, "_ensure_database", lambda: True)

    def py(module: str, *args: str) -> int:
        log.append(f"py {args[0]}" + (" --defer-embed" if "--defer-embed" in args else ""))
        return code if args[0] == "build" else 0

    monkeypatch.setattr(launcher, "_py", py)
    assert launcher.cmd_graph(launcher.parse(["graph", "--yes"])) == code
    expected = [
        "--profile llm stop llm",
        "start builder",
        "py build --defer-embed",
        "--profile builder stop builder",
    ]
    expected += ["py embed"] if embedded else []
    assert log == [*expected, "start llm"]
