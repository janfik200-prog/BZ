"""ГеоRAG — один вход для всех действий.

Запуск из папки проекта в любом терминале — cmd, PowerShell, Терминал Windows:

    python georag.py <команда> [ключи]

База и модель
    start                       поднять базу и модель Qwen3 в vLLM (Docker)
    stop                        остановить базу и модель, данные остаются
    status                      что сейчас работает
    init                        создать таблицы — один раз на новой машине
    stats                       сколько статей и фрагментов в базе

Статьи из интернета
    add "тема"                  найти и разобрать статьи по одной теме
    all                         то же по всем темам из config/topics.yaml
    all --topics topics-anabar.yaml     по темам из другого файла в config
    all --topics topics-check.yaml      темы под вопросы оценки (eval)
    clean                       показать статьи не по теме
    clean "тема"                то же, и Qwen3 проверяет каждую статью по теме
    clean --apply               убрать их в .local/acquired/_отсев
    ingest                      загрузить добытое в базу
    tidy                        убрать служебный текст (колонтитулы, благодарности) из базы

Поиск и чат-бот
    web                         в браузере: поиск, статьи, чат-бот и граф связей
    web --tg                    то же + Телеграм-бот (в одном окне, модель векторов общая)
    tg                          только Телеграм-бот
    tg --check                  бот молчит? проверить токен, доступ, кто писал
    serve                       то же без браузера (только адреса /api)
    search "запрос"             найти фрагменты статей в терминале
    formula ID                  формулы статьи: LaTeX, словами, какие фрагменты на них ссылаются
    formula ID 2                одна формула — с номером (2) в статье
    formula --find "…"          формулы по смыслу («как плотность зависит от пористости»)
    ask "вопрос"                спросить чат-бота в терминале (нужна модель)
                                (хотите подробно или кратко — так и напишите в вопросе)
    collect                     выписки по частым предметам заранее: широкий вопрос о них
                                не ждёт чтения статей (идёт и в ночном прогоне)
    collect --limit 40          то же для 40 самых упоминаемых предметов (обычно 20)
    collect --dry-run           что будет прочитано — без модели

Граф связей
    graph                       Qwen3 выписывает факты из статей (правила — config/graph-rules.txt)
    graph --limit 30            только 30 фрагментов — проба
    graph --redo                всё заново (после правки правил)
    graph --new                 что нашла модель и где ошибалась
    graph --recheck             перепроверить готовые факты новыми правилами кода (без модели)
    graph --confirm             подтвердить факты вопросами по цитате (идёт и после graph)
    graph --names               свести написания одного и того же по смыслу (идёт и после graph)
    graph --vectors             векторы фактов для поиска по связям (идёт и после graph)
    graph --types               тип связи и виды концов по цитате (идёт и после graph)
    synonyms                    предложения в словарь синонимов (config/synonyms.yaml)
    facts                       у чего больше всего фактов из статей
    facts "Анабарский щит"      все факты о нём с цитатами, файл CSV для Excel
                                (старое имя команды — dataset — тоже работает)
    questions                   вопросы оценки из фактов графа → .local/eval-graph.yaml
    gaps                        где базе не хватает статей → темы .local/topics-graph.yaml

Ночная автоматика
    schedule                    каждую ночь в 03:00: добыча → загрузка → граф → выписки
    schedule --at 01:30         то же в другое время
    unschedule                  убрать ночной прогон
    nightly                     сделать ночной прогон прямо сейчас

Проверка
    test                        проверить код (проверки — у разработчика, в .local/tests)
    lint                        проверки кода: ruff, black --check, mypy --strict (requirements-dev.txt)
    eval                        оценить базу и чат-бота по вопросам config/eval-questions.yaml
    eval --only-search          то же, только поиск — без модели
    eval --only s01,t01         только эти вопросы
    eval --no-graph             то же без графа — сравнить с обычным прогоном, что граф даёт
    eval --facts                то же с поиском по векторам фактов графа (по умолчанию выключен)
    eval --questions .local/eval-graph.yaml   по вопросам из графа (сделать: questions)
    help                        эта справка

Ключи
    --max-docs 5      сколько статей разбирать за тему (add, all)
    --limit 10        сколько результатов показать (search); сколько фрагментов (graph)
    --workers 8       сколько фрагментов у модели одновременно (graph; по умолчанию 8)
    --no-llm          не спрашивать модель, отбирать эвристикой (add, all)
    --force           обработать заново даже то, что уже было (add, all, ingest)
    --figures         описать рисунки статей моделью со зрением (add, all; нужна qwen3.5:9b)
    --no-formulas     разбирать без распознавания формул (add, all): быстрее, формулы — картинками

Окружение .venv запускатель находит сам: запускать можно обычным `python`,
нужные библиотеки всё равно возьмутся из .venv проекта.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# Только для этой машины — статьи, логи, ключи, кэши — в .local (georag/paths.py).
# Байт-код Python — туда же, а не папками __pycache__ по всему коду; дочерние
# команды получают это через переменную окружения.
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(ROOT / ".local" / "cache" / "python"))
sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
from georag.paths import LOCAL, LOGS  # noqa: E402

LOCAL.mkdir(exist_ok=True)


# --- настройки проекта ------------------------------------------------------ #
# E-mail для вежливого режима OpenAlex и Crossref. В коде его нет, чтобы не
# светить адрес в открытом репозитории: берётся из переменной GEORAG_MAILTO
# или из файла .local/mailto.txt.
def _mailto() -> str:
    env = os.environ.get("GEORAG_MAILTO", "").strip()
    if env:
        return env
    f = LOCAL / "mailto.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


MAILTO = _mailto()

# Ключ OpenAlex — с 13.02.2026 без него поиск в OpenAlex не работает. Бесплатный:
# openalex.org/settings/api. Хранится в .local/openalex_key.txt, дочерние
# команды получают его через переменную окружения.
_key_file = LOCAL / "openalex_key.txt"
if _key_file.exists() and not os.environ.get("GEORAG_OPENALEX_KEY"):
    os.environ["GEORAG_OPENALEX_KEY"] = _key_file.read_text(encoding="utf-8-sig").strip()
# Модель и адрес её сервера (vLLM) — одни на весь проект, из georag/llm.py (адрес
# можно поменять переменной GEORAG_LLM). Этот модуль лёгкий: только стандартная
# библиотека, поэтому импортируется и до перехода в .venv.
from georag.llm import DEFAULT_MODEL as MODEL  # noqa: E402
from georag.llm import LLM_HOST as LLM  # noqa: E402

TASK_NAME = "GeoRAG-acquire"  # имя ночной задачи в планировщике
NIGHTLY_MAX_DOCS = 10

# Проверки кода — только у разработчика, в .local/tests: в репозиторий не входят.
TESTS_DIR = LOCAL / "tests"
TESTS = {
    "smoke_test.py": "разбор PDF",
    "acquire_smoke_test.py": "добыча статей",
    "index_smoke_test.py": "база и поиск",
    "graph_smoke_test.py": "граф и факты",
    "chat_smoke_test.py": "чат-бот",
    "eval_smoke_test.py": "оценка",
    "tg_smoke_test.py": "Телеграм-бот",
}


# --------------------------------------------------------------------------- #
#  Окружение
# --------------------------------------------------------------------------- #
def _venv_python() -> Path | None:
    """Python из .venv проекта. Рядом с проектом тоже смотрим: туда его однажды уже ставили."""
    tail = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for base in (ROOT / ".venv", ROOT.parent / ".venv"):
        if (base / tail).exists():
            return base / tail
    return None


def _in_venv(python: Path) -> bool:
    """Запущены ли мы уже из этого окружения. Сравниваем папки окружений, а не
    файлы python: в .venv он бывает ссылкой на системный, и файлы совпадут."""
    try:
        return Path(sys.prefix).resolve() == python.parent.parent.resolve()
    except OSError:
        return False


def _relaunch_in_venv() -> None:
    """Если запущены не из .venv — перезапуститься в нём с теми же аргументами."""
    python = _venv_python()
    if python is None or _in_venv(python):
        return
    try:
        code = subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


def _need_venv() -> None:
    if _venv_python() is None:
        print(
            "Не найдено окружение .venv в папке проекта. Создать один раз:\n"
            "    python -m venv .venv\n"
            "    .venv\\Scripts\\python -m pip install -r requirements.txt\n"
            "Подробно — в ЗАПУСК.md.",
            file=sys.stderr,
        )
        sys.exit(1)


def _run(args: list[str], **kwargs: Any) -> int:
    """Запустить команду из папки проекта. Ctrl+C прерывает её, а не запускатель."""
    try:
        return subprocess.call(args, cwd=ROOT, **kwargs)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError:
        print(f"Не найдена программа: {args[0]}", file=sys.stderr)
        return 127


def _py(module: str, *args: str) -> int:
    return _run([sys.executable, "-m", module, *args])


# --------------------------------------------------------------------------- #
#  База в Docker
# --------------------------------------------------------------------------- #
def _docker(*args: str, quiet: bool = False) -> subprocess.CompletedProcess[str] | None:
    """docker compose. Модель (сервис llm) — в профиле llm: без --profile llm база
    поднимается одна, и загрузка статей не ждёт модель и не делит с ней видеокарту."""
    try:
        return subprocess.run(
            ["docker", "compose", *args], cwd=ROOT, capture_output=quiet, text=True
        )
    except FileNotFoundError:
        print(
            "Docker не найден. Установите и запустите Docker Desktop — " "см. ЗАПУСК.md.",
            file=sys.stderr,
        )
        return None


# Пользователь и база — из docker-compose.yml (POSTGRES_USER, POSTGRES_DB). pg_isready
# под другим именем засорял бы журнал базы строками «role … does not exist».
DB_USER = "georag"


def _db_state() -> tuple[str, str] | None:
    """Контейнер базы и его healthcheck: ("running", "healthy"), ("running", "starting"),
    ("exited", ""); ("", "") — контейнера нет. None — Docker недоступен (об этом уже
    сказано)."""
    done = _docker("ps", "--all", "--format", "{{.State}} {{.Health}}", "db", quiet=True)
    if done is None:
        return None
    if done.returncode != 0:
        print(
            "Docker не отвечает. Запустите Docker Desktop и дождитесь, "
            "пока значок кита перестанет мигать.",
            file=sys.stderr,
        )
        return None
    words = (done.stdout or "").split()
    return (words[0] if words else "", words[1] if len(words) > 1 else "")


def _pg_isready() -> bool:
    """Принимает ли база подключения: pg_isready внутри контейнера, как в healthcheck.
    По TCP (-h 127.0.0.1): при первом запуске образ создаёт базу на временном сервере,
    который слушает только сокет, — без -h «готова» прозвучало бы раньше времени."""
    check = ["pg_isready", "-q", "-h", "127.0.0.1", "-U", DB_USER, "-d", DB_USER]
    done = _docker("exec", "-T", "db", *check, quiet=True)
    return done is not None and done.returncode == 0


def _wait_database(limit: float = 60) -> bool:
    """Дождаться, пока база примет подключения: опрос раз в секунду, не дольше limit
    секунд. Только docker CLI и стандартная библиотека: start идёт и без .venv.

    Готова — когда healthcheck контейнера говорит healthy. Пока он молчит (первая его
    проверка — через 10 с после старта) или если его нет, спрашиваем pg_isready сами.
    Контейнер остановился (exited, dead) — до предела не ждём."""
    started = time.monotonic()
    told = False
    while True:
        db = _db_state()
        if db is None:
            return False
        state, health = db
        if state == "running" and (health == "healthy" or _pg_isready()):
            if told:
                print(f"База готова ({time.monotonic() - started:.0f} с).")
            return True
        if not state:
            print("Контейнера базы нет. Поднять: python georag.py start", file=sys.stderr)
            return False
        if state in ("exited", "dead"):
            print(
                f"База остановилась ({state}). Причина — в журнале: docker logs georag-db",
                file=sys.stderr,
            )
            return False
        if time.monotonic() - started >= limit:
            print(
                f"База не ответила за {limit:.0f} с. Причина — в журнале: docker logs "
                "georag-db; если база ещё запускается, повторите команду через минуту.",
                file=sys.stderr,
            )
            return False
        if not told:
            print("Жду, пока база будет готова…", flush=True)
            told = True
        time.sleep(1)


def _ensure_database() -> bool:
    """Поднять базу, если она не запущена, и дождаться, пока она примет подключения.
    Ложь — если Docker недоступен или база не поднялась."""
    db = _db_state()
    if db is None:
        return False
    if db == ("running", "healthy"):
        return True
    if db[0] not in ("running", "restarting"):
        print("База не запущена. Поднимаю...")
        up = _docker("up", "-d")
        if up is None or up.returncode != 0:
            return False
    return _wait_database()


def _llm_up(timeout: int = 5) -> bool:
    """Отвечает ли сервер модели (vLLM): /v1/models."""
    try:
        with urllib.request.urlopen(f"{LLM}/v1/models", timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_llm(wait: bool = True, limit: int = 1800) -> bool:
    """Поднять модель в vLLM, если не отвечает. wait — дождаться готовности: модель
    грузится 1–2 минуты, а в первый раз ещё и скачивается (около 10 ГБ)."""
    if _llm_up():
        return True
    if not LLM.startswith(("http://localhost", "http://127.0.0.1")):
        print(f"Модель на {LLM} не отвечает — её сервер запускают там.", file=sys.stderr)
        return False
    print("Модель не запущена. Поднимаю vLLM...")
    up = _docker("--profile", "llm", "up", "-d", "llm")
    if up is None or up.returncode != 0:
        return False
    if not wait:
        print("Модель грузится 1–2 минуты; пока она грузится, работает поиск.")
        return True
    started = time.monotonic()
    while time.monotonic() - started < limit:
        if _llm_up():
            print(f"Модель готова ({time.monotonic() - started:.0f} с).")
            return True
        time.sleep(5)
    print("Модель не поднялась. Журнал: docker logs georag-llm", file=sys.stderr)
    return False


# --------------------------------------------------------------------------- #
#  Команды
# --------------------------------------------------------------------------- #
def _acquire(opts: argparse.Namespace, topic_args: list[str]) -> int:
    args = [
        *topic_args,
        "--max-docs",
        str(opts.max_docs),
        "--device",
        "cuda",
        *(["--mailto", MAILTO] if MAILTO else []),
        "--model",
        MODEL,
    ]
    if opts.no_llm:
        args.append("--no-llm")
    if opts.force:
        args.append("--force")
    if opts.figures:
        args.append("--figures")
    if opts.no_formulas:
        args.append("--no-formulas")
    return _py("georag.acquire.cli", *args)


def cmd_start(opts: argparse.Namespace) -> int:
    up = _docker("--profile", "llm", "up", "-d")
    if up is None or up.returncode != 0:
        return 1
    print()
    ready = _wait_database()
    _docker("--profile", "llm", "ps")
    if not ready:
        return 1
    return 0 if _ensure_llm() else 1


def cmd_stop(opts: argparse.Namespace) -> int:
    done = _docker("--profile", "llm", "stop")
    if done is not None and done.returncode == 0:
        print("База и модель остановлены. Данные на месте, видеокарта свободна.")
        return 0
    return 1


def cmd_status(opts: argparse.Namespace) -> int:
    done = _docker("--profile", "llm", "ps")
    print(f"Модель на {LLM}: " + ("отвечает" if _llm_up() else "не отвечает"))
    return 0 if done is not None and done.returncode == 0 else 1


def cmd_init(opts: argparse.Namespace) -> int:
    return _py("georag.index.cli", "init") if _ensure_database() else 1


def cmd_stats(opts: argparse.Namespace) -> int:
    return _py("georag.index.cli", "stats") if _ensure_database() else 1


def cmd_add(opts: argparse.Namespace) -> int:
    if not opts.text:
        print('Нужна тема: python georag.py add "выделение рудных узлов"', file=sys.stderr)
        return 1
    return _acquire(opts, [opts.text])


def _topics_file(name: str) -> str | None:
    """Файл тем: как написали, из папки config или .local (темы от gaps) —
    «topics-anabar.yaml» хватает."""
    for candidate in (Path(name), Path("config") / name, LOCAL.relative_to(ROOT) / name):
        if (ROOT / candidate).is_file():
            return candidate.as_posix()
    return None


def cmd_all(opts: argparse.Namespace) -> int:
    topics = _topics_file(opts.topics)
    if topics is None:
        print(f"Нет файла тем: {opts.topics} (темы лежат в папке config)", file=sys.stderr)
        return 1
    return _acquire(opts, ["--topics", topics])


def cmd_clean(opts: argparse.Namespace) -> int:
    args = ["--apply"] if opts.apply else []
    if opts.text:
        args += ["--topic", opts.text, "--model", MODEL]
    return _py("georag.acquire.clean", *args)


def cmd_ingest(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["ingest"]
    if opts.force:
        args.append("--force")
    return _py("georag.index.cli", *args)


def cmd_tidy(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["tidy"]
    return _py("georag.index.cli", *args)


def cmd_search(opts: argparse.Namespace) -> int:
    if not opts.text:
        print('Нужен запрос: python georag.py search "рудные узлы"', file=sys.stderr)
        return 1
    if not _ensure_database():
        return 1
    args = ["search", opts.text, "--limit", str(opts.limit or 10)]
    return _py("georag.index.cli", *args)


def cmd_web(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    _ensure_llm(wait=False)
    return _py(
        "georag.web.server",
        "--model",
        MODEL,
        *(["--tg"] if opts.tg else []),
    )


def cmd_serve(opts: argparse.Namespace) -> int:
    """Тот же сервер, что web, но браузер не открывается (только адреса /api)."""
    if not _ensure_database():
        return 1
    _ensure_llm(wait=False)
    args = ["--no-browser", "--model", MODEL]
    return _py("georag.web.server", *args, *(["--tg"] if opts.tg else []))


def cmd_graph(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    if opts.new:
        return _py("georag.graph.cli", "report")
    if opts.recheck:
        return _py("georag.graph.cli", "recheck")
    if opts.names:
        return _py("georag.graph.cli", "names", *(["--redo"] if opts.redo else []))
    if opts.vectors:
        return _py("georag.graph.cli", "vectors", *(["--redo"] if opts.redo else []))
    workers = ["--workers", str(opts.workers)] if opts.workers else []
    if not _ensure_llm():
        return 1
    if opts.types:
        redo = ["--redo"] if opts.redo else []
        return _py("georag.graph.cli", "types", "--model", MODEL, *redo, *workers)
    if opts.confirm:
        extra = ["--limit", str(opts.limit)] if opts.limit else []
        return _py("georag.graph.cli", "confirm", "--model", MODEL, *extra, *workers)
    args = ["build", "--model", MODEL] + (["--redo"] if opts.redo else []) + workers
    if opts.limit:
        args += ["--limit", str(opts.limit)]
    return _py("georag.graph.cli", *args)


def cmd_synonyms(opts: argparse.Namespace) -> int:
    if not _ensure_database() or not _ensure_llm():
        return 1
    return _py(
        "georag.graph.cli",
        "synonyms",
        "--model",
        MODEL,
    )


def cmd_verify(opts: argparse.Namespace) -> int:
    print(
        "Проверки понятий больше нет: граф строят факты с цитатами, каждый проверяет код "
        "сразу. Строить граф: python georag.py graph"
    )
    return 0


def cmd_tg(opts: argparse.Namespace) -> int:
    if opts.check:  # проверка токена и доступа: база и модели не нужны
        return _py("georag.tg.bot", "--check")
    if not _ensure_database():
        return 1
    _ensure_llm(wait=False)
    return _py("georag.tg.bot", "--model", MODEL)


def cmd_dataset(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py("georag.graph.cli", "dataset", *(["--name", opts.text] if opts.text else []))


def cmd_questions(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py(
        "georag.graph.cli",
        "questions",
        "--model",
        MODEL,
        *(["--limit", str(opts.limit)] if opts.limit else []),
        *(["--no-llm"] if opts.no_llm else []),
    )


def cmd_gaps(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py("georag.graph.cli", "gaps", *(["--limit", str(opts.limit)] if opts.limit else []))


def cmd_ask(opts: argparse.Namespace) -> int:
    if not opts.text:
        print('Нужен вопрос: python georag.py ask "как выделяют рудные узлы?"', file=sys.stderr)
        return 1
    if not _ensure_database() or not _ensure_llm():
        return 1
    return _py(
        "georag.chat.cli",
        opts.text,
        "--model",
        MODEL,
    )


def cmd_collect(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    # Векторы здесь нужны на одно имя за предмет — видеокарту оставляем модели.
    args = ["--model", MODEL, "--device", "cpu"]
    if opts.limit is not None:
        args += ["--top", str(opts.limit)]
    if opts.dry_run:
        return _py("georag.chat.collect", "--dry-run", *args)
    if not _ensure_llm():
        return 1
    return _py("georag.chat.collect", *args)


def cmd_eval(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    if not opts.only_search and not _ensure_llm():
        return 1
    args = ["--model", MODEL]
    if opts.only_search:
        args.append("--only-search")
    if opts.only:
        args += ["--only", opts.only]
    if opts.questions:
        args += ["--questions", opts.questions]
    if opts.no_graph:
        args.append("--no-graph")
    if opts.facts:
        args.append("--facts")
    return _py("georag.evaluation", *args)


def cmd_lint(opts: argparse.Namespace) -> int:
    """ruff, black --check и mypy --strict.
    Пакет georag и файл georag.py проверяются mypy отдельно: у них одно имя."""
    # Проверки кода лежат в .local (исключена в pyproject) — их файлы передаются явно;
    # tests/ из репозитория — тоже (mypy его не проверяет: проверки без типов).
    tests = [str(p) for p in sorted(TESTS_DIR.rglob("*.py"))]
    tests += ["tests"] if (ROOT / "tests").is_dir() else []
    steps = [
        ("ruff", ["-m", "ruff", "check", "georag", "georag.py", *tests]),
        ("black", ["-m", "black", "--check", "georag", "georag.py", *tests]),
        ("mypy --strict: пакет", ["-m", "mypy", "-p", "georag"]),
        ("mypy --strict: georag.py", ["-m", "mypy", "georag.py"]),
    ]
    failed = []
    for title, args in steps:
        print(f"== {title}", flush=True)
        if subprocess.run([sys.executable, *args], cwd=ROOT).returncode != 0:
            failed.append(title)
    print("\nВсе проверки кода прошли." if not failed else f"\nНе прошло: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_test(opts: argparse.Namespace) -> int:
    """Проверки кода. Идут на выдуманных примерах, база и модель не нужны и не трогаются.

    На экран — одна строка на часть; подробности — только у того, что не прошло
    (или со всеми подробностями: python georag.py test --verbose)."""
    if not TESTS_DIR.is_dir():
        print("Проверок кода в этой копии нет: они только у разработчика (.local/tests).")
        return 0
    print("Проверка кода на выдуманных примерах — ваша база, статьи и граф не трогаются.\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    failed = []
    for file, name in TESTS.items():
        test = str(TESTS_DIR / file)
        if opts.verbose:
            code = _run([sys.executable, test])
            out = ""
        else:
            try:
                proc = subprocess.run(
                    [sys.executable, test],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except KeyboardInterrupt:
                return 130
            code, out = proc.returncode, proc.stdout
        total = re.findall(r"Итого: (\d+) пройдено", out)
        mark = "OK  " if code == 0 else "СБОЙ"
        print(f"  [{mark}] {name}" + (f" — проверок {total[-1]}" if total and code == 0 else ""))
        if code != 0:
            failed.append(name)
            if out:
                print("\n".join("        " + line for line in out.rstrip().splitlines()[-40:]))
    if failed:
        print(
            "\nНе прошли: " + ", ".join(failed) + ". Пришлите этот вывод целиком.", file=sys.stderr
        )
        return 1
    print("\nВсе проверки прошли.")
    return 0


# --------------------------------------------------------------------------- #
#  Ночная автоматика
# --------------------------------------------------------------------------- #
def cmd_nightly(opts: argparse.Namespace) -> int:
    """То, что запускает планировщик: добыча → загрузка в базу → граф → выписки.

    Всё пишется в .local/logs/run-ДАТА.txt; логи старше месяца удаляются, чтобы папка
    не росла бесконечно. Если модель не поднялась, отбор статей идёт эвристикой —
    прогон не останавливается.
    """
    logs = LOGS
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"run-{datetime.now():%Y%m%d-%H%M%S}.txt"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    with open(log_path, "w", encoding="utf-8") as log:

        def say(line: str) -> None:
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        def step(title: str, args: list[str]) -> int:
            say(f"\n=== {title} · {datetime.now():%H:%M:%S} ===")
            try:
                proc = subprocess.Popen(
                    args,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                say(f"не запустилось: {exc}")
                return 1
            for line in proc.stdout or []:
                say(line.rstrip("\n"))
            code = proc.wait()
            say(f"--- код возврата {code}")
            return code

        say(f"=== ночной прогон {datetime.now():%Y-%m-%d %H:%M:%S} ===")
        acquire = [
            "-m",
            "georag.acquire.cli",
            "--topics",
            "config/topics.yaml",
            "--device",
            "cuda",
            "--max-docs",
            str(NIGHTLY_MAX_DOCS),
            *(["--mailto", MAILTO] if MAILTO else []),
            "--model",
            MODEL,
        ]
        if not _ensure_database() or not _ensure_llm():
            say("Модель не отвечает — отбор пойдёт эвристикой")
            acquire.append("--no-llm")

        codes = [step("добыча статей", [sys.executable, *acquire])]
        if _ensure_database():
            codes.append(
                step("загрузка в базу", [sys.executable, "-m", "georag.index.cli", "ingest"])
            )
            # Граф: Qwen3 выписывает факты из новых фрагментов, не больше 1000 за ночь —
            # около получаса при 4 запросах сразу (~2 с на фрагмент с подтверждением).
            # За ночь добывается до NIGHTLY_MAX_DOCS статей — это ~370 фрагментов; при
            # прежних 300 очередь непройденных росла бы от ночи к ночи.
            codes.append(
                step(
                    "граф связей",
                    [
                        sys.executable,
                        "-m",
                        "georag.graph.cli",
                        "build",
                        "--model",
                        MODEL,
                        "--limit",
                        "1000",
                    ],
                )
            )
            # Выписки по предметам, о которых спрашивали, и самым упоминаемым — с новыми
            # статьями ночи (chat/collect.py): иначе первый широкий вопрос о предмете ждёт
            # чтения до 30 статей, 1,5–2 минуты. Не дольше часа; не успел — в следующую ночь.
            codes.append(
                step(
                    "выписки по частым предметам",
                    [
                        sys.executable,
                        "-m",
                        "georag.chat.collect",
                        "--model",
                        MODEL,
                        "--device",
                        "cpu",
                        "--minutes",
                        "60",
                    ],
                )
            )
        else:
            say("База не поднялась — загрузка, граф и выписки пропущены до следующего раза")
            codes.append(1)
        say(f"\n=== конец {datetime.now():%Y-%m-%d %H:%M:%S} ===")

    cutoff = datetime.now() - timedelta(days=30)
    for old in logs.glob("run-*.txt"):
        try:
            if datetime.fromtimestamp(old.stat().st_mtime) < cutoff:
                old.unlink()
        except OSError:
            pass
    return max(codes)


def task_xml(python: Path, at: str) -> str:
    """Описание ночной задачи для планировщика Windows.

    Через XML, а не короткими ключами schtasks: только так задаётся «запустить,
    как только компьютер проснётся, если время пропущено» и «не останавливать
    на батарее». Без этого ноутбук, спавший в 03:00, пропускал бы прогон.
    """
    from xml.sax.saxutils import escape

    hour, minute = (int(x) for x in at.split(":"))
    start = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>ГеоRAG: ночное пополнение базы знаний — добыча, загрузка, граф, выписки</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start:%Y-%m-%dT%H:%M:%S}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT6H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(python))}</Command>
      <Arguments>"{escape(str(Path(__file__).resolve()))}" nightly</Arguments>
      <WorkingDirectory>{escape(str(ROOT))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def cmd_schedule(opts: argparse.Namespace) -> int:
    try:
        hour, minute = (int(x) for x in opts.at.split(":"))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError
    except ValueError:
        print(f"Время — в виде ЧЧ:ММ, например 03:00. Пришло: {opts.at}", file=sys.stderr)
        return 1
    python = _venv_python() or Path(sys.executable)
    if os.name != "nt":
        print(
            "На этой системе планировщик Windows недоступен. Строка для crontab -e:\n"
            f'    {minute} {hour} * * * "{python}" "{Path(__file__).resolve()}" nightly'
        )
        return 0

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "georag-task.xml"
        xml.write_text(task_xml(python, opts.at), encoding="utf-16")
        done = _run(["schtasks", "/Create", "/TN", TASK_NAME, "/XML", str(xml), "/F"])
    if done == 0:
        print(
            f"\nГотово: каждую ночь в {opts.at:0>5} — добыча, загрузка в базу и граф.\n"
            "Если компьютер в это время спал, прогон случится, когда он проснётся.\n"
            "Логи — в папке .local\\logs. Проверить сразу, не дожидаясь ночи:\n"
            "    python georag.py nightly"
        )
    return done


def cmd_unschedule(opts: argparse.Namespace) -> int:
    if os.name != "nt":
        print("Уберите строку с georag.py из crontab -e.")
        return 0
    done = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if done == 0:
        print("Ночной прогон убран.")
    return done


def cmd_help(opts: argparse.Namespace) -> int:
    print(__doc__)
    return 0


COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "status": cmd_status,
    "init": cmd_init,
    "stats": cmd_stats,
    "add": cmd_add,
    "all": cmd_all,
    "clean": cmd_clean,
    "ingest": cmd_ingest,
    "tidy": cmd_tidy,
    "search": cmd_search,
    "web": cmd_web,
    "serve": cmd_serve,
    "graph": cmd_graph,
    "verify": cmd_verify,
    "synonyms": cmd_synonyms,
    "ask": cmd_ask,
    "collect": cmd_collect,
    "eval": cmd_eval,
    "dataset": cmd_dataset,
    "facts": cmd_dataset,
    "questions": cmd_questions,
    "gaps": cmd_gaps,
    "tg": cmd_tg,
    "test": cmd_test,
    "lint": cmd_lint,
    "nightly": cmd_nightly,
    "schedule": cmd_schedule,
    "unschedule": cmd_unschedule,
    "help": cmd_help,
}
# Эти работают без .venv: им нужны только Docker и планировщик.
NO_VENV = {"start", "stop", "status", "help", "unschedule"}


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python georag.py", add_help=False)
    p.add_argument("command", nargs="?", default="help")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--max-docs", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--only-search", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--questions", default="")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--new", action="store_true")
    p.add_argument("--recheck", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--names", action="store_true")
    p.add_argument("--vectors", action="store_true")
    p.add_argument("--types", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--no-formulas", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--facts", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tg", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--at", default="03:00")
    p.add_argument("--topics", default="config/topics.yaml")
    p.add_argument("-h", "--help", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Сам файл запускают и старым Python — ему нужно понятное сообщение, а не ошибка разбора.
    if sys.version_info < (3, 12):  # noqa: UP036
        print(
            f"Нужен Python 3.12 или новее, а запущен {sys.version.split()[0]} — ЗАПУСК.md, раздел 1.",
            file=sys.stderr,
        )
        return 1
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0].lower() == "formula":
        # У команды свои ключи (doc_id, номер, --find, --json) — передаются как есть.
        _need_venv()
        if not _ensure_database():
            return 1
        return _py("georag.index.formulas", *argv[1:])
    opts = parse(argv)
    if opts.help:
        opts.command = "help"
    command = COMMANDS.get(opts.command.lower())
    if command is None:
        print(f"Нет такой команды: {opts.command}. Список — python georag.py help", file=sys.stderr)
        return 1
    if opts.command.lower() not in NO_VENV:
        _need_venv()
    return command(opts)


if __name__ == "__main__":
    _relaunch_in_venv()
    raise SystemExit(main())
