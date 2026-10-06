"""ГеоRAG — один вход для всех действий.

Запуск из папки проекта в любом терминале — cmd, PowerShell, Терминал Windows:

    python georag.py <команда> [ключи]

База
    start                       поднять базу (Docker)
    stop                        остановить базу, данные остаются
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
    clean --apply               убрать их в data/acquired/_отсев
    ingest                      загрузить добытое в базу
    tidy                        убрать служебный текст (колонтитулы, благодарности) из базы

Поиск и чат-бот
    web                         в браузере: поиск, статьи, чат-бот и граф связей
    web --tg                    то же + Телеграм-бот (в одном окне, модель векторов общая)
    tg                          только Телеграм-бот
    tg --check                  бот молчит? проверить токен, доступ, кто писал
    serve                       то же без браузера (только адреса /api)
    search "запрос"             найти фрагменты статей в терминале
    ask "вопрос"                спросить чат-бота в терминале (нужна Ollama)
                                (хотите подробно или кратко — так и напишите в вопросе)

Граф связей
    graph                       Qwen3 выписывает факты из статей (правила — config/graph-rules.txt)
    graph --limit 30            только 30 фрагментов — проба
    graph --redo                всё заново (после правки правил)
    graph --new                 что нашла модель и где ошибалась
    graph --recheck             перепроверить готовые факты новыми правилами кода (без модели)
    graph --confirm             подтвердить факты вопросами по цитате (идёт и после graph)
    synonyms                    предложения в словарь синонимов (config/synonyms.yaml)
    facts                       у чего больше всего фактов из статей
    facts "Анабарский щит"      все факты о нём с цитатами, файл CSV для Excel
                                (старое имя команды — dataset — тоже работает)
    questions                   вопросы оценки из фактов графа → config/eval-graph.yaml
    gaps                        где базе не хватает статей → темы config/topics-graph.yaml

Ночная автоматика
    schedule                    каждую ночь в 03:00: добыча → загрузка → граф
    schedule --at 01:30         то же в другое время
    unschedule                  убрать ночной прогон
    nightly                     сделать ночной прогон прямо сейчас

Проверка
    test                        проверить код (на выдуманных примерах, база не трогается)
    lint                        проверки кода: ruff, black --check, mypy --strict
    eval                        оценить базу и чат-бота по вопросам config/eval-questions.yaml
    eval --only-search          то же, только поиск — без модели
    eval --only s01,t01         только эти вопросы
    eval --questions config/eval-graph.yaml   по вопросам из графа (сделать: questions)
    help                        эта справка

Ключи
    --max-docs 5      сколько статей разбирать за тему (add, all)
    --limit 10        сколько результатов показать (search); сколько фрагментов (graph)
    --no-llm          не спрашивать модель, отбирать эвристикой (add, all)
    --force           обработать заново даже то, что уже было (add, all, ingest)
    --ollama          считать векторы через Ollama, а не на видеокарте

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


# --- настройки проекта ------------------------------------------------------ #
# E-mail для вежливого режима OpenAlex и Crossref. В коде его нет, чтобы не
# светить адрес в открытом репозитории: берётся из переменной GEORAG_MAILTO
# или из файла mailto.txt рядом с этим файлом (он в .gitignore).
def _mailto() -> str:
    env = os.environ.get("GEORAG_MAILTO", "").strip()
    if env:
        return env
    f = ROOT / "mailto.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


MAILTO = _mailto()

# Ключ OpenAlex — с 13.02.2026 без него поиск в OpenAlex не работает. Бесплатный:
# openalex.org/settings/api. Хранится в openalex_key.txt (в .gitignore), дочерние
# команды получают его через переменную окружения.
_key_file = ROOT / "openalex_key.txt"
if _key_file.exists() and not os.environ.get("GEORAG_OPENALEX_KEY"):
    os.environ["GEORAG_OPENALEX_KEY"] = _key_file.read_text(encoding="utf-8-sig").strip()
# Модель и адрес Ollama — одни на весь проект, из georag/llm.py (адрес можно
# поменять переменной GEORAG_OLLAMA). Этот модуль лёгкий: только стандартная
# библиотека, поэтому импортируется и до перехода в .venv.
from georag.llm import DEFAULT_MODEL as MODEL  # noqa: E402
from georag.llm import OLLAMA_HOST as OLLAMA  # noqa: E402

TASK_NAME = "GeoRAG-acquire"  # имя ночной задачи в планировщике
NIGHTLY_MAX_DOCS = 10

TESTS = [
    "tests/smoke_test.py",
    "tests/acquire_smoke_test.py",
    "tests/index_smoke_test.py",
    "tests/graph_smoke_test.py",
    "tests/chat_smoke_test.py",
    "tests/eval_smoke_test.py",
    "tests/tg_smoke_test.py",
]


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


def _ensure_database() -> bool:
    """Поднять базу, если она не запущена. Ложь — если Docker недоступен."""
    state = _docker("ps", "--format", "{{.State}}", quiet=True)
    if state is None:
        return False
    if state.returncode != 0:
        print(
            "Docker не отвечает. Запустите Docker Desktop и дождитесь, "
            "пока значок кита перестанет мигать.",
            file=sys.stderr,
        )
        return False
    if "running" not in (state.stdout or ""):
        print("База не запущена. Поднимаю...")
        up = _docker("up", "-d")
        if up is None or up.returncode != 0:
            return False
        time.sleep(5)
    return True


# --------------------------------------------------------------------------- #
#  Команды
# --------------------------------------------------------------------------- #
def _acquire(opts: argparse.Namespace, topic_args: list[str]) -> int:
    args = [
        *topic_args,
        "--max-docs",
        str(opts.max_docs),
        "--no-ocr",
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
    return _py("georag.acquire.cli", *args)


def cmd_start(opts: argparse.Namespace) -> int:
    up = _docker("up", "-d")
    if up is None or up.returncode != 0:
        return 1
    print("\nЖду, пока база будет готова...")
    time.sleep(5)
    _docker("ps")
    return 0


def cmd_stop(opts: argparse.Namespace) -> int:
    done = _docker("stop")
    if done is not None and done.returncode == 0:
        print("База остановлена. Данные на месте.")
        return 0
    return 1


def cmd_status(opts: argparse.Namespace) -> int:
    done = _docker("ps")
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
    """Файл тем: как написали, или из папки config — «topics-anabar.yaml» хватает."""
    for candidate in (Path(name), Path("config") / name):
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
    if opts.ollama:
        args += ["--embedder", "ollama"]
    return _py("georag.index.cli", *args)


def cmd_tidy(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["tidy"]
    if opts.ollama:
        args += ["--embedder", "ollama"]
    return _py("georag.index.cli", *args)


def cmd_search(opts: argparse.Namespace) -> int:
    if not opts.text:
        print('Нужен запрос: python georag.py search "рудные узлы"', file=sys.stderr)
        return 1
    if not _ensure_database():
        return 1
    args = ["search", opts.text, "--limit", str(opts.limit or 10)]
    if opts.ollama:
        args += ["--embedder", "ollama"]
    return _py("georag.index.cli", *args)


def cmd_web(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py(
        "georag.web.server",
        "--model",
        MODEL,
        *(["--embedder", "ollama"] if opts.ollama else []),
        *(["--tg"] if opts.tg else []),
    )


def cmd_serve(opts: argparse.Namespace) -> int:
    """Тот же сервер, что web, но браузер не открывается (только адреса /api)."""
    if not _ensure_database():
        return 1
    args = ["--no-browser", "--model", MODEL] + (["--embedder", "ollama"] if opts.ollama else [])
    return _py("georag.web.server", *args, *(["--tg"] if opts.tg else []))


def cmd_graph(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    if opts.new:
        return _py("georag.graph.cli", "report")
    if opts.recheck:
        return _py("georag.graph.cli", "recheck")
    if opts.confirm:
        extra = ["--limit", str(opts.limit)] if opts.limit else []
        return _py("georag.graph.cli", "confirm", "--model", MODEL, *extra)
    args = ["build", "--model", MODEL] + (["--redo"] if opts.redo else [])
    if opts.limit:
        args += ["--limit", str(opts.limit)]
    return _py("georag.graph.cli", *args)


def cmd_synonyms(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py(
        "georag.graph.cli",
        "synonyms",
        "--model",
        MODEL,
        *(["--embedder", "ollama"] if opts.ollama else []),
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
    return _py(
        "georag.tg.bot", "--model", MODEL, *(["--embedder", "ollama"] if opts.ollama else [])
    )


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
    if not _ensure_database():
        return 1
    return _py(
        "georag.chat.cli",
        opts.text,
        "--model",
        MODEL,
        *(["--embedder", "ollama"] if opts.ollama else []),
    )


def cmd_eval(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["--model", MODEL]
    if opts.only_search:
        args.append("--only-search")
    if opts.only:
        args += ["--only", opts.only]
    if opts.questions:
        args += ["--questions", opts.questions]
    if opts.ollama:
        args += ["--embedder", "ollama"]
    return _py("georag.evaluation", *args)


TEST_NAMES = {
    "tests/smoke_test.py": "разбор PDF",
    "tests/acquire_smoke_test.py": "добыча статей",
    "tests/index_smoke_test.py": "база и поиск",
    "tests/graph_smoke_test.py": "граф и факты",
    "tests/chat_smoke_test.py": "чат-бот",
    "tests/eval_smoke_test.py": "оценка",
    "tests/tg_smoke_test.py": "Телеграм-бот",
}


def cmd_lint(opts: argparse.Namespace) -> int:
    """ruff, black --check и mypy --strict — раздел 8.2 docs/СИСТЕМНЫЙ-ПРОМПТ.md.
    Пакет georag и файл georag.py проверяются mypy отдельно: у них одно имя."""
    steps = [
        ("ruff", ["-m", "ruff", "check", "georag", "georag.py", "tests"]),
        ("black", ["-m", "black", "--check", "georag", "georag.py", "tests"]),
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
    """Проверки кода. Идут на выдуманных примерах, база и Ollama не нужны и не трогаются.

    На экран — одна строка на часть; подробности — только у того, что не прошло
    (или со всеми подробностями: python georag.py test --verbose)."""
    print("Проверка кода на выдуманных примерах — ваша база, статьи и граф не трогаются.\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    failed = []
    for test in TESTS:
        name = TEST_NAMES.get(test, test)
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
def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5):
            return True
    except Exception:  # noqa: BLE001
        return False


def cmd_nightly(opts: argparse.Namespace) -> int:
    """То, что запускает планировщик: добыча → загрузка в базу → граф.

    Всё пишется в logs/run-ДАТА.txt; логи старше месяца удаляются, чтобы папка
    не росла бесконечно. Если Ollama не поднята, отбор статей идёт эвристикой —
    прогон не останавливается.
    """
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
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
        if not _ollama_up():
            say("Ollama не отвечает — отбор пойдёт эвристикой")
            acquire.append("--no-llm")

        codes = [step("добыча статей", [sys.executable, *acquire])]
        if _ensure_database():
            codes.append(
                step("загрузка в базу", [sys.executable, "-m", "georag.index.cli", "ingest"])
            )
            # Граф: Qwen3 выписывает факты из новых фрагментов (не больше 300 за ночь).
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
                        "300",
                    ],
                )
            )
        else:
            say("База не поднялась — загрузка и граф пропущены до следующего раза")
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
    <Description>ГеоRAG: ночное пополнение базы знаний — добыча, загрузка, граф</Description>
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
            "Логи — в папке logs. Проверить сразу, не дожидаясь ночи:\n"
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
    p.add_argument("--ollama", action="store_true")
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
    p.add_argument("--tg", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--at", default="03:00")
    p.add_argument("--topics", default="config/topics.yaml")
    p.add_argument("-h", "--help", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opts = parse(sys.argv[1:] if argv is None else argv)
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
