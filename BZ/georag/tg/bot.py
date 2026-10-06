"""Телеграм-бот базы знаний. Работает рядом со страницей в браузере, а не вместо неё.

    python georag.py tg            только бот
    python georag.py web --tg      страница в браузере и бот сразу (модель векторов одна)

Интерфейс — кнопками (как выглядит — georag/tg/ui.py):

* меню внизу экрана: Поиск, Статьи, Факты, Новый разговор, Помощь.
  Любой другой текст — вопрос чат-боту;
* под ответом: фрагменты [1]…[8] (нажал — фрагмент целиком), «Как искал»,
  «Новый разговор»; пока бот пишет — «Остановить»;
* поиск и список статей листаются кнопками; у статьи — ссылка на издателя и
  её фрагменты; у датасета — таблица для Excel и «Спросить бота».

Настроек у ответа нет: подробно или кратко — человек пишет в самом вопросе,
модель так и отвечает. Команды тоже работают: /search запрос, /articles,
/dataset [название], /f 2, /new, /stats, /help.

Без новых библиотек: Bot API Телеграма — обычный HTTP, ходим через requests.
Бот сам спрашивает у Телеграма новые сообщения (long polling), поэтому ни
открытого порта, ни внешнего адреса у компьютера не нужно. Ответ модели
пишется в отдельном потоке — бот тем временем принимает нажатия, поэтому
«Остановить» срабатывает сразу.

Доступ — telegram_users.txt, по одной записи в строке, # — комментарий:

* id человека — кому можно;
* строка «*» — бот открыт всем, без ограничений.

Без «*» чужим бот отвечает только их id (не чаще раза в 10 минут). Файл
перечитывается на каждом сообщении, перезапускать бота после правки не нужно.

Модель пишет ответы по одному (MAX_PARALLEL): кто спросил, пока она занята,
получает «в очереди» с местом и ждёт. Вторая копия бота на том же компьютере не
запускается: две копии забирают одни и те же сообщения, и на один /start
приходило по десять ответов.

Токен бота (выдаёт @BotFather) — в telegram_token.txt или в GEORAG_TG_TOKEN.
Оба файла в .gitignore: в репозиторий не попадают.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ui
from .ui import (  # noqa: F401 — для внешних
    LIMIT,
    answer_html,
    esc,
    search_notes,
    source_line,
    split,
)

ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = ROOT / "telegram_token.txt"
USERS_FILE = ROOT / "telegram_users.txt"
API = os.environ.get("GEORAG_TG_API", "").strip() or "https://api.telegram.org"

EDIT_EVERY = 1.5  # как часто обновлять сообщение, пока модель пишет, с
POLL_TIMEOUT = 50  # сколько секунд Телеграм держит запрос новых сообщений
HISTORY_TURNS = 4  # сколько прошлых реплик помнит разговор
KEEP = 20  # сколько последних ответов и списков помнить для кнопок
SEARCH_HITS = 20
HELP = ui.HELP

OPEN_ACCESS = {"*", "все", "всем", "all", "everyone"}  # строка в telegram_users.txt: открыт всем
MAX_PARALLEL = 1  # ответов модели одновременно: видеокарта одна
MAX_SESSIONS = 300  # разговоров в памяти; старые забываются
MAX_TERRITORIES = 400  # сущностей в разговоре (для кнопок картинки связей)
DENIED_EVERY = 600  # как часто чужому напоминать его id, с
SEEN_UPDATES = 1000  # сколько последних update_id помнить, чтобы не взять дважды

COMMANDS = [
    ("search", "поиск фрагментов статей"),
    ("articles", "статьи в базе"),
    ("facts", "факты о чём-то из статей, с цитатами"),
    ("stats", "что в базе и готова ли модель"),
    ("new", "новый разговор"),
    ("help", "что умеет бот"),
]

STALE = "Эта кнопка от старого сообщения — бот его уже не помнит (перезапускали?). Спросите заново."


class TelegramError(RuntimeError):
    pass


def read_token() -> str:
    """Токен бота. Пробелов и переносов в токене не бывает — если при копировании
    они попали в файл, убираются."""
    token = os.environ.get("GEORAG_TG_TOKEN", "")
    if not token.strip() and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8-sig")
    return "".join(token.split())


def access(path: Path = USERS_FILE) -> tuple[bool, set[int]]:
    """(открыт ли бот всем, id владельцев). id по одному в строке, «*» — всем, # — комментарий."""
    env = os.environ.get("GEORAG_TG_USERS", "")
    lines = env.replace(",", "\n").splitlines()
    if path.exists():
        lines += path.read_text(encoding="utf-8-sig").splitlines()
    owners, everyone = set(), False
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line.lower() in OPEN_ACCESS:
            everyone = True
        elif line.lstrip("-").isdigit():
            owners.add(int(line))
    return everyone, owners


def allowed_users(path: Path = USERS_FILE) -> set[int]:
    """Владельцы: id из файла (без «*»)."""
    return access(path)[1]


def _instance_lock(token: str) -> Any:
    """Одна копия бота на компьютере: занять локальный порт, выведенный из номера бота.
    Порт занят — бот уже запущен в другом окне. Процесс завершился — порт свободен сам."""
    import socket

    bot_id = int(token.split(":", 1)[0]) if token.split(":", 1)[0].isdigit() else 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: порт только наш
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", 40000 + bot_id % 20000))
    except OSError:
        sock.close()
        return None
    return sock


# --------------------------------------------------------------------------- #
#  Bot API
# --------------------------------------------------------------------------- #
class Telegram:
    def __init__(self, token: str, api: str = API, timeout: int = 30):
        self.base = f"{api.rstrip('/')}/bot{token}"
        self.timeout = timeout

    def call(
        self, method: str, files: Any = None, http_timeout: int | None = None, **params: Any
    ) -> Any:
        import json

        import requests

        url = f"{self.base}/{method}"
        params = {k: v for k, v in params.items() if v is not None}
        if files:
            data = {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in params.items()
            }
            response = requests.post(
                url, data=data, files=files, timeout=http_timeout or self.timeout
            )
        else:
            response = requests.post(url, json=params, timeout=http_timeout or self.timeout)
        try:
            data = response.json()
        except ValueError:
            raise TelegramError(f"{method}: ответ не JSON, HTTP {response.status_code}") from None
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description') or response.status_code}")
        return data.get("result")  # ответ Bot API: объект, список или True

    def send(self, chat_id: int, text: str, markup: dict[str, Any] | None = None) -> dict[str, Any]:
        sent: dict[str, Any] = self.call(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=markup,
        )
        return sent

    def edit(
        self, chat_id: int, message_id: int, text: str, markup: dict[str, Any] | None = None
    ) -> None:
        try:
            self.call(
                "editMessageText",
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
                reply_markup=markup,
            )
        except TelegramError as exc:
            if "not modified" not in str(exc):  # тот же текст — не ошибка
                raise

    def photo(
        self, chat_id: int, png: bytes, caption: str = "", markup: dict[str, Any] | None = None
    ) -> Any:
        return self.call(
            "sendPhoto",
            chat_id=chat_id,
            caption=caption,
            parse_mode="HTML",
            reply_markup=markup,
            files={"photo": ("graph.png", png, "image/png")},
        )

    def edit_photo(
        self,
        chat_id: int,
        message_id: int,
        png: bytes,
        caption: str = "",
        markup: dict[str, Any] | None = None,
    ) -> None:
        """Заменить картинку в том же сообщении — переход к соседу не плодит сообщения."""
        media = {
            "type": "photo",
            "media": "attach://graph",
            "caption": caption,
            "parse_mode": "HTML",
        }
        self.call(
            "editMessageMedia",
            chat_id=chat_id,
            message_id=message_id,
            media=media,
            reply_markup=markup,
            files={"graph": ("graph.png", png, "image/png")},
        )

    def document(self, chat_id: int, name: str, data: bytes, caption: str = "") -> None:
        self.call(
            "sendDocument", chat_id=chat_id, caption=caption, files={"document": (name, data)}
        )

    def answer_callback(self, callback_id: str, text: str = "", alert: bool = False) -> None:
        try:
            self.call(
                "answerCallbackQuery",
                callback_query_id=callback_id,
                text=text[:190] or None,
                show_alert=alert or None,
            )
        except TelegramError:
            pass  # нажатие устарело — не беда

    def typing(self, chat_id: int) -> None:
        try:
            self.call("sendChatAction", chat_id=chat_id, action="typing")
        except TelegramError:
            pass

    def updates(self, offset: int | None, timeout: int = POLL_TIMEOUT) -> list[dict[str, Any]]:
        params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        return self.call("getUpdates", http_timeout=timeout + 15, **params) or []


# --------------------------------------------------------------------------- #
#  Разговор
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    history: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)  # последнего ответа — для /f
    answers: dict[int, dict[str, Any]] = field(default_factory=dict)  # ответы — для кнопок под ними
    lists: dict[int, dict[str, Any]] = field(default_factory=dict)  # выдачи с листанием
    territories: list[dict[str, Any]] = field(default_factory=list)  # список датасетов, как показан
    awaiting: str | None = None  # ждём от человека: "search" — что искать
    busy: bool = False  # модель сейчас пишет ответ
    stop: threading.Event = field(default_factory=threading.Event)
    counter: int = 0

    def next_id(self) -> int:
        self.counter += 1
        return self.counter

    def remember(self, store: dict[int, dict[str, Any]], value: dict[str, Any]) -> int:
        key = self.next_id()
        store[key] = value
        while len(store) > KEEP:
            store.pop(next(iter(store)))
        return key


class Bot:
    def __init__(
        self,
        tg: Telegram,
        dsn: str | None,
        embedder: Any,
        chat_settings: Any,
        users_file: Path = USERS_FILE,
        log: Any = print,
    ) -> None:
        self.tg = tg
        self.dsn = dsn
        self.embedder = embedder
        self.chat_settings = chat_settings
        self.users_file = users_file
        self.log = log
        self.sessions: dict[int, Session] = {}
        self.offset: int | None = None
        self.background = False  # в run() — ответы модели в отдельном потоке
        self.workers: dict[int, threading.Thread] = {}
        self.slots = threading.BoundedSemaphore(MAX_PARALLEL)  # модель пишет по одному
        self.lock = threading.Lock()
        self.waiting = 0  # сколько ответов ждут очереди
        self.denied: dict[int, float] = {}  # чужой → когда последний раз отвечали
        self.seen: list[int] = []  # последние update_id: не брать дважды

    def session(self, chat_id: int) -> Session:
        session = self.sessions.pop(chat_id, None) or Session()
        self.sessions[chat_id] = session  # в конец: недавний разговор
        while len(self.sessions) > MAX_SESSIONS:  # старые забываются, кроме тех, где пишут
            old = next((k for k, s in self.sessions.items() if not s.busy), None)
            if old is None or old == chat_id:
                break
            self.sessions.pop(old)
        return session

    # -- доступ и лимиты ---------------------------------------------------- #
    def allowed(self, user_id: Any) -> bool:
        everyone, owners = access(self.users_file)
        return everyone or user_id in owners

    def guest(self, user_id: Any) -> bool:
        """Не из списка — пишет в открытый бот (для журнала)."""
        return user_id not in access(self.users_file)[1]

    def deny(self, chat_id: int, user_id: Any) -> None:
        """Чужому — его id, но не на каждое сообщение: не чаще раза в DENIED_EVERY."""
        now = time.monotonic()
        if now - self.denied.get(user_id, -DENIED_EVERY) < DENIED_EVERY:
            return
        self.denied[user_id] = now
        self.tg.send(
            chat_id,
            f"Нет доступа. Ваш id: <code>{user_id}</code>\n"
            "Владелец базы добавляет его в telegram_users.txt.",
        )

    # -- разбор входящих ---------------------------------------------------- #
    def handle(self, update: dict[str, Any]) -> None:
        if update.get("callback_query"):
            self.on_callback(update["callback_query"])
        else:
            self.on_message(update.get("message") or {})

    def _guard(self, chat_id: int, what: str, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 — одно сообщение не должно ронять бота
            self.log(f"  ошибка на {what[:40]!r}: {type(exc).__name__}: {exc}")
            try:
                self.tg.send(chat_id, f"Ошибка: {esc(type(exc).__name__)}: {esc(str(exc)[:300])}")
            except Exception:  # noqa: BLE001
                pass

    def on_message(self, message: dict[str, Any]) -> None:
        chat_id = (message.get("chat") or {}).get("id")
        user_id = (message.get("from") or {}).get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None or not text:
            return
        if not self.allowed(user_id):
            self.deny(chat_id, user_id)
            self.log(f"  чужой пользователь {user_id}: {text[:40]!r}")
            return
        self.log(f"  ← {user_id}{' (гость)' if self.guest(user_id) else ''}: {text[:60]!r}")
        self._guard(chat_id, text, self.route, chat_id, self.session(chat_id), text)

    def route(self, chat_id: int, session: Session, text: str) -> None:
        if text in ui.MENU:  # кнопка меню внизу экрана
            command, arg = "/" + ui.MENU[text], ""
        elif text == "Настройки":  # кнопка из старого меню (до v12)
            command, arg = "/settings", ""
        else:
            command, _, arg = text.partition(" ")
            command = command.split("@", 1)[0].lower()
            arg = arg.strip()
            if not command.startswith("/"):
                command, arg = "", text
        if command:
            session.awaiting = None
        if command in ("/start", "/help"):
            self.tg.send(chat_id, HELP, ui.menu_keyboard())
        elif command == "/new":
            self.new_conversation(chat_id, session)
        elif command in ("/brief", "/detail", "/settings"):
            self.tg.send(
                chat_id,
                "Настроек больше нет: нужно подробно или кратко — так и "
                "напишите в вопросе, например «подробно: как выделяют "
                "рудные узлы?». Что в базе и готова ли модель — /stats",
                ui.menu_keyboard(),
            )
        elif command == "/f":
            self.fragment(chat_id, session, arg)
        elif command == "/search":
            if arg:
                self.search(chat_id, session, arg)
            else:
                session.awaiting = "search"
                self.tg.send(
                    chat_id,
                    "Что искать? Напишите запрос — например: "
                    "<i>плотность линеаментов и рудные узлы</i>",
                    ui.inline([[ui.button("Отмена", "cancel")]]),
                )
        elif command == "/articles":
            self.articles(chat_id, session)
        elif command in ("/facts", "/fact", "/dataset", "/datasets"):  # /dataset — старое имя
            if arg:
                self.dataset_by_name(chat_id, session, arg)
            else:
                self.datasets(chat_id, session)
        elif command == "/stats":
            self.stats(chat_id)
        elif command:
            self.tg.send(
                chat_id, "Такой команды нет. Что умеет бот — «Помощь» в меню.", ui.menu_keyboard()
            )
        elif session.awaiting == "search":
            session.awaiting = None
            self.search(chat_id, session, arg)
        else:
            self.question(chat_id, session, arg)

    def new_conversation(self, chat_id: int, session: Session) -> None:
        session.history.clear()
        session.sources = []
        session.awaiting = None
        self.tg.send(chat_id, "Начали новый разговор. Задайте вопрос.", ui.menu_keyboard())

    # -- нажатия на кнопки под сообщениями ----------------------------------- #
    def on_callback(self, cq: dict[str, Any]) -> None:
        cid = cq.get("id", "")
        user_id = (cq.get("from") or {}).get("id")
        message = cq.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        mid = message.get("message_id")
        data = cq.get("data") or ""
        if not self.allowed(user_id):
            self.tg.answer_callback(cid, "Нет доступа", alert=True)
            return
        if chat_id is None or mid is None:
            self.tg.answer_callback(cid)
            return
        session = self.session(chat_id)
        toast = ""
        try:
            toast = self.press(chat_id, mid, session, data) or ""
        except Exception as exc:  # noqa: BLE001
            self.log(f"  ошибка на кнопке {data!r}: {type(exc).__name__}: {exc}")
            toast = f"Ошибка: {type(exc).__name__}: {str(exc)[:120]}"
        self.tg.answer_callback(cid, toast, alert=toast == STALE or toast.startswith("Ошибка"))

    def press(self, chat_id: int, mid: int, session: Session, data: str) -> str | None:
        """Нажатие → действие. Возвращает короткую подсказку для всплывашки."""
        action, *args = data.split(":")
        nums = [int(a) for a in args if a.lstrip("-").isdigit()]
        if action == "noop":
            return None
        if action == "cancel":
            session.awaiting = None
            self.tg.edit(chat_id, mid, "Отменено.")
            return None
        if action == "new":
            self.new_conversation(chat_id, session)
            return "Новый разговор"
        if action == "stop":
            if session.busy:
                session.stop.set()
                return "Останавливаю…"
            return "Ответ уже готов"
        if action in ("f", "how"):
            record = session.answers.get(nums[0]) if nums else None
            if record is None:
                return STALE
            if action == "f":
                src = next((s for s in record["sources"] if s["n"] == nums[1]), None)
                if src is None:
                    return STALE
                self.show_fragment(chat_id, session, src)
            else:
                self.tg.send(chat_id, ui.how_text(record))
            return None
        if action in ("pg", "it"):
            lst = session.lists.get(nums[0]) if nums else None
            if lst is None or len(nums) < 2:
                return STALE
            if action == "pg":
                page = max(0, min(nums[1], ui.pages(len(lst["items"]), lst["kind"]) - 1))
                lst["page"] = page
                self.tg.edit(
                    chat_id,
                    mid,
                    ui.list_text(lst, page),
                    ui.list_keyboard(nums[0], lst, page, lst.get("extra")),
                )
            else:
                self.open_item(chat_id, session, nums[0], lst, nums[1])
            return None
        if action == "art":  # «Вся статья» у фрагмента
            doc_id = session.lists.get(nums[0], {}).get("doc_id") if nums else None
            if not doc_id:
                return STALE
            self.article_card(chat_id, session, doc_id)
            return None
        if action == "chunks":
            lst = session.lists.get(nums[0]) if nums else None
            if lst is None:
                return STALE
            self.article_chunks(
                chat_id, session, lst["doc_id"], lst.get("title") or "", lst.get("url") or ""
            )
            return None
        if action == "askq":
            lst = session.lists.get(nums[0]) if nums else None
            if lst is None or not lst.get("query"):
                return STALE
            self.question(chat_id, session, lst["query"])
            return None
        if action == "dsl":
            self.datasets(chat_id, session, mid)
            return None
        if action in ("ds", "dn", "dt", "csv", "ask"):
            if not nums or not 0 <= nums[0] < len(session.territories):
                return STALE
            name = session.territories[nums[0]]["name"]
            if action == "ds":  # из списка — картинка новым сообщением
                self.graph_card(chat_id, session, nums[0])
            elif action == "dn":  # сосед на картинке — в том же сообщении
                self.graph_card(chat_id, session, nums[0], mid)
            elif action == "dt":  # факты списком, с числом статей
                self.dataset_card(chat_id, session, nums[0])
            elif action == "csv":
                self.dataset_csv(chat_id, name)
                return "Таблица отправлена"
            else:
                self.question(chat_id, session, ui.dataset_question(name))
            return None
        return STALE

    # -- чат-бот ------------------------------------------------------------ #
    def question(
        self, chat_id: int, session: Session, text: str, history: list[dict[str, Any]] | None = None
    ) -> None:
        if session.busy:
            self.tg.send(
                chat_id, "Ещё пишу прошлый ответ — дождитесь или нажмите «Остановить» " "под ним."
            )
            return
        session.busy = True
        session.stop.clear()
        args = (chat_id, session, text, history)
        if self.background:
            worker = threading.Thread(
                target=self._answer_safe, args=args, name=f"tg-answer-{chat_id}", daemon=True
            )
            self.workers[chat_id] = worker
            worker.start()
        else:
            self._answer_safe(*args)

    def _answer_safe(self, chat_id: Any, session: Any, text: Any, history: Any) -> None:
        try:
            self._wait_turn(chat_id)
            try:
                self._guard(chat_id, text, self.answer, chat_id, session, text, history)
            finally:
                self.slots.release()
        finally:
            session.busy = False

    def _wait_turn(self, chat_id: int) -> None:
        """Модель одна: пока она отвечает другим, вопрос ждёт — человеку видно место в очереди."""
        if self.slots.acquire(blocking=False):
            return
        with self.lock:
            self.waiting += 1
            place = self.waiting
        try:
            note = None
            try:
                note = self.tg.send(
                    chat_id,
                    f"<i>Модель сейчас отвечает другим — вы в очереди, "
                    f"перед вами {place}. Ответ начнётся сам.</i>",
                )
            except TelegramError:
                pass
            self.slots.acquire()
            if note:
                try:
                    self.tg.edit(
                        chat_id, note["message_id"], "<i>Очередь подошла — отвечаю ниже.</i>"
                    )
                except TelegramError:
                    pass
        finally:
            with self.lock:
                self.waiting -= 1

    def answer(
        self, chat_id: int, session: Session, text: str, history: list[dict[str, Any]] | None = None
    ) -> None:
        from ..chat import answer as chat
        from ..index import db

        history = list(session.history if history is None else history)
        aid = session.next_id()
        self.tg.typing(chat_id)
        msg = self.tg.send(chat_id, "<i>разбираю вопрос…</i>", ui.stop_keyboard(aid))
        mid = msg["message_id"]
        settings = self.chat_settings
        sources, answer, general, done, error, info = [], "", None, {}, None, {}
        stopped = False
        last_edit = time.monotonic()
        with db.connect(self.dsn) as conn:
            db.autocommit(conn)  # ответ пишется минутами — без открытой транзакции
            events = chat.answer(conn, self.embedder, text, history, settings)
            try:
                for ev in events:
                    if session.stop.is_set():
                        stopped = True
                        break
                    kind = ev["type"]
                    if kind == "status" and not answer:
                        self.tg.edit(
                            chat_id, mid, f"<i>{esc(ev['text'])}</i>", ui.stop_keyboard(aid)
                        )
                    elif kind == "sources":
                        sources, info = ev["sources"], ev
                    elif kind == "general":
                        general, info = ev["text"], ev
                    elif kind == "token":
                        answer += ev["text"]
                        if time.monotonic() - last_edit >= EDIT_EVERY:
                            # Пока пишется — простым текстом: обрезанный кусок с тегами
                            # Телеграм не примет.
                            try:
                                self.tg.edit(
                                    chat_id, mid, esc(answer[-LIMIT:]) + " …", ui.stop_keyboard(aid)
                                )
                            except TelegramError:
                                pass
                            last_edit = time.monotonic()
                    elif kind == "revised":
                        answer = ev["text"]  # без фраз без ссылки на фрагмент
                    elif kind == "done":
                        done = ev
                    elif kind == "error":
                        error = ev["error"]
            finally:
                events.close()  # остановили — Ollama перестаёт писать
        if error:
            self.tg.edit(chat_id, mid, f"Ошибка: {esc(error)}")
            return
        if stopped:
            done = {**done, "stopped": True}
            if not answer.strip():
                self.tg.edit(chat_id, mid, "<i>Остановлено.</i>")
                return

        session.sources = sources
        session.history = (
            history
            + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer, "mode": done.get("mode")},
            ]
        )[-HISTORY_TURNS * 2 :]
        record = {
            "question": text,
            "sources": sources,
            "history": history,
            "info": info,
            "used": done.get("used"),
            "mode": done.get("mode"),
            "model": done.get("model"),
            "seconds": done.get("seconds"),
        }
        session.answers[aid] = record
        while len(session.answers) > KEEP:
            session.answers.pop(next(iter(session.answers)))

        markup = ui.answer_keyboard(aid, sources)
        pieces = split(ui.answer_text(answer, general, sources, done))
        try:
            self._deliver(chat_id, mid, pieces, markup)
        except TelegramError as exc:
            if "parse" not in str(exc).lower():
                raise
            # Разметка не разобралась — тот же ответ без неё, чтобы он всё равно дошёл.
            self._deliver(chat_id, mid, [ui.plain(p) for p in pieces], markup)

    def _deliver(self, chat_id: int, mid: int, pieces: list[str], markup: Any) -> None:
        """Первый кусок — правкой сообщения «работаю», остальные — новыми; кнопки — у последнего."""
        last = len(pieces) - 1
        self.tg.edit(chat_id, mid, pieces[0], markup if last == 0 else None)
        for i, piece in enumerate(pieces[1:], start=1):
            self.tg.send(chat_id, piece, markup if i == last else None)

    def fragment(self, chat_id: int, session: Session, arg: str) -> None:
        if not session.sources:
            self.tg.send(
                chat_id, "Сначала задайте вопрос — фрагменты берутся из последнего ответа."
            )
            return
        number = int(arg) if arg.isdigit() else 0
        src = next((s for s in session.sources if s["n"] == number), None)
        if src is None:
            self.tg.send(
                chat_id,
                f"Фрагменты последнего ответа: 1–{len(session.sources)}. "
                "Или нажмите номер под ответом.",
            )
            return
        self.show_fragment(chat_id, session, src)

    def show_fragment(
        self, chat_id: int, session: Session, src: dict[str, Any], with_article: bool = True
    ) -> None:
        article = None
        doc_id = src.get("doc_id") or ""
        if with_article and doc_id and src.get("kind") != "датасет":
            ref = session.remember(session.lists, {"kind": "ref", "doc_id": doc_id, "items": []})
            article = f"art:{ref}"
        pieces = split(ui.fragment_text(src))
        for i, piece in enumerate(pieces):
            self.tg.send(
                chat_id, piece, ui.fragment_keyboard(src, article) if i == len(pieces) - 1 else None
            )

    # -- поиск и статьи ----------------------------------------------------- #
    def _list(self, chat_id: int, session: Session, lst: dict[str, Any], extra: Any = None) -> None:
        """Список с листанием. extra(номер списка) → кнопки под номерами."""
        lst["page"] = 0
        pid = session.remember(session.lists, lst)
        if extra:
            lst["extra"] = extra(pid)
        self.tg.send(chat_id, ui.list_text(lst, 0), ui.list_keyboard(pid, lst, 0, lst.get("extra")))

    def search(self, chat_id: int, session: Session, query: str) -> None:
        from ..chat.answer import source_dict
        from ..index import db
        from ..index.search import hybrid_search

        query = query.strip()
        if not query:
            self.tg.send(chat_id, "Что искать? Например: /search плотность линеаментов")
            return
        from ..graph.synonyms import cached, synonym_variants

        self.tg.typing(chat_id)
        also = synonym_variants(query, cached())  # «Донбасс» — и как «Donetsk basin»
        with db.connect(self.dsn) as conn:
            hits = hybrid_search(
                conn,
                self.embedder,
                query,
                limit=SEARCH_HITS,
                max_per_doc=2,
                min_similarity=0.45,
                alternatives=also,
            )
        if not hits:
            self.tg.send(
                chat_id,
                f"По запросу «{esc(query)}» близкого в базе не нашлось.\n"
                "Попробуйте другими словами или по-английски.",
            )
            return
        items = [source_dict(i, h) for i, h in enumerate(hits, start=1)]
        self._list(
            chat_id,
            session,
            {
                "kind": "search",
                "items": items,
                "query": query,
                "title": f"<b>Поиск:</b> {esc(query)} — найдено {len(items)}",
            },
            extra=lambda pid: [ui.button("Спросить бота об этом", f"askq:{pid}")],
        )

    def articles(self, chat_id: int, session: Session) -> None:
        from ..index import db

        with db.connect(self.dsn) as conn:
            docs = db.documents(conn, limit=1000)
        if not docs:
            self.tg.send(chat_id, "В базе пока нет статей.")
            return
        self._list(
            chat_id,
            session,
            {
                "kind": "articles",
                "items": docs,
                "title": f"<b>Статей в базе: {len(docs)}</b> " "(новые сверху)",
            },
        )

    def open_item(
        self, chat_id: int, session: Session, pid: int, lst: dict[str, Any], idx: int
    ) -> None:
        if not 0 <= idx < len(lst["items"]):
            return
        item = lst["items"][idx]
        if lst["kind"] == "search":
            self.show_fragment(chat_id, session, item)
        elif lst["kind"] == "articles":
            self.article_card(chat_id, session, item["doc_id"], item)
        else:
            src = {**item, "title": lst.get("title_plain") or "", "url": lst.get("url") or ""}
            self.show_fragment(chat_id, session, src, with_article=False)

    def article_card(
        self, chat_id: int, session: Session, doc_id: str, doc: dict[str, Any] | None = None
    ) -> None:
        from ..index import db

        if doc is None:
            with db.connect(self.dsn) as conn:
                doc = next(
                    (d for d in db.documents(conn, limit=100000) if d["doc_id"] == doc_id), None
                )
            if doc is None:
                self.tg.send(chat_id, "Этой статьи в базе уже нет.")
                return
        ref = session.remember(
            session.lists,
            {
                "kind": "ref",
                "doc_id": doc_id,
                "items": [],
                "title": doc.get("title") or doc_id,
                "url": doc.get("url") or "",
            },
        )
        row = [ui.button("Фрагменты статьи", f"chunks:{ref}")]
        if ui.is_link(doc.get("url")):
            row.insert(0, ui.url_button("У издателя", doc["url"]))
        self.tg.send(chat_id, ui.article_text(doc), ui.inline([row]))

    def article_chunks(
        self, chat_id: int, session: Session, doc_id: str, title: str, url: str = ""
    ) -> None:
        from ..index import db

        with db.connect(self.dsn) as conn:
            chunks = db.document_chunks(conn, doc_id)
        if not chunks:
            self.tg.send(chat_id, "У статьи нет фрагментов в базе.")
            return
        items = [{**c, "doc_id": doc_id} for c in chunks]
        self._list(
            chat_id,
            session,
            {
                "kind": "chunks",
                "items": items,
                "doc_id": doc_id,
                "title": f"<b>{esc(title[:120])}</b> — фрагментов {len(items)}",
                "title_plain": title,
                "url": url,
            },
        )

    # -- датасеты ----------------------------------------------------------- #
    def datasets(self, chat_id: int, session: Session, mid: int | None = None) -> None:
        from ..graph import api
        from ..graph.synonyms import load
        from ..index import db

        with db.connect(self.dsn) as conn:
            data = api.datasets_response(conn, load(), limit=ui.DATASET_BUTTONS)
        rows = data["datasets"]
        session.territories = rows
        text, markup = ui.datasets_text(rows, data.get("total")), ui.datasets_keyboard(rows)
        if mid:
            try:
                self.tg.edit(chat_id, mid, text, markup)
                return
            except TelegramError:
                pass  # под картинкой текст не правится — новым
        self.tg.send(chat_id, text, markup)

    def _dataset(self, name: str) -> dict[str, Any]:
        from ..graph import api
        from ..graph.synonyms import load
        from ..index import db

        with db.connect(self.dsn) as conn:
            return api.dataset_response(conn, load(), {"name": name})

    def dataset_card(
        self, chat_id: int, session: Session, idx: int, mid: int | None = None
    ) -> None:
        data = self._dataset(session.territories[idx]["name"])
        pieces = split(ui.dataset_text(data))
        markup = ui.dataset_keyboard(idx)
        if mid and len(pieces) == 1:
            self.tg.edit(chat_id, mid, pieces[0], markup)
            return
        for i, piece in enumerate(pieces):
            self.tg.send(chat_id, piece, markup if i == len(pieces) - 1 else None)

    def dataset_by_name(self, chat_id: int, session: Session, name: str) -> None:
        from ..graph.api import RequestError

        try:
            data = self._dataset(name)
        except RequestError as exc:
            near = ", ".join(exc.known[:8])
            self.tg.send(
                chat_id,
                f"«{esc(name)}»: {esc(exc.message)}."
                + (f" Больше всего фактов у: {esc(near)}" if near else ""),
            )
            return
        self.graph_card(chat_id, session, self._territory(session, data), data=data)

    def _territory(self, session: Session, data: dict[str, Any]) -> int:
        """Номер сущности в списке разговора — для кнопок (в callback_data только номер)."""
        idx = next(
            (i for i, r in enumerate(session.territories) if r["name"] == data["name"]), None
        )
        if idx is None:
            if len(session.territories) >= MAX_TERRITORIES:
                session.territories = []  # старые кнопки ответят «от старого сообщения»
            session.territories.append(
                {"name": data["name"], "documents": data["documents"], "facts": len(data["facts"])}
            )
            idx = len(session.territories) - 1
        return idx

    def graph_card(
        self,
        chat_id: int,
        session: Session,
        idx: int,
        mid: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Картинка связей сущности, под ней — соседи кнопками, цитаты, таблица, вопрос."""
        from . import picture

        data = data or self._dataset(session.territories[idx]["name"])
        try:
            png, neighbors = picture.render(data)
        except Exception as exc:  # noqa: BLE001 — не нарисовалось: факты списком, как раньше
            self.log(f"  картинка связей не нарисовалась: {type(exc).__name__}: {exc}")
            self.dataset_card(chat_id, session, idx)
            return
        idx = self._territory(session, data)  # сначала сама сущность: список мог обнулиться
        buttons = []
        for name in neighbors[: ui.NEIGHBOR_BUTTONS]:
            n = next((i for i, r in enumerate(session.territories) if r["name"] == name), None)
            if n is None:
                session.territories.append({"name": name, "documents": 0, "facts": 0})
                n = len(session.territories) - 1
            buttons.append((name, n))
        caption, markup = ui.graph_caption(data, len(neighbors)), ui.graph_keyboard(idx, buttons)
        if mid:
            try:
                self.tg.edit_photo(chat_id, mid, png, caption, markup)
                return
            except TelegramError:
                pass  # не картинка или устарело — новым сообщением
        self.tg.photo(chat_id, png, caption, markup)

    def dataset_csv(self, chat_id: int, name: str) -> None:
        from ..graph import api

        data = self._dataset(name)
        self.tg.document(
            chat_id,
            f"факты-{data['name'][:60]}.csv",
            api.dataset_csv(data).encode("utf-8"),
            caption=f"Факты о «{data['name']}» с цитатами — открывается в Excel",
        )

    # -- состояние ---------------------------------------------------------- #
    def _stats(self) -> dict[str, Any] | None:
        from ..index import db

        try:
            with db.connect(self.dsn) as conn:
                return db.stats(conn)
        except Exception:  # noqa: BLE001 — база лежит: так и покажем
            return None

    def stats(self, chat_id: int) -> None:
        from ..chat import answer as chat

        status = chat.ollama_status(self.chat_settings.host, self.chat_settings.model, timeout=3)
        self.tg.send(chat_id, ui.status_text(status, self._stats()))

    # -- цикл --------------------------------------------------------------- #
    def poll_once(self, timeout: int = POLL_TIMEOUT) -> int:
        updates = self.tg.updates(self.offset, timeout)
        for update in updates:
            uid = update["update_id"]
            self.offset = uid + 1
            if uid in self.seen:  # то же сообщение второй раз — не отвечать дважды
                continue
            self.seen = (self.seen + [uid])[-SEEN_UPDATES:]
            self.handle(update)
        return len(updates)

    def run(self, stop: threading.Event | None = None) -> None:
        token = self.tg.base.rsplit("/bot", 1)[-1]
        self._lock = _instance_lock(token)
        if self._lock is None:
            self.log(
                "Телеграм-бот НЕ запущен: он уже работает в другом окне на этом компьютере. "
                "Две копии забирают одни и те же сообщения и отвечают по несколько раз. "
                "Оставьте одну (закройте лишнее окно Ctrl+C)."
            )
            return
        self.background = True
        try:
            self.tg.call(
                "setMyCommands", commands=[{"command": c, "description": d} for c, d in COMMANDS]
            )
        except Exception as exc:  # noqa: BLE001 — без меню команд бот всё равно работает
            self.log(f"Меню команд не задалось: {exc}")
        try:  # с webhook getUpdates не работает — сообщения не приходят
            self.tg.call("deleteWebhook")
        except Exception:  # noqa: BLE001
            pass
        last_error = ""
        while not (stop and stop.is_set()):
            try:
                self.poll_once()
                last_error = ""
            except Exception as exc:  # noqa: BLE001 — сеть моргнула: подождать и дальше
                text = explain_error(exc)
                if text != last_error:  # одно и то же не повторять каждые 5 с
                    self.log(text)
                    last_error = text
                time.sleep(5)


def explain_error(exc: Exception) -> str:
    """Ошибка Телеграма → что с ней делать, по-человечески."""
    text = str(exc)
    if "409" in text or "Conflict" in text or "terminated by other getUpdates" in text:
        return (
            "Телеграм: этот бот уже запущен в ДРУГОМ окне (или на другом компьютере) — "
            "сообщения забирает та копия. Закройте лишние окна (Ctrl+C) и оставьте одно."
        )
    if "401" in text or "Unauthorized" in text or "Not Found" in text:
        return (
            "Телеграм не принимает токен. Возьмите токен заново у @BotFather "
            "(/mybots → бот → API Token) и сохраните в telegram_token.txt."
        )
    return f"Телеграм не ответил ({type(exc).__name__}: {text[:200]}) — повтор через 5 с"


def check(
    log: Any = print, token: str | None = None, api: str = API, users_file: Path = USERS_FILE
) -> int:
    """python georag.py tg --check: всё ли в порядке с ботом, без базы и моделей."""
    token = read_token() if token is None else token
    log("Проверка Телеграм-бота\n")
    if not token:
        log(
            f"[СБОЙ] Токена нет: нет файла {TOKEN_FILE.name} в папке проекта (или он пустой).\n"
            "       Сохранить: python -c \"open('telegram_token.txt','w',encoding='utf-8')"
            ".write('ТОКЕН')\""
        )
        return 1
    if not re.fullmatch(r"\d{6,}:[\w-]{30,}", token):
        log(
            f"[СБОЙ] Токен не похож на токен (вида 123456789:AA…): «{token[:12]}…», "
            f"длина {len(token)}.\n       Скопируйте его у @BotFather целиком и сохраните заново."
        )
        return 1
    log(f"[OK  ] Токен есть: {token.split(':')[0]}:…")
    tg = Telegram(token, api=api, timeout=20)
    try:
        me = tg.call("getMe")
    except Exception as exc:  # noqa: BLE001
        log("[СБОЙ] " + explain_error(exc))
        return 1
    log(f"[OK  ] Телеграм принял токен. Ваш бот: @{me.get('username')} — пишите именно ему.")
    try:
        hook = tg.call("getWebhookInfo") or {}
        if hook.get("url"):
            tg.call("deleteWebhook")
            log("[OK  ] У бота был webhook — убран (из-за него сообщения не доходили).")
    except Exception:  # noqa: BLE001
        pass
    everyone, users = access(users_file)
    if everyone:
        log("[OK  ] Бот открыт всем («*» в telegram_users.txt), без ограничений.")
    elif users:
        log(f"[OK  ] Кому бот отвечает (telegram_users.txt): {', '.join(map(str, sorted(users)))}")
    else:
        log(
            "[ !! ] В telegram_users.txt никого нет — бот на всё отвечает «Нет доступа. Ваш id».\n"
            "       Открыть всем: строка * в telegram_users.txt."
        )
    try:
        waiting = tg.call("getUpdates", timeout=0, http_timeout=20) or []
    except Exception as exc:  # noqa: BLE001
        log("[СБОЙ] " + explain_error(exc))
        return 1
    senders = {}
    for u in waiting:
        who = (u.get("message") or u.get("callback_query") or {}).get("from") or {}
        if who.get("id"):
            senders[who["id"]] = who.get("first_name") or who.get("username") or ""
    if senders:
        log(f"[ !! ] Боту написали, но бот сейчас НЕ запущен — сообщений ждёт: {len(waiting)}.")
        for uid, name in senders.items():
            mark = (
                "есть доступ"
                if uid in users
                else "гость — бот открыт всем" if everyone else "НЕТ в telegram_users.txt"
            )
            log(f"       от {name} — id {uid} ({mark})")
        if not everyone and any(uid not in users for uid in senders):
            log(
                "       Добавить id: python -c \"open('telegram_users.txt','a',"
                "encoding='utf-8').write('ID\\n')\""
            )
    else:
        log(
            "[ !! ] Непрочитанных сообщений нет. Если вы писали боту — либо он уже запущен и "
            "всё забрал,\n       либо писали не тому боту (нужен @"
            + str(me.get("username"))
            + "), либо не нажали «Начать» (/start)."
        )
    log(
        "\nЗапуск бота: python georag.py web --tg (или только бот: python georag.py tg).\n"
        "Окно не закрывать — пока оно открыто, бот отвечает."
    )
    return 0


def start_in_thread(
    dsn: Any, embedder: Any, chat_settings: Any, log: Any = print
) -> threading.Thread | None:
    """Бот в фоне того же процесса — для `web --tg`: модель векторов одна на двоих."""
    token = read_token()
    if not token:
        log("Телеграм-бот не запущен: нет токена (telegram_token.txt). Как завести — ЗАПУСК.md.")
        return None
    bot = Bot(Telegram(token), dsn, embedder, chat_settings, log=log)
    try:
        me = bot.tg.call("getMe")
    except Exception as exc:  # noqa: BLE001
        log(
            "Телеграм-бот не запущен. "
            + explain_error(exc)
            + "\nПроверить всё по шагам: python georag.py tg --check"
        )
        return None
    everyone, owners = access()
    who = "всем" if everyone else f"тем, кто в telegram_users.txt ({len(owners)} чел.)"
    log(
        f"Телеграм-бот: @{me.get('username')} — отвечает {who}; входящие сообщения видны в этом окне"
    )
    thread = threading.Thread(target=bot.run, name="telegram", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..chat import answer as chat
    from ..common import add_db_args, add_embedder_args, add_llm_args
    from ..index import db
    from ..index.embed import build_embedder

    parser = argparse.ArgumentParser(description="Телеграм-бот базы знаний")
    add_db_args(parser)
    add_embedder_args(parser)
    add_llm_args(parser)
    parser.add_argument("--check", action="store_true", help="проверить токен и доступ и выйти")
    args = parser.parse_args(argv)
    if args.check:
        return check()

    token = read_token()
    if not token:
        print(
            "Нет токена бота. Получить у @BotFather в Телеграме и сохранить:\n"
            "    python -c \"open('telegram_token.txt','w').write('ТОКЕН')\"",
            file=sys.stderr,
        )
        return 1
    tg = Telegram(token)
    try:
        me = tg.call("getMe")
    except Exception as exc:  # noqa: BLE001
        print(explain_error(exc) + "\nПодробнее: python georag.py tg --check", file=sys.stderr)
        return 1
    try:
        with db.connect(args.dsn) as conn:
            info = db.stats(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна: {exc}\nПоднимите её: python georag.py start", file=sys.stderr)
        return 1
    everyone, users = access()
    print(f"В базе: {info['documents']} статей. Бот: @{me.get('username')}")
    if everyone:
        print("Бот открыт всем, без ограничений.")
    elif not users:
        print(
            "В telegram_users.txt никого нет — бот никому не ответит по существу. Напишите\n"
            "ему что-нибудь: он пришлёт ваш id, его и записать в telegram_users.txt."
        )
    print("Загружаю модель эмбеддингов…")
    embedder = build_embedder(args.embedder, device=args.device)
    embedder.encode_one("прогрев")
    settings = chat.Settings(model=args.model, host=args.ollama_host)
    status = chat.ollama_status(settings.host, settings.model)
    print(f"Чат-бот: {settings.model} — " + ("готов" if status["ok"] else status["error"]))
    print(
        f"Бот работает: пишите @{me.get('username')} в Телеграме (сначала /start). "
        "Входящие сообщения видны здесь. Остановить — Ctrl+C\n"
    )
    try:
        Bot(tg, args.dsn, embedder, settings).run()
    except KeyboardInterrupt:
        print("Остановлен.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
