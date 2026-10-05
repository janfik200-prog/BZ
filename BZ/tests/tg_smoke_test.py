"""Проверка Телеграм-бота: поддельный Bot API в этом же процессе, база и модели подменены.

Запуск:  python tests/tg_smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.chat import answer as A  # noqa: E402
from georag.graph import api as graph_api  # noqa: E402
from georag.index import db  # noqa: E402
from georag.index import search as S  # noqa: E402
from georag.index.search import Hit  # noqa: E402
from georag.tg import bot as B  # noqa: E402
from georag.tg import ui as U  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


class FakeAPI(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict | str]] = []
    queue: list[dict] = []
    reject_html = False
    counter = 0
    conflict = False
    webhook = ""

    def log_message(self, *a):
        pass

    def do_POST(self):
        method = self.path.rsplit("/", 1)[-1]
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if "multipart" in (self.headers.get("Content-Type") or ""):
            payload = body.decode("utf-8", "replace")
        else:
            payload = json.loads(body or b"{}")
        FakeAPI.calls.append((method, payload))
        result: object = True
        ok, desc = True, ""
        if method == "sendMessage":
            FakeAPI.counter += 1
            result = {"message_id": FakeAPI.counter}
        if (
            method in ("sendMessage", "editMessageText")
            and FakeAPI.reject_html
            and "<b>" in payload.get("text", "")
        ):
            ok, desc = False, "Bad Request: can't parse entities"
        if method == "getUpdates":
            result, FakeAPI.queue = FakeAPI.queue, []
            if FakeAPI.conflict:
                ok, desc = False, "Conflict: terminated by other getUpdates request"
        if method == "getWebhookInfo":
            result = {"url": FakeAPI.webhook}
        if method == "deleteWebhook":
            FakeAPI.webhook = ""
        if method == "getMe":
            result = {"username": "georag_test_bot"}
        data = json.dumps({"ok": ok, "result": result, "description": desc}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def sent(methods=("sendMessage", "editMessageText")) -> list[dict]:
    return [p for m, p in FakeAPI.calls if m in methods and isinstance(p, dict)]


def texts(method="sendMessage") -> list[str]:
    return [p.get("text", "") for p in sent((method,))]


def last() -> dict:
    msgs = sent()
    return msgs[-1] if msgs else {}


def last_text() -> str:
    return last().get("text", "")


def buttons(payload: dict) -> list[dict]:
    markup = payload.get("reply_markup") or {}
    return [b for row in markup.get("inline_keyboard", []) for b in row]


def labels(payload: dict) -> list[str]:
    return [b["text"] for b in buttons(payload)]


def data_of(payload: dict, label: str) -> str:
    return next(b["callback_data"] for b in buttons(payload) if b["text"] == label)


def multipart_field(payload: str, name: str) -> str:
    """Поле из multipart-тела (sendPhoto, editMessageMedia)."""
    head = f'name="{name}"'
    if head not in payload:
        return ""
    return payload.split(head, 1)[1].split("\r\n\r\n", 1)[1].split("\r\n--", 1)[0]


def photos(method="sendPhoto") -> list[str]:
    return [p for m, p in FakeAPI.calls if m == method and isinstance(p, str)]


def photo_labels(payload: str) -> list[str]:
    markup = json.loads(multipart_field(payload, "reply_markup") or "{}")
    return [b["text"] for row in markup.get("inline_keyboard", []) for b in row]


def photo_data(payload: str, label: str) -> str:
    markup = json.loads(multipart_field(payload, "reply_markup") or "{}")
    return next(
        b["callback_data"] for row in markup["inline_keyboard"] for b in row if b["text"] == label
    )


def toasts() -> list[str]:
    return [p.get("text", "") for m, p in FakeAPI.calls if m == "answerCallbackQuery"]


@contextmanager
def fake_connect(dsn=None):
    yield object()


def msg(text, user=42, chat=42, update_id=1):
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat}, "from": {"id": user}, "text": text},
    }


def press(data, user=42, chat=42, message_id=1):
    return {
        "update_id": 1,
        "callback_query": {
            "id": "cb1",
            "from": {"id": user},
            "data": data,
            "message": {"message_id": message_id, "chat": {"id": chat}},
        },
    }


class _TG:
    """Телеграм без сети: запоминает, что бот отправил."""

    def __init__(self, token="123456:TEST"):
        self.base = f"https://api/bot{token}"
        self.sent, self.toasts, self.queue = [], [], []

    def send(self, chat_id, text, markup=None):
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent)}

    def edit(self, chat_id, mid, text, markup=None):
        self.sent.append((chat_id, "(правка) " + text))

    def answer_callback(self, cid, text="", alert=False):
        self.toasts.append(text)

    def typing(self, chat_id):
        pass

    def updates(self, offset, timeout=0):
        out, self.queue = self.queue, []
        return out


def test_open_access() -> None:
    print("\nОткрытый бот: всем можно, у гостей лимиты, модель по очереди, одна копия")
    tmp = Path(tempfile.mkdtemp())
    users = tmp / "users.txt"
    users.write_text("# все\n*\n42  # владелец\n", encoding="utf-8")
    check("«*» — открыт всем", B.access(users) == (True, {42}) and B.allowed_users(users) == {42})
    tg = _TG()
    bot = B.Bot(tg, None, object(), A.Settings(), users_file=users, log=lambda *a: None)
    bot.handle(msg("/help", user=7, chat=7))
    check(
        "гость получает помощь, а не «Нет доступа»",
        tg.sent and "Нет доступа" not in tg.sent[-1][1] and bot.guest(7) and not bot.guest(42),
    )

    tg.sent.clear()
    for _ in range(50):
        bot.handle(msg("/help", user=8, chat=8))
    check(
        "без ограничений: 50 сообщений подряд — 50 ответов", len(tg.sent) == 50, str(len(tg.sent))
    )
    answered = []
    bot.answer = lambda chat_id, session, text, history=None: answered.append(text)
    for i in range(40):
        bot.handle(msg(f"вопрос {i}", user=9, chat=9))
    check("без ограничений: 40 вопросов подряд — все в работу", len(answered) == 40)

    closed = tmp / "closed.txt"
    closed.write_text("42\n", encoding="utf-8")
    tg2 = _TG()
    shut = B.Bot(tg2, None, object(), A.Settings(), users_file=closed, log=lambda *a: None)
    for _ in range(5):
        shut.handle(msg("/start", user=7, chat=7))
    check(
        "закрытый бот: чужому id — один раз, а не на каждое сообщение",
        [t for _, t in tg2.sent if "Нет доступа" in t] and len(tg2.sent) == 1,
        str(len(tg2.sent)),
    )

    tg3 = _TG()
    dup = B.Bot(tg3, None, object(), A.Settings(), users_file=users, log=lambda *a: None)
    tg3.queue = [msg("/help", update_id=5), msg("/help", update_id=6)]
    dup.poll_once(0)
    tg3.queue = [msg("/help", update_id=5)]  # Телеграм прислал то же ещё раз
    dup.poll_once(0)
    check("одно и то же сообщение дважды — ответ один", len(tg3.sent) == 2, str(len(tg3.sent)))

    first = B._instance_lock("987654:X")
    second = B._instance_lock("987654:X")
    check("вторая копия бота не запускается", first is not None and second is None)
    first.close()
    again = B._instance_lock("987654:X")
    check("первую закрыли — снова можно", again is not None)
    again.close()
    runner = B.Bot(
        _TG("987654:X"), None, object(), A.Settings(), users_file=users, log=lambda *a: None
    )
    held = B._instance_lock("987654:X")
    logged = []
    runner.log = logged.append
    runner.run(threading.Event())
    held.close()
    check("run при запущенной копии — выходит и объясняет", logged and "другом окне" in logged[0])

    tg4 = _TG()
    q = B.Bot(tg4, None, object(), A.Settings(), users_file=users, log=lambda *a: None)
    q.slots.acquire()  # модель занята другим
    done = threading.Event()
    t = threading.Thread(target=lambda: (q._wait_turn(5), done.set()))
    t.start()
    t.join(0.5)
    check(
        "модель занята — «в очереди» с местом",
        not done.is_set() and any("в очереди, перед вами 1" in s for _, s in tg4.sent),
    )
    q.slots.release()
    t.join(2)
    check(
        "очередь подошла — отвечает",
        done.is_set() and any("Очередь подошла" in s for _, s in tg4.sent),
    )
    q.slots.release()


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api_url = f"http://127.0.0.1:{server.server_address[1]}"

    tmp = Path(tempfile.mkdtemp())
    users = tmp / "users.txt"
    users.write_text("# я\n42\n", encoding="utf-8")

    real = (
        db.connect,
        A.answer,
        S.hybrid_search,
        db.documents,
        db.stats,
        db.document_chunks,
        graph_api.datasets_response,
        graph_api.dataset_response,
        A.ollama_status,
    )
    db.connect = fake_connect
    db.stats = lambda conn: {"documents": 1, "chunks": 5, "embedded": 5, "years": (2020, 2020)}
    A.ollama_status = lambda host, model, timeout=5: {"ok": True, "model": model}
    tg = B.Telegram("T0KEN", api=api_url)
    bot = B.Bot(tg, None, object(), A.Settings(), users_file=users, log=lambda *a: None)
    try:
        print("\nПроверка бота: python georag.py tg --check")
        out = []

        def run_check(token):
            out.clear()
            code = B.check(log=out.append, token=token, api=api_url, users_file=users)
            return code, "\n".join(out)

        code, text = run_check("")
        check("нет токена — так и сказано", code == 1 and "Токена нет" in text)
        code, text = run_check("это не токен")
        check("испорченный токен — так и сказано", code == 1 and "не похож на токен" in text)
        good = "123456789:" + "A" * 35
        FakeAPI.webhook = "https://example.org/hook"
        FakeAPI.queue = [msg("привет", user=7, chat=7), msg("а?", user=42, chat=42)]
        code, text = run_check(good)
        check(
            "токен принят, имя бота, webhook убран",
            code == 0
            and "@georag_test_bot" in text
            and "webhook — убран" in text
            and FakeAPI.webhook == "",
            text,
        )
        check(
            "кто писал, пока бот не запущен, и есть ли у него доступ",
            "id 7 (НЕТ в telegram_users.txt)" in text
            and "id 42 (есть доступ)" in text
            and "НЕ запущен" in text,
            text,
        )
        FakeAPI.conflict = True
        code, text = run_check(good)
        check("бот запущен в другом окне — так и сказано", code == 1 and "ДРУГОМ окне" in text)
        FakeAPI.conflict = False
        check(
            "ошибки Телеграма — понятным текстом",
            "токен" in B.explain_error(B.TelegramError("getMe: Unauthorized"))
            and "ДРУГОМ" in B.explain_error(B.TelegramError("getUpdates: Conflict: x")),
        )
        FakeAPI.calls.clear()

        print("\nДоступ и меню")
        bot.handle(msg("привет", user=7, chat=7))
        check("чужому — отказ и его id", "Нет доступа" in last_text() and "7" in last_text())
        check("файл доступа читается с комментариями", B.allowed_users(users) == {42})
        bot.handle(msg("/start"))
        kb = last().get("reply_markup") or {}
        check(
            "/start — помощь и меню внизу экрана",
            "База знаний ГеоRAG" in last_text()
            and kb.get("is_persistent")
            and [b["text"] for row in kb["keyboard"] for b in row]
            == ["Поиск", "Статьи", "Факты", "Новый разговор", "Помощь"],
        )
        check(
            "в помощи: подробно или кратко — пишется в вопросе",
            "так и напишите в вопросе" in last_text() and "Настройки" not in last_text(),
        )
        FakeAPI.calls.clear()
        bot.handle(press("new", user=7, chat=7))
        check("чужой нажал кнопку — отказ всплывашкой", toasts() == ["Нет доступа"] and not sent())

        print("\nВопрос → ответ по базе, кнопки под ответом")
        long_answer = (
            "**Вывод**\nПлотность линеаментов выше у узлов [1].\n\n"
            "- Гравитационные аномалии отмечают интрузии [2] <важно> & проверено.\n"
            + "Ещё строка про разломы [1].\n" * 300
        )

        def fake_answer(conn, emb, question, history, settings, llm=None, planner=None):
            fake_answer.seen = (question, list(history))
            yield {"type": "status", "text": "ищу в базе…"}
            yield {
                "type": "sources",
                "queries": [question, "ore cluster gravity"],
                "dataset": None,
                "parts": [
                    {"question": "как выделяют", "queries": ["q"], "found": 1},
                    {"question": "чем", "queries": ["q2"], "found": 1},
                ],
                "extra": ["ore cluster gravity"],
                "missing": ["гравиметрия"],
                "rounds": 2,
                "checked": 14,
                "judged": True,
                "sources": [
                    {
                        "n": 1,
                        "doc_id": "d1",
                        "title": "Линеаменты & узлы",
                        "year": 2021,
                        "pages": [14],
                        "url": "https://x/1",
                        "text": "Полный текст фрагмента один.",
                        "headings": ["Методы"],
                    },
                    {
                        "n": 2,
                        "doc_id": "d2",
                        "title": "Гравиметрия",
                        "year": 2019,
                        "pages": [],
                        "url": "",
                        "text": "Текст два.",
                        "headings": [],
                    },
                ],
            }
            yield {"type": "token", "text": long_answer}
            yield {
                "type": "done",
                "mode": "база",
                "used": [1, 2],
                "unknown": [],
                "coverage": [300, 302],
                "model": "qwen3:14b",
                "seconds": 12.5,
            }

        A.answer = fake_answer
        FakeAPI.calls.clear()
        bot.handle(msg("как выделяют рудные узлы?"))
        edits, sends = texts("editMessageText"), texts()
        first = sent(("sendMessage",))[0]
        check(
            "сначала «работаю» с кнопкой «Остановить»",
            sends[0].startswith("<i>") and labels(first) == ["Остановить"],
        )
        check("показано «печатает»", any(m == "sendChatAction" for m, _ in FakeAPI.calls))
        check(
            "ответ правкой того же сообщения, разметка",
            any(t.startswith("<b>Вывод</b>") for t in edits),
            edits[-1][:60],
        )
        everything = "\n".join(edits + sends)
        check(
            "пункты и спецсимволы экранированы",
            "• Гравитационные аномалии" in everything and "&lt;важно&gt; &amp;" in everything,
        )
        check(
            "длинный ответ — несколько сообщений в пределах лимита",
            len(sends) >= 3 and all(len(t) <= 4096 for t in sends + edits),
            str(len(sends)),
        )
        final = last()
        check(
            "кнопки — только у последнего куска",
            labels(final) == ["[1]", "[2]", "Как искал", "Новый разговор"]
            and sum(bool(buttons(p)) for p in sent(("sendMessage",))[1:]) == 1,
            str(labels(final)),
        )
        check(
            "источники со ссылками и страницами",
            '<a href="https://x/1">Линеаменты &amp; узлы</a> — 2021, с. 14' in everything,
        )
        check(
            "подвал: покрытие ссылками и модель",
            "со ссылкой 300 из 302" in everything and "qwen3:14b" in everything,
        )
        check("разговор запомнен", len(bot.sessions[42].history) == 2)

        FakeAPI.calls.clear()
        bot.handle(press(data_of(final, "[1]")))
        frag = last()
        check(
            "кнопка [1] — фрагмент целиком",
            "Полный текст фрагмента один." in frag["text"] and "Методы" in frag["text"],
        )
        check(
            "у фрагмента — ссылка на издателя и «Вся статья»",
            labels(frag) == ["Статья у издателя", "Вся статья"],
            str(labels(frag)),
        )
        check("нажатие подтверждено", toasts() == [""])

        db.documents = lambda conn, limit=15: [
            {
                "doc_id": "d1",
                "title": "Линеаменты & узлы",
                "year": 2021,
                "journal": "Геология",
                "url": "https://x/1",
                "authors": ["Иванов И.И."],
                "source": "openalex",
                "pages": 12,
                "chunks": 5,
            }
        ]
        db.document_chunks = lambda conn, doc_id: [
            {"ord": i, "text": f"Кусок {i} статьи.", "headings": ["Введение"], "pages": [i + 1]}
            for i in range(9)
        ]
        bot.handle(press(data_of(frag, "Вся статья")))
        card = last()
        check(
            "«Вся статья» — карточка статьи",
            "Иванов И.И." in card["text"]
            and "Фрагментов в базе: 5" in card["text"]
            and labels(card) == ["У издателя", "Фрагменты статьи"],
            str(labels(card)),
        )
        bot.handle(press(data_of(card, "Фрагменты статьи")))
        chunks = last()
        check(
            "фрагменты статьи — список со страницами",
            "фрагментов 9" in chunks["text"]
            and "стр. 1 из 2" in chunks["text"]
            and "Дальше ›" in labels(chunks),
        )
        bot.handle(press(data_of(chunks, "Дальше ›"), message_id=77))
        page2 = last()
        check(
            "«Дальше» — та же карточка правкой",
            page2.get("message_id") == 77
            and "стр. 2 из 2" in page2["text"]
            and "Кусок 6" in page2["text"],
        )
        bot.handle(press(data_of(page2, "7")))
        check("номер — фрагмент статьи", "Кусок 6 статьи." in last_text())

        FakeAPI.calls.clear()
        bot.handle(press(data_of(final, "Как искал")))
        how = last_text()
        check(
            "«Как искал» — части, второй круг, запросы",
            "вопрос разобран на части: 1) как выделяют" in how
            and "не хватало: гравиметрия; запросы: «ore cluster gravity»" in how
            and "модель прочла найденное (фрагментов: 14)" in how
            and "кругов поиска: 2" in how
            and "• как выделяют рудные узлы?" in how
            and "процитировано 2" in how,
            how[:300],
        )

        bot.handle(msg("/f 1"))
        check("/f 1 — по-прежнему работает", "Полный текст фрагмента один." in last_text())
        bot.handle(msg("/f 9"))
        check("/f с чужим номером — подсказка", "1–2" in last_text())

        FakeAPI.calls.clear()
        bot.handle(press("f:999:1"))
        check("кнопка от забытого ответа — всплывашка «устарела»", toasts() == [B.STALE])

        bot.handle(msg("Новый разговор"))
        check(
            "«Новый разговор» из меню",
            bot.sessions[42].history == [] and "новый разговор" in last_text().lower(),
        )

        print("\nНастроек нет — состояние по /stats")
        bot.handle(msg("/brief"))
        check("/brief — подсказка: пишите в вопросе", "напишите в вопросе" in last_text())
        bot.handle(msg("Настройки"))
        check(
            "кнопка «Настройки» из старого меню — подсказка и новое меню",
            "напишите в вопросе" in last_text()
            and "Настройки" not in str(last().get("reply_markup")),
        )
        bot.handle(msg("/stats"))
        check(
            "/stats — база и модель",
            "статей 1, фрагментов 5" in last_text() and "qwen3:14b — готова" in last_text(),
            last_text(),
        )

        print("\nОтвет без базы")

        def general(conn, emb, question, history, settings, llm=None, planner=None):
            yield {"type": "general", "queries": [question], "text": A.NOT_IN_BASE}
            yield {"type": "token", "text": "Общий ответ."}
            yield {
                "type": "done",
                "mode": "без базы",
                "used": [],
                "unknown": [],
                "coverage": [0, 0],
                "model": "qwen3:14b",
                "seconds": 3,
            }

        A.answer = general
        bot.handle(msg("как приготовить борщ?"))
        check(
            "пометка «в базе нет» и «не из базы знаний»",
            "такой информации нет" in last_text() and "не из базы знаний" in last_text(),
        )
        check(
            "без базы — без кнопок фрагментов",
            labels(last()) == ["Как искал", "Новый разговор"],
            str(labels(last())),
        )

        print("\nРазметка не принята Телеграмом — ответ всё равно доходит")
        A.answer = fake_answer
        FakeAPI.reject_html = True
        FakeAPI.calls.clear()
        bot.handle(msg("вопрос"))
        FakeAPI.reject_html = False
        check(
            "повтор без тегов, кнопки на месте",
            "<b>" not in last_text() and "Ещё строка" in last_text() and "[1]" in labels(last()),
        )

        print("\nОшибка модели")

        def broken(*a, **k):
            yield {"type": "error", "error": "Ollama не отвечает"}

        A.answer = broken
        bot.handle(msg("вопрос"))
        check("ошибка — сообщением", "Ollama не отвечает" in last_text())

        print("\nОстановить, пока пишет")
        gate = threading.Event()

        def slow(conn, emb, question, history, settings, llm=None, planner=None):
            slow.closed = False
            try:
                yield {
                    "type": "sources",
                    "queries": [question],
                    "sources": [{"n": 1, "doc_id": "d1", "title": "Т", "text": "т", "url": ""}],
                }
                for i in range(200):
                    yield {"type": "token", "text": f"слово{i} "}
                    if i == 3:
                        gate.set()
                    time.sleep(0.02)
                yield {
                    "type": "done",
                    "mode": "база",
                    "used": [1],
                    "unknown": [],
                    "coverage": [1, 1],
                    "model": "m",
                    "seconds": 1,
                }
            finally:
                slow.closed = True

        A.answer = slow
        bot.background = True
        FakeAPI.calls.clear()
        bot.handle(msg("долгий вопрос"))
        gate.wait(5)
        stop_data = next(
            b["callback_data"]
            for p in sent(("sendMessage",))
            for b in buttons(p)
            if b["text"] == "Остановить"
        )
        bot.handle(msg("второй вопрос"))
        check("пока пишет — второй вопрос не берётся", "Ещё пишу прошлый ответ" in last_text())
        bot.handle(press(stop_data))
        bot.workers[42].join(5)
        check(
            "«Остановить» — ответ обрезан и помечен",
            "остановлено" in last_text() and "слово199" not in last_text(),
            last_text()[-80:],
        )
        check("поток модели закрыт", slow.closed)
        check("после остановки бот свободен", not bot.sessions[42].busy)
        bot.handle(press(stop_data))
        check("«Остановить» на готовом ответе — всплывашка", toasts()[-1] == "Ответ уже готов")
        bot.background = False

        print("\nПоиск кнопкой, листание")
        S.hybrid_search = lambda conn, emb, q, **kw: [
            Hit(
                chunk_id=i,
                doc_id=f"d{i}",
                ord=0,
                text=f"Текст про линеаменты {i}.",
                title=f"Статья {i}",
                year=2020,
                url=f"https://x/{i}",
                pages=[3],
                similarity=0.71,
                found_by="оба",
            )
            for i in range(1, 8)
        ]
        FakeAPI.calls.clear()
        bot.handle(msg("Поиск"))
        check(
            "«Поиск» — спрашивает, что искать, с отменой",
            "Что искать" in last_text() and labels(last()) == ["Отмена"],
        )
        bot.handle(msg("линеаменты"))
        res = last()
        check(
            "следующий текст — запрос, выдача со страницами",
            "<b>Поиск:</b> линеаменты — найдено 7" in res["text"]
            and "похожесть 71%" in res["text"]
            and "стр. 1 из 2" in res["text"],
            res["text"][:200],
        )
        check(
            "кнопки выдачи: номера, листание, «Спросить бота»",
            labels(res)[:5] == ["1", "2", "3", "4", "5"]
            and "Дальше ›" in labels(res)
            and "Спросить бота об этом" in labels(res),
            str(labels(res)),
        )
        check("ожидание запроса снято", bot.sessions[42].awaiting is None)
        bot.handle(press(data_of(res, "2")))
        check("номер — фрагмент из выдачи", "Текст про линеаменты 2." in last_text())
        bot.handle(press(data_of(res, "Дальше ›"), message_id=9))
        check(
            "вторая страница правкой",
            "Статья 6" in last_text()
            and "стр. 2 из 2" in last_text()
            and "Спросить бота об этом" in labels(last()),
        )
        A.answer = fake_answer
        bot.handle(press(data_of(res, "Спросить бота об этом")))
        check("«Спросить бота» — запрос уходит вопросом", fake_answer.seen[0] == "линеаменты")
        bot.handle(msg("/search"))
        bot.handle(press("cancel", message_id=3))
        check("отмена поиска", bot.sessions[42].awaiting is None and last_text() == "Отменено.")
        S.hybrid_search = lambda conn, emb, q, **kw: []
        bot.handle(msg("/search чепуха"))
        check("пусто — так и сказано", "близкого в базе не нашлось" in last_text())

        print("\nСтатьи")
        bot.handle(msg("Статьи"))
        check(
            "«Статьи» — список", "Статей в базе: 1" in last_text() and "фрагментов 5" in last_text()
        )
        bot.handle(press(data_of(last(), "1")))
        check("номер статьи — карточка", "Иванов И.И." in last_text())
        bot.handle(msg("/stats"))
        check("/stats", "фрагментов 5" in last_text())

        print("\nДатасеты")
        graph_api.datasets_response = lambda conn, syn, limit=30: {
            "datasets": [
                {"name": "Анабарский щит", "documents": 3, "facts": 1},
                {"name": "Оленёкское поднятие", "documents": 1, "facts": 2},
            ],
            "total": 2,
        }
        data = {
            "name": "Анабарский щит",
            "includes": ["Билляхская зона"],
            "documents": 3,
            "facts": [
                {
                    "about": "Анабарский щит",
                    "direction": "←",
                    "relation": "приурочено к",
                    "other": "золотое оруденение",
                    "documents": 2,
                    "fragments": 3,
                    "quotes": [
                        {
                            "quote": "Золото приурочено к щиту.",
                            "doc_id": "d1",
                            "title": "Т",
                            "year": 2020,
                            "url": "",
                            "chunk_id": 1,
                        }
                    ],
                    "chunk_ids": [1],
                }
            ],
        }

        gold = {
            "name": "золотое оруденение",
            "includes": [],
            "documents": 2,
            "facts": [
                {
                    "about": "золотое оруденение",
                    "direction": "→",
                    "relation": "приурочено к",
                    "other": "Анабарский щит",
                    "documents": 2,
                    "fragments": 3,
                    "quotes": [],
                    "chunk_ids": [1],
                }
            ],
        }

        def one_dataset(conn, syn, payload):
            if payload["name"].lower() == "золотое оруденение":
                return gold
            if payload["name"].lower() != "анабарский щит":
                raise graph_api.RequestError(
                    "такого в фактах из статей нет", payload["name"], ["Анабарский щит"]
                )
            return data

        graph_api.dataset_response = one_dataset
        bot.handle(msg("Факты"))
        lst = last()
        check(
            "«Факты» — сущности кнопками",
            labels(lst)
            == ["Анабарский щит · 3 ст. · 1 факт.", "Оленёкское поднятие · 1 ст. · 2 факт."],
            str(labels(lst)),
        )
        FakeAPI.calls.clear()
        bot.handle(press(data_of(lst, labels(lst)[0]), message_id=11))
        pic = photos()[-1] if photos() else ""
        check(
            "сущность — картинка связей (PNG) с подписью",
            "image/png" in pic and "PNG" in pic,
            pic[:80],
        )
        from georag.tg import picture as P

        two = {
            "name": "щит",
            "includes": ["зона"],
            "documents": 2,
            "facts": [
                {
                    "about": "щит",
                    "direction": "←",
                    "relation": "приурочено к",
                    "other": "золото",
                    "documents": 2,
                },
                {
                    "about": "зона",
                    "direction": "→",
                    "relation": "контролирует",
                    "other": "дайки",
                    "documents": 1,
                },
            ],
        }
        edges, near = P.neighborhood(two)
        check(
            "картинка: соседи и части, часть связана с центром «входит в»",
            near == ["золото", "зона", "дайки"]
            and ("зона", "входит в", "щит") in edges
            and ("зона", "контролирует", "дайки") in edges,
            str(edges),
        )
        pos = P.layout("щит", near, edges)
        check(
            "картинка: все узлы внутри кадра, связанное через часть — дальше от центра",
            all(0 < x < P.W and 0 < y < P.H for x, y in pos.values())
            and abs(pos["дайки"][1] - P.H / 2) + abs(pos["дайки"][0] - P.W / 2)
            > abs(pos["зона"][1] - P.H / 2) + abs(pos["зона"][0] - P.W / 2),
        )
        png, _ = P.render(two)
        check("картинка рисуется: PNG", png[:4] == b"\x89PNG")
        many = {
            "name": "X",
            "includes": [],
            "documents": 1,
            "facts": [
                {"about": "X", "direction": "→", "relation": "r", "other": f"n{i}", "documents": 1}
                for i in range(40)
            ],
        }
        check(
            "больше NEIGHBORS соседей — на картинке только главные",
            len(P.neighborhood(many)[1]) == P.NEIGHBORS,
        )
        check(
            "в подписи — сущность и сколько фактов",
            "Анабарский щит" in multipart_field(pic, "caption")
            and "фактов 1" in multipart_field(pic, "caption"),
        )
        check(
            "под картинкой — соседи, цитаты, таблица, вопрос, назад",
            photo_labels(pic)
            == [
                "→ золотое оруденение",
                "→ Билляхская зона",
                "Цитаты",
                "Таблица для Excel",
                "Спросить бота",
                "‹ Все сущности",
            ],
            str(photo_labels(pic)),
        )
        FakeAPI.calls.clear()
        bot.handle(press(photo_data(pic, "→ золотое оруденение"), message_id=12))
        media = photos("editMessageMedia")
        check(
            "сосед — картинка в том же сообщении",
            media
            and multipart_field(media[0], "message_id") == "12"
            and "золотое оруденение" in multipart_field(media[0], "media"),
        )
        FakeAPI.calls.clear()
        bot.handle(press(photo_data(pic, "Цитаты")))
        cardd = last()
        check(
            "«Цитаты» — факты списком с кнопками",
            "Факты: Анабарский щит" in cardd["text"]
            and "золотое оруденение — приурочено к — Анабарский щит" in cardd["text"]
            and labels(cardd)
            == ["Картинка связей", "Таблица для Excel", "Спросить бота", "‹ Все сущности"],
            cardd["text"][:200],
        )
        FakeAPI.calls.clear()
        bot.handle(press(data_of(cardd, "Таблица для Excel")))
        docs = [p for m, p in FakeAPI.calls if m == "sendDocument"]
        check(
            "«Таблица для Excel» — файл CSV",
            docs
            and "filename" in docs[0]
            and "csv" in docs[0]
            and toasts() == ["Таблица отправлена"],
        )
        bot.handle(press(data_of(cardd, "Спросить бота")))
        check(
            "«Спросить бота» — вопрос о сущности",
            fake_answer.seen[0].startswith("Что известно о «Анабарский щит»"),
        )
        bot.handle(press(data_of(cardd, "‹ Все сущности"), message_id=11))
        check("«Все сущности» — назад к списку", "Факты из статей" in last_text())
        FakeAPI.calls.clear()
        bot.handle(msg("/facts анабарский щит"))
        check(
            "/facts с именем — картинка связей с кнопками",
            photos() and "Таблица для Excel" in photo_labels(photos()[-1]),
        )
        FakeAPI.calls.clear()
        bot.handle(msg("/dataset анабарский щит"))
        check("старая команда /dataset — то же", photos())
        bot.handle(msg("/dataset Марс"))
        check(
            "чужое имя — понятный отказ и что есть",
            "нет" in last_text() and "Анабарский щит" in last_text(),
        )

        bot.handle(msg("/чтоэто"))
        check("неизвестная команда — подсказка", "Помощь" in last_text())

        print("\nЦикл опроса")
        FakeAPI.queue = [msg("/stats", update_id=10), press("noop") | {"update_id": 11}]
        n = bot.poll_once(timeout=0)
        check(
            "сообщения и нажатия из очереди обработаны, смещение сдвинуто",
            n == 2 and bot.offset == 12,
        )
        getup = [p for m, p in FakeAPI.calls if m == "getUpdates"][-1]
        check("бот просит и нажатия кнопок", "callback_query" in getup["allowed_updates"])
        check("split не рвёт короткий текст", B.split("раз\nдва") == ["раз\nдва"])
        check(
            "датасет назван в «как искал»",
            "собрано из фактов графа о «Анабар»" in B.search_notes({"dataset": "Анабар"}),
        )
        check(
            "callback_data короче 64 байт",
            all(
                len(b["callback_data"].encode()) <= 64
                for _, p in FakeAPI.calls
                if isinstance(p, dict)
                for b in buttons(p)
                if "callback_data" in b
            ),
        )
        long_names = U.datasets_keyboard(
            [{"name": "Очень длинное название территории " * 3, "documents": 1, "facts": 1}]
        )
        check(
            "длинное имя не попадает в callback",
            long_names["inline_keyboard"][0][0]["callback_data"] == "ds:0",
        )
    finally:
        (
            db.connect,
            A.answer,
            S.hybrid_search,
            db.documents,
            db.stats,
            db.document_chunks,
            graph_api.datasets_response,
            graph_api.dataset_response,
            A.ollama_status,
        ) = real
        server.shutdown()
    test_open_access()

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
