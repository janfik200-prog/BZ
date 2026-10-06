"""Проверка чат-бота: поиск, сообщения модели, ссылки, поток Ollama. Без базы и модели.

Запуск:  python tests/chat_smoke_test.py

Ollama подменяется маленьким сервером в этом же процессе: он отвечает так же,
как настоящая, — строками JSON, кусками.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.chat import answer as A  # noqa: E402
from georag.index.search import Hit  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def source(n: int, title: str = "Статья", text: str = "Текст фрагмента.") -> dict:
    return {
        "n": n,
        "doc_id": f"d{n}",
        "ord": 0,
        "title": title,
        "year": 2021,
        "journal": None,
        "authors": [],
        "pages": [14],
        "headings": [],
        "url": "",
        "text": text,
        "similarity": 0.7,
        "found_by": "оба",
    }


def hit(chunk_id: int, doc: str, similarity: float | None, fts: bool = False) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc,
        ord=chunk_id,
        text=f"текст {chunk_id}",
        title=f"Статья {doc}",
        similarity=similarity,
        fts_rank=1 if fts else None,
        fts_strict=fts,
        vec_rank=1 if similarity is not None else None,
        found_by="вектор",
    )


# --------------------------------------------------------------------------- #
def test_query() -> None:
    print("\nЧто искать")
    check(
        "обычный вопрос ищется как есть",
        A.search_query("Как выделяют рудные узлы по линеаментам?", [])
        == "Как выделяют рудные узлы по линеаментам?",
    )
    history = [
        {"role": "user", "content": "гидротермальные изменения по снимкам"},
        {"role": "assistant", "content": "…"},
    ]
    check(
        "короткий вопрос вдогонку — вместе с прошлым",
        A.search_query("а по ASTER?", history)
        == "гидротермальные изменения по снимкам а по ASTER?",
    )
    long_q = "какие методы применяли для прогноза золота на Анабарском щите"
    check("длинный новый вопрос — без прошлого", A.search_query(long_q, history) == long_q)
    check(
        "короткий, но новый вопрос — без прошлого",
        A.search_query("Как испечь ржаной хлеб?", history) == "Как испечь ржаной хлеб?",
    )
    check(
        "вопрос со ссылкой на сказанное — вместе с прошлым",
        A.search_query("какие там породы?", history).startswith("гидротермальные"),
    )


def test_messages() -> None:
    print("\nСообщения для модели")
    long_text = "слово " * 1000
    sources = [
        source(1, "Линеаменты", "Плотность линеаментов выше вблизи узлов."),
        source(2, "Длинная", long_text),
    ]
    history = [
        (
            {"role": "user", "content": f"вопрос {i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": f"ответ {i}"}
        )
        for i in range(10)
    ]
    msgs = A.build_messages("Что известно?", sources, history, "qwen3:14b")
    check(
        "первым — правила",
        msgs[0]["role"] == "system" and "ТОЛЬКО по фрагментам" in msgs[0]["content"],
    )
    check("история обрезана до двух обменов", len(msgs) == 1 + 4 + 1, str(len(msgs)))
    last = msgs[-1]["content"]
    check(
        "фрагменты пронумерованы",
        "[1] «Линеаменты» (2021, с. 14)" in last and "[2] «Длинная»" in last,
    )
    check("длинный фрагмент укорочен", len(last) < 3200, str(len(last)))
    check("вопрос в конце", "Вопрос: Что известно?" in last)
    check("Qwen3 — без размышлений вслух", last.endswith("/no_think"))
    other = A.build_messages("Что известно?", sources, [], "llama3")
    check("другой модели /no_think не нужен", "/no_think" not in other[-1]["content"])
    check(
        "правило: только по фрагментам, иначе — одна фраза отказа",
        "даже если знаешь их сам" in msgs[0]["content"]
        and "«В базе знаний нет ответа на этот вопрос.»" in msgs[0]["content"],
    )
    general = A.general_messages("Что такое рудный узел?", history, "qwen3:14b")
    check(
        "ответ без базы — свои правила, без фрагментов",
        general[0]["content"] == A.GENERAL_SYSTEM and "Фрагменты" not in general[-1]["content"],
    )
    check("ответ без базы — запрет на ссылки", "Не ставь номера" in A.GENERAL_SYSTEM)
    check("ответ без базы помнит разговор", len(general) == 1 + 4 + 1)


def test_refusal() -> None:
    print("\nОтказ модели узнаётся")
    check("точная фраза", A.is_refusal("В базе знаний нет ответа на этот вопрос."))
    check("в кавычках и жирным", A.is_refusal("**«В базе знаний нет ответа»** на этот вопрос"))
    check("с «е» вместо «ё» и регистром", A.is_refusal("в БАЗЕ знаний нет ответа на вопрос"))
    check("обычный ответ — не отказ", not A.is_refusal("В базе знаний описаны три метода [1]."))


def test_citations() -> None:
    print("\nСсылки в ответе")
    used, unknown = A.citations("Утверждение [1]. Ещё [2][3]. И [1, 4]. Выдумка [9].", 4)
    check("номера собраны", used == [1, 2, 3, 4], str(used))
    check("несуществующий номер замечен", unknown == [9], str(unknown))
    check("без ссылок — пусто", A.citations("Просто текст.", 3) == ([], []))
    check("год в скобках — не ссылка", A.citations("В 2021 году (см. [2]).", 3) == ([2], []))


def test_coverage() -> None:
    print("\nСколько утверждений со ссылкой")
    text = (
        "По данным статей базы:\n"
        "- Оруденение приурочено к узлам пересечения разломов и зонам дробления [1].\n"
        "- Калиевый метасоматоз и окварцевание выделяют по снимкам ASTER.[2][3]\n"
        "- Россыпи в погребённых палеодолинах указывают на коренной источник выше.\n\n"
        "Прямых оценок золотоносности Анабарского щита во фрагментах нет [4]."
    )
    check(
        "считаются только утверждения, вводная фраза — нет",
        A.coverage(text) == (3, 4),
        str(A.coverage(text)),
    )
    meta = (
        "По данным статей базы, рудные узлы выделяют по сочетанию нескольких признаков.\n"
        "Оруденение приурочено к узлам пересечения разломов и зонам дробления [1].\n"
        "Для Анабарского щита в найденных фрагментах прямых оценок золотоносности нет."
    )
    check(
        "фразы о самих фрагментах ссылки не требуют",
        A.coverage(meta) == (1, 1),
        str(A.coverage(meta)),
    )
    check(
        "факт без ссылки — считается",
        A.coverage("Плотность линеаментов максимальна вблизи рудных узлов на севере щита.")
        == (0, 1),
    )
    check(
        "ссылка после точки относится к предложению",
        A.coverage("Калиевый метасоматоз сменяется окварцеванием по всей зоне.[2]") == (1, 1),
    )
    check("пустой ответ", A.coverage("") == (0, 0))
    headed = (
        "**1) Роль узлов пересечения разломов в геологии**\n"
        "Узлы пересечения разломов контролируют оруденение в Восточном Донбассе [1].\n"
        "### 2) Как используют гравитационные аномалии при прогнозе?\n"
        "- **Аномалии золота:** золото сопровождается аномалиями серебра в зоне."
    )
    check(
        "заголовки разделов — не утверждения; пункт с жирным началом — утверждение",
        A.coverage(headed) == (1, 2),
        str(A.coverage(headed)),
    )
    check(
        "правило: без номера фрагмента не писать, и вводные тоже",
        "Чего нельзя подкрепить номером фрагмента — не пиши" in A.SYSTEM,
    )
    block = A.fragment_block(dict(source(1, "Т", "Индекс описан в [12] и [3, 5–7]."), n=1))
    check(
        "ссылки статьи на литературу не похожи на номера фрагментов",
        "(лит. 12)" in block
        and "(лит. 3, 5–7)" in block
        and "[12]" not in block
        and block.startswith("[1] «Т»"),
        block,
    )


def test_close() -> None:
    print("\nВ ответ идут только близкие фрагменты")
    real = A.hybrid_search
    try:

        def fake(results):
            def search(conn, emb, query, limit, max_per_doc):
                return [h for h in results.get(query, [])]

            return search

        A.hybrid_search = fake(
            {
                "q1": [hit(1, "a", 0.7), hit(2, "b", 0.6), hit(3, "c", 0.5)],
                "q2": [hit(3, "c", 0.5), hit(4, "d", 0.2)],
            }
        )
        sources = A.find_sources(None, None, ["q1", "q2"], A.Settings())
        check(
            "дальний фрагмент отброшен",
            [s["doc_id"] for s in sources] == ["c", "a", "b"],
            str([s["doc_id"] for s in sources]),
        )
        check("найденное двумя запросами — выше", sources[0]["doc_id"] == "c")

        A.hybrid_search = fake({"q": [hit(1, "a", 0.44), hit(2, "b", 0.3), hit(3, "c", 0.1)]})
        check(
            "близкого нет — пусто, ближайшее не подсовывается",
            A.find_sources(None, None, ["q"], A.Settings()) == [],
        )

        A.hybrid_search = fake({"q": [hit(1, "a", None, fts=True), hit(2, "b", 0.1)]})
        sources = A.find_sources(None, None, ["q"], A.Settings())
        check("совпадение по словам — близкое", [s["doc_id"] for s in sources] == ["a"])

        A.hybrid_search = fake({"q": [hit(i, "a", 0.8) for i in range(1, 6)] + [hit(9, "b", 0.7)]})
        sources = A.find_sources(None, None, ["q"], A.Settings())
        check(
            "не больше трёх фрагментов из статьи, пока есть другие",
            [s["doc_id"] for s in sources][:4] == ["a", "a", "a", "b"],
            str([s["doc_id"] for s in sources]),
        )

        A.hybrid_search = fake({})
        check("в базе пусто — пусто", A.find_sources(None, None, ["q"], A.Settings()) == [])
    finally:
        A.hybrid_search = real


def test_think() -> None:
    print("\nБлок размышлений срезается")

    def run(chunks):
        f = A.ThinkFilter()
        return "".join(f.feed(c) for c in chunks) + f.flush()

    check("обычный текст проходит", run(["Ответ ", "по делу [1]."]) == "Ответ по делу [1].")
    check("пустой блок Qwen3 срезан", run(["<think>\n\n</think>\n\n", "Ответ."]) == "Ответ.")
    check(
        "блок, разрезанный на куски, срезан",
        run(["<thi", "nk>рассуждаю", " долго</th", "ink>\n", "Итог [2]."]) == "Итог [2].",
    )
    check("текст с «<» в начале не теряется", run(["<5% ", "каолинита"]) == "<5% каолинита")
    check("незакрытый блок не показывается", run(["<think>думаю и думаю"]) == "")


# --------------------------------------------------------------------------- #
#  Поддельная Ollama
# --------------------------------------------------------------------------- #
class _Ollama(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen3:14b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(payload)
        if payload.get("format") == "json":
            content = json.dumps(
                {
                    "queries": [
                        "рудный узел критерии выделения",
                        "ore cluster delineation",
                        "Ore cluster delineation",
                        "lineament density",
                    ]
                }
            )
            body = json.dumps({"message": {"content": "<think>\n</think>" + content}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if type(self).mode == "missing":
            body = b'{"error":"model \\"qwen3:14b\\" not found, try pulling it first"}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in ["<think>\n\n</think>\n\n", "Плотность линеаментов ", "выше у узлов [1]", "."]:
            self.wfile.write(
                (json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode()
            )
        self.wfile.write(
            (
                json.dumps({"message": {"content": ""}, "done": True, "eval_count": 12}) + "\n"
            ).encode()
        )


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ollama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_ollama() -> None:
    print("\nПоток от Ollama")
    server, host = _serve()
    try:
        settings = A.Settings(host=host)
        _Ollama.mode, _Ollama.seen = "ok", []
        parts = list(A.stream_ollama([{"role": "user", "content": "?"}], settings))
        text = "".join(p.get("text", "") for p in parts)
        check("куски собираются в ответ", "Плотность линеаментов выше у узлов [1]." in text)
        check("конец с числом токенов", parts[-1] == {"done": True, "tokens": 12})
        sent = _Ollama.seen[0]
        check(
            "размышления выключены, ответ потоком",
            sent["think"] is False and sent["stream"] is True,
        )

        _Ollama.mode = "missing"
        try:
            list(A.stream_ollama([], settings))
            check("нет модели — понятная ошибка", False)
        except A.LLMError as exc:
            check("нет модели — понятная ошибка", "ollama pull qwen3:14b" in str(exc), str(exc))

        _Ollama.mode, _Ollama.seen = "ok", []
        planned = A.plan_queries("как выделяют рудные узлы?", [], settings)
        check(
            "запросы от модели: без повторов, не больше трёх",
            planned
            == ["рудный узел критерии выделения", "ore cluster delineation", "lineament density"],
            str(planned),
        )
        sent = _Ollama.seen[0]
        check("запросы — JSON без размышлений", sent["format"] == "json" and sent["think"] is False)
        check(
            "разбор вопроса — в том же окне, что ответ (Ollama не перезагружает модель)",
            sent["options"]["num_ctx"] == settings.num_ctx,
        )
        check(
            "модель молчит — поиск идёт по самому вопросу",
            A.plan_queries("вопрос", [], A.Settings(host="http://127.0.0.1:9")) == [],
        )

        _Ollama.seen = []
        plan = A.Plan([A.Part("q", ["q"])], main="q")
        batch = [{"id": 1, "hit": hit(1, "a", 0.9), "part": 1, "round": 1}]
        verdict = A.judge_fragments("вопрос", plan, batch, [], settings)
        sent = _Ollama.seen[0]
        check(
            "отбор моделью — JSON, окно то же, что у ответа, кандидаты в сообщении",
            isinstance(verdict, dict)
            and sent["format"] == "json"
            and sent["options"]["num_ctx"] == settings.num_ctx
            and "[1] «Статья a»" in sent["messages"][-1]["content"],
        )
        check(
            "модель молчит при отборе — None (дальше отбор по порогу)",
            A.judge_fragments("вопрос", plan, batch, [], A.Settings(host="http://127.0.0.1:9"))
            is None,
        )

        check("статус: модель на месте", A.ollama_status(host, "qwen3:14b")["ok"])
        missing = A.ollama_status(host, "qwen3:32b")
        check(
            "статус: чего не хватает",
            not missing["ok"] and "ollama pull qwen3:32b" in missing["error"],
        )
    finally:
        server.shutdown()

    dead = A.ollama_status("http://127.0.0.1:9", "qwen3:14b", timeout=2)
    check("статус: Ollama не запущена", not dead["ok"] and "ollama.com" in dead["error"])
    try:
        list(A.stream_ollama([], A.Settings(host="http://127.0.0.1:9")))
        check("молчащая Ollama — понятная ошибка", False)
    except A.LLMError as exc:
        check("молчащая Ollama — понятная ошибка", "не отвечает" in str(exc))


# --------------------------------------------------------------------------- #
#  Всё вместе, поиск подменён
# --------------------------------------------------------------------------- #
def _script(*answers):
    """Поддельная модель: на каждый вызов — свой ответ кусками; запоминает сообщения."""
    calls = []

    def llm(messages, settings):
        calls.append(messages)
        for piece in answers[len(calls) - 1]:
            yield {"text": piece}
        yield {"done": True, "tokens": 9}

    return llm, calls


def test_answer() -> None:
    print("\nОтвет целиком")
    real = A.find_sources
    try:
        seen_queries = []

        def found(conn, emb, queries, settings, **kw):
            seen_queries.extend(queries)
            return [source(1, "Линеаменты"), source(2)]

        A.find_sources = found
        planner = lambda *a: ["ore cluster delineation", "Что с линеаментами?"]  # noqa: E731

        llm, calls = _script(
            [
                "<think></think>Плотность линеаментов выше вблизи рудных узлов [1]. ",
                "И ещё одно утверждение, которое опирается на выдуманный фрагмент [7].",
            ]
        )
        events = list(
            A.answer(
                None,
                None,
                "Что с линеаментами?",
                [],
                A.Settings(),
                judge=None,
                llm=llm,
                planner=planner,
            )
        )
        kinds = [e["type"] for e in events]
        check(
            "нашлось — источники раньше текста",
            kinds.index("sources") < kinds.index("token"),
            str(kinds),
        )
        check(
            "искал сам вопрос и запросы модели, без повторов",
            seen_queries == ["Что с линеаментами?", "ore cluster delineation"],
            str(seen_queries),
        )
        text = "".join(e["text"] for e in events if e["type"] == "token")
        check("размышления в ответ не попали", text.startswith("Плотность"), repr(text[:30]))
        done = events[-1]
        check(
            "ответ из базы помечен как из базы", done["mode"] == "база" and "general" not in kinds
        )
        check("процитированный фрагмент отмечен", done["used"] == [1])
        check("выдуманная ссылка замечена", done["unknown"] == [7])
        check(
            "посчитано, у скольких утверждений ссылка",
            done["coverage"] == [2, 2],
            str(done["coverage"]),
        )
        check("модель спрошена один раз", len(calls) == 1)

        llm, calls = _script(
            ["В базе зна", "ний нет ответа на этот вопрос."],
            ["Рудный узел — ", "группа сближенных месторождений."],
        )
        events = list(
            A.answer(
                None,
                None,
                "Что такое рудный узел?",
                [],
                A.Settings(),
                judge=None,
                llm=llm,
                planner=lambda *a: [],
            )
        )
        kinds = [e["type"] for e in events]
        text = "".join(e["text"] for e in events if e["type"] == "token")
        check(
            "модель не нашла ответа во фрагментах — статьи не показываются",
            "sources" not in kinds,
            str(kinds),
        )
        check(
            "и сказано, что в базе нет",
            any(e["type"] == "general" and "такой информации нет" in e["text"] for e in events),
        )
        check(
            "отказ в текст ответа не попал",
            text == "Рудный узел — группа сближенных месторождений.",
            repr(text),
        )
        check(
            "второй вызов — без фрагментов",
            len(calls) == 2 and calls[1][0]["content"] == A.GENERAL_SYSTEM,
        )
        check(
            "ответ помечен как не из базы",
            events[-1]["mode"] == "без базы" and events[-1]["used"] == [],
        )

        A.find_sources = lambda conn, emb, queries, settings, **kw: []
        llm, calls = _script(["Ответ из общих знаний."])
        events = list(
            A.answer(
                None,
                None,
                "про что-то чужое",
                [],
                A.Settings(),
                judge=None,
                llm=llm,
                planner=lambda *a: [],
            )
        )
        kinds = [e["type"] for e in events]
        check(
            "ничего не найдено — сразу ответ без базы, модель спрошена один раз",
            "general" in kinds
            and "sources" not in kinds
            and len(calls) == 1
            and calls[0][0]["content"] == A.GENERAL_SYSTEM,
            str(kinds),
        )
        check("пометка раньше текста", kinds.index("general") < kinds.index("token"))

        def broken(messages, settings):
            raise A.LLMError("Ollama не отвечает")
            yield  # noqa: B901 — генератор

        events = list(
            A.answer(
                None,
                None,
                "вопрос",
                [],
                A.Settings(),
                judge=None,
                llm=broken,
                planner=lambda *a: [],
            )
        )
        check("ошибка модели — событием, не падением", events[-1]["type"] == "error")
        A.find_sources = found
        events = list(
            A.answer(
                None,
                None,
                "вопрос",
                [],
                A.Settings(),
                judge=None,
                llm=broken,
                planner=lambda *a: [],
            )
        )
        check("ошибка при ответе по базе — тоже событием", events[-1]["type"] == "error")

        events = list(A.answer(None, None, "   ", [], A.Settings(), judge=None, llm=broken))
        check("пустой вопрос — ошибка", events == [{"type": "error", "error": "пустой вопрос"}])
    finally:
        A.find_sources = real


class _NeighborConn:
    """Соединение, которое на запрос соседей отдаёт заданные строки."""

    def __init__(self, rows):
        self.rows, self.sql = rows, []

    def cursor(self):
        conn = self

        class Cur:
            def execute(self, sql, params=None):
                conn.sql.append((sql, params))

            def fetchall(self):
                return conn.rows

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Cur()


def test_style() -> None:
    print("\nОдин режим: объём ответа — по просьбе в вопросе")
    check(
        "настроек «кратко/подробно» нет",
        not hasattr(A.Settings(), "detail") and not hasattr(A, "BRIEF"),
    )
    msgs = A.build_messages("Что известно?", [source(1, "Т", "текст")], [], "qwen3:14b")
    rules = msgs[0]["content"]
    check(
        "правило: объём — как просит человек",
        "как просит человек в вопросе" in rules
        and "подробно, развёрнуто" in rules
        and "кратко, в двух словах" in rules
        and "по существу" in rules,
    )
    check("частичный ответ разрешён", "лишь частично" in rules)
    check("отказ по-прежнему одной фразой", "«В базе знаний нет ответа на этот вопрос.»" in rules)
    check("без базы — тоже по просьбе", "как просит человек" in A.GENERAL_SYSTEM)
    check("окно модели под длинный ответ", A.Settings().num_ctx >= 12000)
    check("фрагментов для ответа — восемь, до трёх из статьи", A.TOP_K >= 8 and A.PER_DOC >= 3)

    short = source(1, "Т", "Заголовок и две строки.")
    short.update(doc_id="d", ord=4, headings=["Методы"], pages=[3])
    long_ = source(2, "Т2", "длинный " * 200)
    long_.update(doc_id="e", ord=1, headings=["X"], pages=[1])
    conn = _NeighborConn([("d", 5, "Продолжение раздела о методах.", ["Методы"], [4])])
    A.add_neighbors(conn, [short, long_])
    check(
        "к короткому фрагменту добавлено продолжение",
        short["text"].endswith("Продолжение раздела о методах.") and short.get("with_next"),
    )
    check("страницы продолжения учтены", short["pages"] == [3, 4], str(short["pages"]))
    check("длинный не трогается", not long_.get("with_next"))
    check("соседей просят одним запросом", len(conn.sql) == 1)

    other = source(1, "Т", "Короткий.")
    other.update(doc_id="d", ord=4, headings=["Методы"], pages=[3])
    A.add_neighbors(_NeighborConn([("d", 5, "Уже другое.", ["Выводы"], [5])]), [other])
    check("из другого раздела не приклеивается", other["text"] == "Короткий.")

    class Broken:
        def cursor(self):
            raise RuntimeError("база упала")

    safe = source(1, "Т", "Коротко.")
    safe.update(doc_id="d", ord=1)
    A.add_neighbors(Broken(), [safe])
    check("база не ответила — ответ всё равно будет", safe["text"] == "Коротко.")


# --------------------------------------------------------------------------- #
#  Части вопроса, второй круг, «собрать всё»
# --------------------------------------------------------------------------- #
def fake_search(results):
    calls = []

    def search(conn, emb, query, limit, max_per_doc):
        calls.append(query)
        return [Hit(**{**h.__dict__}) for h in results.get(query, [])]

    search.calls = calls
    return search


def test_plan() -> None:
    print("\nРазбор вопроса")
    st = A.Settings()
    plan = A.parse_plan(
        {
            "части": [
                {
                    "вопрос": "какие признаки у рудных узлов",
                    "запросы": [
                        "признаки рудных узлов",
                        "ore cluster criteria",
                        "ore cluster criteria",
                    ],
                },
                {"вопрос": "какими методами их выделяют", "запросы": ["методы выделения узлов"]},
                {"вопрос": "третья", "запросы": ["q3"]},
                {"вопрос": "четвёртая", "запросы": ["q4"]},
            ],
            "предмет": "Анабарский щит",
            "всё": False,
        },
        "вопрос",
        st,
    )
    check(
        "части и запросы, без повторов",
        [p.queries for p in plan.parts][:2]
        == [["признаки рудных узлов", "ore cluster criteria"], ["методы выделения узлов"]],
        str([p.queries for p in plan.parts]),
    )
    check("частей не больше трёх", len(plan.parts) == 3)
    check("предмет и «всё»", plan.subject == "Анабарский щит" and plan.collect is False)
    plan = A.parse_plan(
        {"части": [{"вопрос": "что известно", "запросы": []}], "территория": "null", "всё": "true"},
        "вопрос",
        st,
    )
    check(
        "«null» строкой — не предмет, «true» строкой — да",
        plan.subject is None and plan.collect is True,
    )
    plan = A.parse_plan({"queries": ["a1", "b2"]}, "вопрос", st)
    check(
        "старый вид ответа — одна часть",
        len(plan.parts) == 1 and plan.parts[0].queries == ["a1", "b2"],
    )
    check(
        "мусор — одна часть без запросов",
        A.parse_plan("ерунда", "вопрос", st).parts[0].queries == [],
    )

    plan = A.make_plan("Что с линеаментами?", [], st, lambda *a: ["q1", "Что с линеаментами?"])
    check(
        "planner со списком — одна часть, сам вопрос первым",
        plan.queries == ["Что с линеаментами?", "q1"] and len(plan.parts) == 1,
        str(plan.queries),
    )
    check("разбор перебирается как список запросов", list(plan) == plan.queries)
    history = [
        {"role": "user", "content": "как выделяют рудные узлы"},
        {"role": "assistant", "content": "…"},
    ]
    plan = A.make_plan(
        "Что известно об Анабаре?",
        history,
        st,
        lambda *a: A.Plan([A.Part("?", [])], subject="Анабар", collect=True),
    )
    check("вопрос о территории — без прошлого вопроса", plan.main == "Что известно об Анабаре?")
    for q in (
        "Что известно об Анабарском щите?",
        "Перечисли признаки на Билляхской зоне",
        "какие методы на Анабаре применяли",
    ):
        check(
            f"«собрать всё» узнаётся по словам: {q}", A.make_plan(q, [], st, lambda *a: []).collect
        )
    check(
        "обычный вопрос — не «собрать всё»",
        not A.make_plan("Как выделяют рудные узлы?", [], st, lambda *a: []).collect,
    )


def test_parts() -> None:
    print("\nУ каждой части — свои места")
    real = A.hybrid_search
    try:
        # Первая часть находится отлично и заняла бы все восемь мест; вторая — хуже.
        A.hybrid_search = fake_search(
            {
                "общий": [hit(i, f"a{i}", 0.9) for i in range(1, 12)],
                "признаки": [hit(i, f"a{i}", 0.9) for i in range(1, 12)],
                "методы": [hit(50, "m1", 0.6), hit(51, "m2", 0.55), hit(52, "m3", 0.3)],
            }
        )
        plan = A.Plan(
            [A.Part("какие признаки", ["признаки"]), A.Part("какие методы", ["методы"])],
            main="общий",
        )
        sources = A.find_sources(None, None, plan, A.Settings())
        docs = [s["doc_id"] for s in sources]
        check("всего не больше восьми", len(sources) == 8, str(docs))
        check("вторая часть не вытеснена", "m1" in docs and "m2" in docs, str(docs))
        check("дальний фрагмент части не взят ради места", "m3" not in docs)
        check(
            "у фрагментов — номер части",
            sources[0]["part"] == 1
            and next(s for s in sources if s["doc_id"] == "m1")["part"] == 2,
        )
        check(
            "найдено по частям посчитано",
            [p.found for p in plan.parts] == [3, 2],
            str([p.found for p in plan.parts]),
        )
        check(
            "один запрос не ищется дважды",
            len(A.hybrid_search.calls) == len(set(A.hybrid_search.calls)),
            str(A.hybrid_search.calls),
        )

        A.hybrid_search = fake_search({"общий": [hit(1, "a", 0.9)], "признаки": [hit(1, "a", 0.9)]})
        plan = A.Plan(
            [A.Part("признаки", ["признаки"]), A.Part("методы", ["методы"])], main="общий"
        )
        sources = A.find_sources(None, None, plan, A.Settings())
        check(
            "по части ничего — found = 0, повторов нет",
            plan.parts[1].found == 0 and [s["doc_id"] for s in sources] == ["a"],
        )
    finally:
        A.hybrid_search = real


def scripted_judge(*answers):
    """Поддельная модель-отборщик: на каждый круг — свой ответ; запоминает, что видела."""
    calls = []

    def judge(question, plan, batch, accepted, settings):
        calls.append(
            {
                "batch": [(c["id"], c["hit"].doc_id) for c in batch],
                "accepted": [c["hit"].doc_id for c in accepted],
                "prompt": A.judge_prompt(question, plan, batch, accepted),
            }
        )
        answer = answers[len(calls) - 1] if len(calls) <= len(answers) else {}
        return answer(batch) if callable(answer) else answer

    judge.calls = calls
    return judge


def ids_of(batch, *docs):
    """Номера кандидатов этих статей — в том порядке, в каком названы статьи."""
    return [c["id"] for d in docs for c in batch if c["hit"].doc_id == d]


def test_judge() -> None:
    print("\nПоиск с моделью: широкий набор, модель отбирает, ищет ещё")
    real = A.hybrid_search
    try:
        A.hybrid_search = fake_search(
            {
                "Что с линеаментами?": [
                    hit(1, "a", 0.9),
                    hit(2, "b", 0.85),
                    hit(3, "c", 0.40),
                    hit(4, "z", 0.2),
                ],
                "lineament density gold": [hit(2, "b", 0.7), hit(7, "g", 0.38), hit(8, "h", 0.1)],
            }
        )
        judge = scripted_judge(
            lambda b: {
                "подходят": ids_of(b, "c", "a"),
                "не_хватает": ["связь с золотом"],
                "запросы": ["lineament density gold", "Что с линеаментами?"],
            },
            lambda b: {
                "подходят": [str(i) for i in ids_of(b, "g")],
                "не_хватает": [],
                "запросы": [],
            },
        )
        llm, calls = _script(["Плотность выше [1], связь с золотом [3]."])
        events = list(
            A.answer(
                None,
                None,
                "Что с линеаментами?",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: [],
                judge=judge,
                datasets=None,
            )
        )
        first = judge.calls[0]
        check(
            "кандидаты — с мягким порогом: 0.40 взят, 0.2 — нет",
            [d for _, d in first["batch"]] == ["a", "b", "c"],
            str(first["batch"]),
        )
        src = next(e for e in events if e["type"] == "sources")
        docs = [s["doc_id"] for s in src["sources"]]
        check(
            "модель отобрала: похожий по словам «b» отброшен, её порядок сохранён",
            docs == ["c", "a", "g"],
            str(docs),
        )
        check(
            "второй круг: только новые кандидаты, уже отобранное названо",
            [d for _, d in judge.calls[1]["batch"]] == ["g"]
            and judge.calls[1]["accepted"] == ["c", "a"]
            and "Уже отобрано раньше" in judge.calls[1]["prompt"],
        )
        check("номера — строкой «[3]» тоже понимаются", "g" in docs)
        check("фрагмент второго круга помечен", src["sources"][2].get("round") == 2)
        check(
            "запросы нового круга — без уже сделанных", src["extra"] == ["lineament density gold"]
        )
        check(
            "в событии: круги, сколько прочла модель, чего не хватало",
            src["rounds"] == 2
            and src["checked"] == 4
            and src["judged"] is True
            and src["missing"] == ["связь с золотом"],
            str({k: src[k] for k in ("rounds", "checked", "missing")}),
        )
        check(
            "модели показаны номера кандидатов и сделанные запросы",
            "[1] «Статья a»" in first["prompt"] and "«Что с линеаментами?»" in first["prompt"],
        )
        check("ответ по отобранному", events[-1]["mode"] == "база" and events[-1]["used"] == [1, 3])
        check(
            "статусы: модель читает, ищет недостающее",
            any("модель читает найденное" in e.get("text", "") for e in events)
            and any(e.get("text", "").startswith("ищу недостающее") for e in events),
        )

        # Кругов не больше трёх, даже если модель всё время просит ещё.
        A.hybrid_search = lambda conn, emb, q, **kw: [hit(abs(hash(q)) % 10000, q[:5], 0.9)]

        def asks_more(n: int):
            """Отбирает всё и просит ещё один запрос — модель, которая не останавливается."""
            return lambda b: {
                "подходят": [c["id"] for c in b],
                "не_хватает": ["ещё"],
                "запросы": [f"ещё запрос {n}"],
            }

        greedy = scripted_judge(*[asks_more(n) for n in range(10)])
        prepared = A.run(
            A.prepare(None, None, "вопрос", [], A.Settings(), lambda *a: [], greedy, None)
        )
        check(
            "кругов не больше трёх",
            prepared.retrieval.rounds == 3 and len(greedy.calls) == 3,
            f"{prepared.retrieval.rounds}, {len(greedy.calls)}",
        )

        # В первом круге пусто — модель подбирает другие запросы.
        A.hybrid_search = fake_search({"ore lineaments": [hit(5, "e", 0.8)]})
        judge = scripted_judge(
            {"подходят": [], "запросы": ["ore lineaments"]},
            lambda b: {"подходят": [c["id"] for c in b]},
        )
        llm, _ = _script(["Нашлось [1]."])
        events = list(
            A.answer(
                None,
                None,
                "про линеаменты",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: [],
                judge=judge,
                datasets=None,
            )
        )
        check(
            "пусто — модель подобрала запросы, нашлось, ответ по базе",
            events[-1]["mode"] == "база"
            and judge.calls[0]["batch"] == []
            and "кандидатов нет" in judge.calls[0]["prompt"],
        )

        # Модель не взяла ничего — «в базе нет», даже если поиск что-то принёс.
        A.hybrid_search = fake_search({"про борщ": [hit(1, "a", 0.5)]})
        llm, calls = _script(["Общий ответ."])
        events = list(
            A.answer(
                None,
                None,
                "про борщ",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: [],
                judge=scripted_judge({"подходят": [], "запросы": []}),
                datasets=None,
            )
        )
        check(
            "модель ничего не взяла — ответ без базы",
            events[-1]["mode"] == "без базы"
            and "sources" not in [e["type"] for e in events]
            and len(calls) == 1,
        )

        # Модель молчит — отбор по порогу, как у обычного поиска.
        A.hybrid_search = fake_search(
            {"Что с линеаментами?": [hit(1, "a", 0.9), hit(3, "c", 0.40)]}
        )
        llm, _ = _script(["Ответ [1]."])
        events = list(
            A.answer(
                None,
                None,
                "Что с линеаментами?",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: [],
                judge=lambda *a: None,
                datasets=None,
            )
        )
        src = next(e for e in events if e["type"] == "sources")
        check(
            "модель молчит — отбор по порогу 0.45",
            [s["doc_id"] for s in src["sources"]] == ["a"] and src["judged"] is False,
        )
        judge = scripted_judge({"подходят": [1], "запросы": ["q2"]}, None)
        A.hybrid_search = fake_search(
            {"Что с линеаментами?": [hit(1, "a", 0.9)], "q2": [hit(2, "b", 0.9)]}
        )
        prepared = A.run(
            A.prepare(
                None, None, "Что с линеаментами?", [], A.Settings(), lambda *a: [], judge, None
            )
        )
        check(
            "модель замолчала во втором круге — берётся отобранное в первом",
            [s["doc_id"] for s in prepared.sources] == ["a"] and prepared.retrieval.judged,
        )

        # Составной вопрос: у каждой части свои места среди отобранного.
        A.hybrid_search = fake_search(
            {
                "признаки": [hit(i, f"a{i}", 0.9) for i in range(1, 12)],
                "методы": [hit(50, "m1", 0.6), hit(51, "m2", 0.55)],
                "общий": [hit(i, f"a{i}", 0.9) for i in range(1, 12)],
            }
        )
        plan = A.Plan(
            [A.Part("какие признаки", ["признаки"]), A.Part("какие методы", ["методы"])],
            main="общий",
        )
        judge = scripted_judge(lambda b: {"подходят": [c["id"] for c in b]})
        llm, calls = _script(["**Признаки** [1] **Методы** [4]"])
        events = list(
            A.answer(
                None,
                None,
                "какие признаки и методы",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: plan,
                judge=judge,
                datasets=None,
            )
        )
        src = next(e for e in events if e["type"] == "sources")
        docs = [s["doc_id"] for s in src["sources"]]
        check(
            "вторая часть не вытеснена", "m1" in docs and "m2" in docs and len(docs) == 8, str(docs)
        )
        check(
            "части — в событии и с числом фрагментов",
            [p["question"] for p in src["parts"]] == ["какие признаки", "какие методы"]
            and src["parts"][1]["found"] == 2,
            str(src["parts"]),
        )
        check("модели при отборе показаны части", "1) какие признаки" in judge.calls[0]["prompt"])
        user = calls[0][-1]["content"]
        check(
            "части вопроса — в сообщении для ответа",
            "2) какие методы" in user and "отдельным разделом" in user,
        )
    finally:
        A.hybrid_search = real


def test_dataset() -> None:
    print("\n«Собрать всё» — из датасета")

    def fact(about, direction, relation, other, docs):
        return {
            "about": about,
            "direction": direction,
            "relation": relation,
            "other": other,
            "documents": docs,
            "fragments": docs,
            "quotes": [],
            "chunk_ids": [],
        }

    data = {
        "name": "Анабарский щит",
        "includes": ["Билляхская зона"],
        "documents": 7,
        "facts": [
            fact("Анабарский щит", "←", "приурочено к", "золотое оруденение", 5),
            fact("Билляхская зона", "→", "выявлено методом", "ASTER", 3),
            fact("Анабарский щит", "←", "проявлено на", "метасоматоз", 1),
        ],
        "evidence": [
            {
                "doc_id": "d9",
                "ord": 4,
                "title": "Разломы Анабара",
                "year": 2020,
                "pages": [3],
                "headings": [],
                "url": "https://x/9",
                "text": "Оруденение контролируется разломами.",
            }
        ],
    }
    summary = A.dataset_summary(data)
    check(
        "сводка: предмет, что входит, статьи",
        "«Анабарский щит»" in summary and "Билляхская зона" in summary and "Статей: 7" in summary,
    )
    check(
        "сводка: все факты «от — связь — к» со статьями",
        "- золотое оруденение — приурочено к — Анабарский щит (5 статей)" in summary
        and "- Билляхская зона — выявлено методом — ASTER (3 статьи)" in summary
        and "метасоматоз — проявлено на — Анабарский щит (1 статья)" in summary,
        summary,
    )
    from georag.graph import api as graph_api
    from georag.graph import synonyms as S

    rows = [
        {
            "id": 1,
            "chunk_id": 1,
            "doc_id": "d1",
            "src": "Billyakh zone",
            "relation": "входит в",
            "dst": "Anabar shield",
            "quote": "q",
            "title": "T",
            "year": 2020,
            "url": "",
            "doi": None,
        },
        {
            "id": 2,
            "chunk_id": 2,
            "doc_id": "d1",
            "src": "золото",
            "relation": "приурочено к",
            "dst": "Билляхская зона",
            "quote": "q",
            "title": "T",
            "year": 2020,
            "url": "",
            "doi": None,
        },
    ]
    g = graph_api.Graph(
        rows,
        S.parse(
            {"имена": {"Анабарский щит": ["Anabar shield"], "Билляхская зона": ["Billyakh zone"]}}
        ),
    )
    check(
        "предмет вопроса узнаётся по тексту, в падеже и по синониму",
        A.resolve_entity(g, None, "Что известно об Анабарском щите?") == "Анабарский щит"
        and A.resolve_entity(g, "Billyakh zone", "?") == "Билляхская зона"
        and A.resolve_entity(g, None, "Как дела на Марсе?") is None,
    )
    check(
        "предмет назван, но в графе его нет — общее слово вопроса не подменяет его",
        A.resolve_entity(g, "Марсианский кратер", "Что известно об Анабарском щите и кратере?")
        is None
        and A.resolve_entity(g, "щит Анабара", "Что известно об Анабарском щите?")
        == "Анабарский щит",
    )
    srcs = A.dataset_sources(data)
    check(
        "сводка — [1], доказательство — [2]",
        [s["n"] for s in srcs] == [1, 2]
        and srcs[0]["kind"] == "датасет"
        and srcs[1]["doc_id"] == "d9",
    )

    real = A.hybrid_search
    try:
        A.hybrid_search = fake_search(
            {"Что известно об Анабарском щите?": [hit(4, "d9", 0.8), hit(11, "k", 0.7)]}
        )
        asked = []

        def datasets(conn, territory, question):
            asked.append((territory, question))
            return data

        plan = A.Plan([A.Part("что известно", [])], subject="Анабарский щит", collect=True)
        llm, calls = _script(["Признаки: структурный контроль [1][2], ASTER [1]."])
        events = list(
            A.answer(
                None,
                None,
                "Что известно об Анабарском щите?",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: plan,
                datasets=datasets,
                judge=scripted_judge(lambda b: {"подходят": [c["id"] for c in b]}),
            )
        )
        check(
            "датасет спрошен с территорией от модели",
            asked == [("Анабарский щит", "Что известно об Анабарском щите?")],
        )
        src = next(e for e in events if e["type"] == "sources")
        docs = [s["doc_id"] for s in src["sources"]]
        check(
            "сначала сводка и доказательства, поиск — после, без повторов",
            docs == ["датасет:Анабарский щит", "d9", "k"],
            str(docs),
        )
        user = calls[0][-1]["content"]
        check(
            "модели сказано перечислить всё из фактов",
            "все факты о предмете вопроса" in user
            and "Перечисли из него всё" in user
            and "Статей: 7" in user,
        )
        check("сводка не обрезана как обычный фрагмент", "метасоматоз" in user)
        check(
            "done: датасет назван",
            events[-1]["dataset"] == "Анабарский щит" and src["dataset"] == "Анабарский щит",
        )
        check(
            "к датасету поиск добавляет отобранное моделью, уже взятое не повторяет",
            events[-1]["rounds"] == 1 and events[-1]["checked"] == 1,
        )

        llm, calls = _script(["Ответ по поиску [1]."])
        events = list(
            A.answer(
                None,
                None,
                "Что известно об Анабарском щите?",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: plan,
                datasets=lambda *a: None,
                judge=scripted_judge(lambda b: {"подходят": [c["id"] for c in b]}),
            )
        )
        src = next(e for e in events if e["type"] == "sources")
        check(
            "датасета нет — обычный поиск",
            src["dataset"] is None and [s["doc_id"] for s in src["sources"]] == ["d9", "k"],
        )

        asked.clear()
        llm, _ = _script(["Ответ [1]."])
        plain = A.Plan([A.Part("как выделяют узлы", [])])
        list(
            A.answer(
                None,
                None,
                "Как выделяют рудные узлы?",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: plain,
                datasets=datasets,
                judge=None,
            )
        )
        check("обычный вопрос — датасет не спрашивается", asked == [])
        check(
            "без базы данных датасета нет", A.collect_dataset(None, "Анабарский щит", "?") is None
        )
    finally:
        A.hybrid_search = real


# --------------------------------------------------------------------------- #
#  Исправления по ревью
# --------------------------------------------------------------------------- #
def test_review_fixes() -> None:
    print("\nИсправления: отбор всех кандидатов, вдогонку, не по теме, история, окно")
    real = A.hybrid_search
    try:
        # Три части по 8 кандидатов и вопрос целиком — модель видит всех, пачками.
        A.hybrid_search = lambda conn, emb, q, **kw: [
            hit(1000 * (sum(map(ord, q)) % 97) + i, f"{q}-{i}", 0.8) for i in range(12)
        ]
        plan = A.Plan([A.Part(f"часть {i}", [f"q{i}"]) for i in (1, 2, 3)], main="общий")
        judge = scripted_judge(*[lambda b: {"подходят": [c["id"] for c in b]}] * 6)
        prepared = A.run(
            A.prepare(None, None, "вопрос", [], A.Settings(), lambda *a: plan, judge, None)
        )
        shown = [d for call in judge.calls for _, d in call["batch"]]
        check(
            "все кандидаты показаны модели, пачками не больше JUDGE_MAX",
            len(shown) > A.JUDGE_MAX * 2
            and len(set(shown)) == len(shown)
            and all(len(c["batch"]) <= A.JUDGE_MAX + A.JUDGE_SLACK for c in judge.calls),
            f"{len(shown)}, {len(judge.calls)}",
        )
        check("у третьей части — все её кандидаты", sum(d.startswith("q3") for d in shown) == 8)
        check("модель прочла всё", prepared.retrieval.checked == len(shown))

        # Новый круг ничего не добавил — поиск останавливается, даже если модель просит ещё.
        A.hybrid_search = lambda conn, emb, q, **kw: [hit(sum(map(ord, q)) % 10000, q[:5], 0.9)]
        lazy = scripted_judge(
            lambda b: {
                "подходят": [c["id"] for c in b],
                "не_хватает": ["x"],
                "запросы": ["второй запрос"],
            },
            lambda b: {"подходят": [], "не_хватает": ["x"], "запросы": ["третий"]},
            lambda b: {"подходят": [c["id"] for c in b]},
        )
        prepared = A.run(
            A.prepare(None, None, "вопрос", [], A.Settings(), lambda *a: [], lazy, None)
        )
        check(
            "круг без новых фрагментов — последний",
            len(lazy.calls) == 2 and prepared.retrieval.rounds == 2,
            str(len(lazy.calls)),
        )
        enough = scripted_judge(
            lambda b: {
                "подходят": [c["id"] for c in b],
                "не_хватает": [],
                "запросы": ["лишний запрос"],
            }
        )
        A.run(A.prepare(None, None, "вопрос", [], A.Settings(), lambda *a: [], enough, None))
        check(
            "всего хватает — новых кругов нет, хоть модель и предложила запрос",
            len(enough.calls) == 1,
        )
        check(
            "запрос-повтор узнаётся по основам слов",
            A.is_repeat(
                "аномалии геохимические золоторудных", ["геохимические аномалии золоторудного"]
            )
            and not A.is_repeat("плотность линеаментов", ["геохимические аномалии"]),
        )

        # Не по теме: один круг, пусто — короткий отказ без ответа из общих знаний.
        A.hybrid_search = fake_search({})
        off = A.Plan([A.Part("хлеб", ["рецепт хлеба"])], on_topic=False)
        judge = scripted_judge({"подходят": [], "не_хватает": ["рецепт"], "запросы": ["выпечка"]})
        llm, calls = _script(["Рецепт…"])
        events = list(
            A.answer(
                None,
                None,
                "Как испечь хлеб?",
                [],
                A.Settings(),
                llm=llm,
                planner=lambda *a: off,
                judge=judge,
                datasets=None,
            )
        )
        general = next(e for e in events if e["type"] == "general")
        check(
            "не по теме — отказ, модель не пишет ответ из общих знаний",
            general.get("off_topic")
            and not calls
            and events[-1]["mode"] == "без базы"
            and "token" not in [e["type"] for e in events],
        )
        check("не по теме — один круг поиска", len(judge.calls) == 1)
        llm, calls = _script(["Рецепт…"])
        list(
            A.answer(
                None,
                None,
                "Как испечь хлеб?",
                [],
                A.Settings(general_off_topic=True),
                llm=llm,
                planner=lambda *a: off,
                judge=scripted_judge({}),
                datasets=None,
            )
        )
        check("по настройке — прежний ответ из общих знаний", len(calls) == 1)
    finally:
        A.hybrid_search = real

    plan = A.parse_plan(
        {
            "вопрос_целиком": "Что известно о Персияновском разломе по космоснимкам?",
            "части": [{"вопрос": "x", "запросы": ["q"]}],
            "по_теме": "false",
        },
        "а по космоснимкам?",
        A.Settings(),
    )
    check(
        "разбор: вопрос целиком и «не по теме»",
        plan.standalone.startswith("Что известно о Персияновском") and plan.on_topic is False,
    )
    history = [
        {"role": "user", "content": "Что известно о Персияновском разломе?"},
        {"role": "assistant", "content": "…"},
    ]
    made = A.make_plan("а по космоснимкам?", history, A.Settings(), lambda *a: plan)
    check("вдогонку ищется вопрос целиком от модели", made.main == plan.standalone)
    msgs = A.build_messages(
        "а по космоснимкам?", [source(1), source(2)], history, "qwen3:14b", made
    )
    check("модели для ответа дан и вопрос целиком", plan.standalone in msgs[-1]["content"])

    turns = A.past_turns(
        [
            {"role": "user", "content": "вопрос"},
            {"role": "assistant", "content": "Факт [1][2]. Ещё [3, 4].", "mode": "база"},
            {"role": "user", "content": "хлеб?"},
            {"role": "assistant", "content": "Рецепт.", "mode": "без базы"},
        ]
    )
    check(
        "в истории нет старых номеров фрагментов",
        turns[1]["content"] == "Факт. Ещё.",
        turns[1]["content"],
    )
    check("ответ из общих знаний в истории помечен", turns[3]["content"].startswith(A.GENERAL_MARK))

    one = [dict(source(1, "Одна статья"), doc_id="d"), dict(source(2, "Одна статья"), doc_id="d")]
    check(
        "все фрагменты из одной статьи — модели сказано не обобщать",
        "из одной статьи «Одна статья»"
        in A.build_messages("?", one, [], "qwen3:14b")[-1]["content"],
    )
    check(
        "правило: называть, к чему относится утверждение",
        "не выдавай частный случай" in A.SYSTEM and "**Заголовок**" not in A.SYSTEM,
    )

    big = [source(i, "Т", "слово " * 400) for i in range(1, 9)]
    long_history = [{"role": r, "content": "реплика " * 300} for r in ("user", "assistant") * 2]
    tight = A.Settings(num_ctx=7000)
    kept, turns = A.fit_budget("вопрос", big, long_history, tight)
    size = A._tokens(A.build_messages("вопрос", kept, turns, tight.model))
    check(
        "сообщение влезает в окно вместе с ответом",
        size <= tight.num_ctx - A.ANSWER_RESERVE and not turns and 3 <= len(kept) < 8,
        f"{size}, {len(kept)}, {len(turns)}",
    )
    check(
        "в большом окне ничего не выброшено",
        A.fit_budget("вопрос", big[:3], [], A.Settings())[0] == big[:3],
    )

    real_find = A.find_sources
    try:
        A.find_sources = lambda conn, emb, queries, settings, **kw: [source(1), source(2)]
        talk = [
            {"role": "user", "content": "Как выделяют рудные узлы?"},
            {"role": "assistant", "content": "Ишимбинский разлом — зона смятия [1]."},
        ]
        new_q = A.Plan([A.Part("?", [])], standalone="Что известно о Персияновском разломе?")
        llm, calls = _script(["Ответ [1]."])
        list(
            A.answer(
                None,
                None,
                "Что известно о Персияновском разломе?",
                talk,
                A.Settings(),
                llm=llm,
                planner=lambda *a: new_q,
                judge=None,
                datasets=None,
            )
        )
        check(
            "самостоятельный вопрос — прошлый разговор модели не показан",
            [m["role"] for m in calls[0]] == ["system", "user"]
            and "Ишимбинский" not in str(calls[0]),
        )
        follow = A.Plan([A.Part("?", [])], standalone="Как выделяют рудные узлы по космоснимкам?")
        llm, calls = _script(["Ответ [1]."])
        list(
            A.answer(
                None,
                None,
                "а по космоснимкам?",
                talk,
                A.Settings(),
                llm=llm,
                planner=lambda *a: follow,
                judge=None,
                datasets=None,
            )
        )
        check("вопрос вдогонку — разговор показан", len(calls[0]) == 4)
    finally:
        A.find_sources = real_find

    data = {"name": "X", "documents": 2, "facts": [], "pass": {"passed": 5, "chunks": 10}}
    check(
        "сводка датасета не называет себя полной, пока граф построен не весь",
        "полный список" not in A.dataset_summary(data) and "5 из 10" in A.dataset_summary(data),
    )


def test_graph_candidates() -> None:
    print("\nКандидаты из графа для вопроса о названном предмете")
    real = A.hybrid_search
    try:
        A.hybrid_search = fake_search({"Что с Персияновским разломом?": [hit(1, "a", 0.9)]})
        from_graph = [
            Hit(
                chunk_id=77,
                doc_id="g",
                ord=3,
                text="Дайки приурочены к разлому.",
                title="Статья g",
                found_by="граф",
            )
        ]
        asked = []

        def graph_hits(conn, subject, question):
            asked.append(subject)
            return from_graph

        judge = scripted_judge(lambda b: {"подходят": [c["id"] for c in b]})
        plan = A.Plan([A.Part("?", [])], subject="Персияновский разлом")
        prepared = A.run(
            A.prepare(
                None,
                None,
                "Что с Персияновским разломом?",
                [],
                A.Settings(),
                lambda *a: plan,
                judge,
                None,
                graph_hits,
            )
        )
        batch = [d for _, d in judge.calls[0]["batch"]]
        check(
            "фрагменты из графа — модели в кандидаты, первыми, без порога близости",
            asked == ["Персияновский разлом"] and batch[0] == "g" and "a" in batch,
            str(batch),
        )
        check(
            "в событии — сколько дал граф",
            prepared.info()["graph"] == 1 and prepared.sources[0]["found_by"] == "граф",
        )
        asked.clear()
        A.run(
            A.prepare(
                None,
                None,
                "Как выделяют узлы?",
                [],
                A.Settings(),
                lambda *a: A.Plan([A.Part("?", [])]),
                scripted_judge({}),
                None,
                graph_hits,
            )
        )
        check("вопрос без предмета — граф не спрашивается", asked == [])
        check(
            "без базы — из графа ничего",
            A.graph_candidates(None, "Персияновский разлом", "?") == [],
        )
    finally:
        A.hybrid_search = real


def test_drop_uncited() -> None:
    """Фразы без ссылки на фрагмент — слова модели, а не статей: из ответа уходят."""
    print("\nФразы без ссылки")
    answer = "\n".join(
        [
            "**Признаки рудных узлов**",
            "Рудные узлы играют важнейшую роль в размещении всех месторождений региона.",
            "Узлы приурочены к пересечениям разломов северо-восточного простирания [1].",
            "Методы выделения применяются следующие:",
            "- Плотность линеаментов считают методом kernel density по снимкам [2].",
            "- Метод в целом широко применяется во многих регионах мира и даёт хорошие результаты.",
            "",
            "**Общий вывод**",
            "Таким образом, все перечисленные признаки очень важны для прогноза оруденения.",
            "Во фрагментах не сказано, как выделяют узлы по радарным снимкам.",
        ]
    )
    new, removed = A.drop_uncited(answer)
    check("фразы без ссылки убраны", len(removed) == 3, str(removed))
    check("со ссылкой остались", "[1]" in new and "[2]" in new)
    check("подводка к списку осталась", "применяются следующие:" in new)
    check("фраза о самих фрагментах осталась", "Во фрагментах не сказано" in new)
    check("пустой пункт списка ушёл", "широко применяется" not in new)
    check("заголовок с текстом остался", "**Признаки рудных узлов**" in new)
    check(
        "ссылка после точки — у своего предложения",
        A.drop_uncited(
            "Узлы связаны с разломами. [1] Это доказано многократно во всех регионах мира."
        )[0]
        == "Узлы связаны с разломами. [1]",
    )
    check(
        "ответ без таких фраз не меняется",
        A.drop_uncited("Узлы связаны с разломами [1].") == ("Узлы связаны с разломами [1].", []),
    )

    real = A.find_sources
    try:
        A.find_sources = lambda *a, **kw: [source(1, "Линеаменты")]
        llm, _ = _script(
            [
                "Плотность линеаментов выше вблизи рудных узлов [1]. ",
                "Это очень важный и общепризнанный признак для всех геологов мира.",
            ]
        )
        events = list(
            A.answer(
                None,
                None,
                "Что с линеаментами?",
                [],
                A.Settings(),
                judge=None,
                llm=llm,
                planner=lambda *a: ["линеаменты"],
            )
        )
        kinds = [e["type"] for e in events]
        revised = next((e for e in events if e["type"] == "revised"), None)
        check(
            "после ответа пришёл очищенный текст",
            revised is not None and kinds.index("revised") < kinds.index("done"),
            str(kinds),
        )
        check(
            "в очищенном нет фразы без ссылки",
            revised is not None
            and "общепризнанный" not in revised["text"]
            and "[1]" in revised["text"],
        )
        done = events[-1]
        check(
            "в итоге — сколько убрано, покрытие по очищенному",
            done["removed"] == 1 and done["coverage"] == [1, 1],
            str(done),
        )
    finally:
        A.find_sources = real


def main() -> int:
    test_query()
    test_messages()
    test_citations()
    test_refusal()
    test_coverage()
    test_close()
    test_think()
    test_ollama()
    test_answer()
    test_style()
    test_plan()
    test_parts()
    test_judge()
    test_dataset()
    test_review_fixes()
    test_graph_candidates()
    test_drop_uncited()

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
