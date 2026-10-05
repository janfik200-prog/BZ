"""Веб-интерфейс к базе знаний: поиск, статьи, чат-бот и граф связей.

    python -m georag.web.server            открыть http://localhost:8000
    python -m georag.web.server --port 9000

Модель эмбеддингов грузится один раз при запуске, а не на каждый запрос:
в терминале каждый поиск стоит полминуты на загрузку весов, здесь — доли
секунды.

Никаких новых библиотек: сервер из стандартной поставки Python, страница —
один файл рядом. Наружу ничего не смотрит, слушает только localhost.

Для страницы:

    GET  /api/stats                    сколько статей и фрагментов
    GET  /api/search?q=...             гибридный поиск
    GET  /api/documents                список статей
    GET  /api/document?doc_id=...      разбор статьи по фрагментам
    GET  /api/chat/status              жива ли Ollama и скачана ли модель
    POST /api/chat                     вопрос чат-боту; ответ потоком, строка JSON на событие
    GET  /api/network                  граф: сущности, связи с цитатами, статьи
    GET  /api/datasets                 у каких сущностей больше всего фактов
    GET  /api/dataset?name=...         датасет сущности: все факты о ней с цитатами
                                       (POST — то же JSON в теле); &format=csv — для Excel

С другой машины к нему ходят через SSH-туннель: порт снаружи не открывается.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from ..chat import answer as chat
from ..common import add_db_args, add_embedder_args, add_llm_args
from ..graph import api as graph_api
from ..graph import store as graph
from ..graph.synonyms import DEFAULT_PATH as SYNONYMS_PATH
from ..graph.synonyms import SynonymsError, synonym_variants
from ..graph.synonyms import load as load_synonyms
from ..index import db
from ..index.embed import build_embedder
from ..index.search import hybrid_search

PAGE = Path(__file__).with_name("index.html")


class State:
    """Общее на весь сервер: адрес базы, загруженная модель, словарь синонимов."""

    dsn: str | None = None
    embedder: Any = None
    chat_settings = chat.Settings()
    synonyms_path = None
    _syn = None
    _syn_mtime = None

    @classmethod
    def synonyms(cls) -> Any:
        """Словарь перечитывается, если файл поменяли, — сервер можно не перезапускать."""
        path = cls.synonyms_path or SYNONYMS_PATH
        mtime = path.stat().st_mtime if path.exists() else None
        if cls._syn is None or mtime != cls._syn_mtime:
            cls._syn = load_synonyms(path)
            cls._syn_mtime = mtime
        return cls._syn


def _hit_to_dict(hit: Any) -> dict[str, Any]:
    return {
        # doc_id и ord нужны, чтобы из выдачи открыть разбор статьи
        # и показать, из какого места взят найденный кусок.
        "doc_id": hit.doc_id,
        "ord": hit.ord,
        "title": hit.title,
        "year": hit.year,
        "journal": hit.journal,
        "url": hit.url,
        "authors": hit.authors[:3],
        "headings": hit.headings[-2:],
        "pages": hit.pages,
        "text": " ".join(hit.text.split()),
        "found_by": hit.found_by,
        "score": round(hit.score, 4),
        "similarity": hit.similarity,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "georag"

    # Стандартный лог пишет каждую картинку — в консоли от него только шум.
    def log_message(self, fmt: Any, *args: Any) -> None:
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, Any], code: int = 200) -> None:
        self._send(
            code,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802 — имя задано базовым классом
        route = urlparse(self.path)
        query = parse_qs(route.query)

        try:
            if route.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route.path == "/api/stats":
                self._json(self._stats())
            elif route.path == "/api/search":
                self._json(self._search(query))
            elif route.path == "/api/documents":
                with db.connect(State.dsn) as conn:
                    self._json({"documents": db.documents(conn)})
            elif route.path == "/api/datasets":
                self._graph(graph_api.datasets_response)
            elif route.path == "/api/dataset":
                self._dataset({k: v[0] for k, v in query.items() if v})
            elif route.path == "/api/network":
                self._graph(graph_api.network_response)
            elif route.path == "/api/chat/status":
                self._json(chat.ollama_status(State.chat_settings.host, State.chat_settings.model))
            elif route.path == "/api/document":
                doc_id = (query.get("doc_id") or [""])[0]
                with db.connect(State.dsn) as conn:
                    self._json({"doc_id": doc_id, "chunks": db.document_chunks(conn, doc_id)})
            else:
                self._json({"error": "нет такой страницы"}, 404)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # страницу закрыли, пока отвечали
        except Exception as exc:  # noqa: BLE001 — ошибка уходит на страницу, сервер живёт
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path not in ("/api/chat", "/api/dataset"):
                self._json({"error": "нет такого адреса"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 256 * 1024:
                self._json({"error": "запрос слишком большой"}, 413)
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json({"error": "тело запроса — не JSON", "detail": str(exc)}, 400)
                return
            if route.path == "/api/chat":
                self._chat(payload)
            else:
                self._dataset(payload)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- чат-бот ------------------------------------------------------------ #
    def _chat(self, payload: dict[str, Any]) -> None:
        """Ответ потоком: строка JSON на событие, страница рисует по мере прихода."""
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        settings = State.chat_settings
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        events = None
        try:
            with db.connect(State.dsn) as conn:
                db.autocommit(conn)  # ответ пишется минутами — без открытой транзакции
                events = chat.answer(
                    conn, State.embedder, str(payload.get("question") or ""), history, settings
                )
                for event in events:
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # страницу закрыли или нажали «Стоп» — не страшно
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                line = {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError:
                pass
        finally:
            if events is not None:
                events.close()  # закрывает и поток от Ollama: она перестаёт писать

    # -- граф знаний: сеть, датасеты ------------------------------------------ #
    def _graph(self, build: Any) -> None:
        """build(conn, синонимы) → ответ. Неизвестное имя — 400, словарь не читается — 500."""
        try:
            syn = State.synonyms()
            with db.connect(State.dsn) as conn:
                self._json(build(conn, syn))
        except graph_api.RequestError as exc:
            self._json(exc.payload(), 400)
        except SynonymsError as exc:
            self._json({"error": f"словарь синонимов: {exc}"}, 500)

    def _dataset(self, payload: dict[str, Any]) -> None:
        """Датасет сущности: JSON, а с format=csv — таблица для Excel."""
        payload = dict(payload)
        if str(payload.pop("format", "")).lower() != "csv":
            self._graph(lambda conn, syn: graph_api.dataset_response(conn, syn, payload))
            return
        try:
            with db.connect(State.dsn) as conn:
                data = graph_api.dataset_response(conn, State.synonyms(), payload)
        except graph_api.RequestError as exc:
            self._json(exc.payload(), 400)
            return
        body = graph_api.dataset_csv(data).encode("utf-8")
        name = quote(f"датасет-{data['name']}.csv")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header(
            "Content-Disposition", f"attachment; filename=\"dataset.csv\"; filename*=UTF-8''{name}"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- действия ---------------------------------------------------------- #
    def _stats(self) -> dict[str, Any]:
        with db.connect(State.dsn) as conn:
            info = db.stats(conn)
        lo, hi = info["years"]
        return {
            "documents": info["documents"],
            "chunks": info["chunks"],
            "year_from": lo,
            "year_to": hi,
        }

    def _search(self, query: dict[str, Any]) -> dict[str, Any]:
        text = (query.get("q") or [""])[0].strip()
        if not text:
            return {"hits": [], "query": ""}

        def number(name: Any, default: Any = None) -> Any:
            raw = (query.get(name) or [""])[0]
            return int(raw) if raw.isdigit() else default

        def fraction(name: Any, default: Any = 0.0) -> Any:
            raw = (query.get(name) or [""])[0]
            try:
                return float(raw)
            except ValueError:
                return default

        # Другие написания из словаря синонимов: «Донбасс» ищется и как «Donetsk basin».
        try:
            also = synonym_variants(text, State.synonyms())
        except SynonymsError:
            also = []
        with db.connect(State.dsn) as conn:
            hits = hybrid_search(
                conn,
                State.embedder,
                text,
                limit=number("limit", 10),
                year_from=number("year_from"),
                max_per_doc=number("per_doc", 2),
                min_similarity=fraction("min_similarity"),
                require_words=(query.get("require_words") or [""])[0] == "1",
                alternatives=also,
            )
        return {"query": text, "also": also, "hits": [_hit_to_dict(h) for h in hits]}


class Server(ThreadingHTTPServer):
    """Порт — только наш. Стандартный сервер Python ставит SO_REUSEADDR, и на Windows
    второй запуск молча садился на тот же порт: страница попадала то в старый процесс
    со старым кодом, то в новый, а в каждом работал свой Телеграм-бот."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Страницу закрыли или обновили, пока сервер отвечал, — не ошибка, журнал не засоряем."""
        import sys as _sys

        if isinstance(
            _sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)
        ):
            return
        super().handle_error(request, client_address)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Веб-интерфейс к базе знаний ГеоRAG")
    parser.add_argument("--port", type=int, default=8000)
    add_db_args(parser)
    add_embedder_args(parser)
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    parser.add_argument(
        "--tg", action="store_true", help="заодно запустить Телеграм-бота (модель векторов общая)"
    )
    add_llm_args(parser)
    args = parser.parse_args(argv)

    # Порт — первым делом: если сервер уже запущен в другом окне, второй не стартует
    # (и не запускает второго Телеграм-бота).
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError:
        print(
            f"Порт {args.port} занят: сервер уже запущен в другом окне. Закройте его "
            f"(Ctrl+C) и запустите заново — или другой порт: --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1

    State.dsn = args.dsn
    State.chat_settings = chat.Settings(model=args.model, host=args.ollama_host)
    try:
        syn = State.synonyms()
        print(f"Словарь синонимов: {len(syn.groups)} имён")
    except SynonymsError as exc:
        print(f"Словарь синонимов не читается: {exc}", file=sys.stderr)
        server.server_close()
        return 1

    # Проверяем базу до старта: лучше упасть здесь, чем показать пустую страницу.
    try:
        with db.connect(args.dsn) as conn:
            info = db.stats(conn)
            graph.init(conn)  # таблицы графа — один раз при старте, не на каждый запрос
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Поднимите её: docker compose up -d", file=sys.stderr)
        server.server_close()
        return 1

    print(f"В базе: {info['documents']} статей, {info['chunks']} чанков")
    print("Загружаю модель эмбеддингов (один раз на весь запуск)...")
    State.embedder = build_embedder(args.embedder, device=args.device)
    State.embedder.encode_one("прогрев")  # чтобы первый поиск не ждал загрузку весов

    if args.tg:
        from ..tg.bot import start_in_thread

        try:
            start_in_thread(State.dsn, State.embedder, State.chat_settings)
        except Exception as exc:  # noqa: BLE001 — без бота страница всё равно работает
            print(f"Телеграм-бот не запустился: {exc}", file=sys.stderr)
    else:
        from ..tg.bot import read_token

        if read_token():
            print(
                "Телеграм-бот НЕ запущен: токен есть, но нужен ключ --tg — "
                "python georag.py web --tg"
            )

    address = f"http://localhost:{args.port}"
    status = chat.ollama_status(State.chat_settings.host, State.chat_settings.model)
    print(
        f"Чат-бот: {State.chat_settings.model} — " + ("готов" if status["ok"] else status["error"])
    )
    print(f"\nГотово: {address}")
    print("Остановить — Ctrl+C\n")

    if not args.no_browser:
        import webbrowser

        webbrowser.open(address)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Остановлен.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
