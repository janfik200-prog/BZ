"""Маленький веб-интерфейс к базе знаний.

    python -m georag.web.server            открыть http://localhost:8000
    python -m georag.web.server --port 9000

Зачем он, если есть команда search: модель эмбеддингов грузится один раз при
запуске, а не на каждый запрос. В терминале каждый поиск стоит полминуты на
загрузку весов; здесь — доли секунды, и можно спокойно пробовать формулировки.

Никаких новых библиотек: сервер из стандартной поставки Python, страница —
один файл рядом. Наружу ничего не смотрит, слушает только localhost.

Для базы данных проекта здесь же контракт (подробно — graph/api.py):

    GET  /api/concepts                 словарь и сколько доказательств у понятий
    POST /api/concept                  запрос по шаблону, JSON в теле
    GET  /api/concept?concept=...      то же из адресной строки — для проверки руками
    GET  /api/graph                    схема: понятия — территории — методы
    GET  /api/network                  сеть знаний: статьи и всё, о чём они

С другой машины к нему ходят через SSH-туннель: порт снаружи не открывается.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..index import db
from ..index.embed import build_embedder
from ..index.search import hybrid_search
from ..graph import api as graph_api
from ..graph import store as graph
from ..graph.vocabulary import VocabularyError, load as load_vocabulary

PAGE = Path(__file__).with_name("index.html")


class State:
    """Общее на весь сервер: адрес базы, загруженная модель, словарь."""

    dsn: str | None = None
    embedder = None
    vocabulary_path = None
    _vocab = None
    _vocab_mtime = None

    @classmethod
    def vocab(cls):
        """Словарь перечитывается, если файл поменяли, — сервер можно не перезапускать."""
        from ..graph.vocabulary import DEFAULT_PATH

        path = cls.vocabulary_path or DEFAULT_PATH
        mtime = path.stat().st_mtime if path.exists() else None
        if cls._vocab is None or mtime != cls._vocab_mtime:
            cls._vocab = load_vocabulary(path)
            cls._vocab_mtime = mtime
        return cls._vocab


def _hit_to_dict(hit) -> dict:
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
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

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
            elif route.path == "/api/concepts":
                self._concepts()
            elif route.path == "/api/concept":
                self._concept({k: v[0] for k, v in query.items() if v})
            elif route.path == "/api/network":
                self._network((query.get("checked_only") or [""])[0] in ("1", "true"))
            elif route.path == "/api/graph":
                self._graph((query.get("checked_only") or [""])[0] in ("1", "true"))
            elif route.path == "/api/entities":
                kind = (query.get("kind") or [""])[0] or None
                with db.connect(State.dsn) as conn:
                    self._json({
                        "counts": graph.counts(conn),
                        "entities": graph.top_entities(conn, kind, 300),
                    })
            elif route.path == "/api/entity":
                key = (query.get("key") or [""])[0]
                with db.connect(State.dsn) as conn:
                    self._json({"key": key, "documents": graph.entity_documents(conn, key)})
            elif route.path == "/api/document":
                doc_id = (query.get("doc_id") or [""])[0]
                with db.connect(State.dsn) as conn:
                    self._json({"doc_id": doc_id, "chunks": db.document_chunks(conn, doc_id)})
            else:
                self._json({"error": "нет такой страницы"}, 404)
        except Exception as exc:  # noqa: BLE001 — ошибка уходит на страницу, сервер живёт
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path != "/api/concept":
                self._json({"error": "нет такого адреса"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 64 * 1024:
                self._json({"error": "запрос слишком большой"}, 413)
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json({"api_version": graph_api.API_VERSION,
                            "error": "тело запроса — не JSON", "detail": str(exc)}, 400)
                return
            self._concept(payload)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- контракт с базой данных проекта ------------------------------------ #
    def _concepts(self) -> None:
        try:
            vocab = State.vocab()
            with db.connect(State.dsn) as conn:
                self._json(graph_api.concepts_response(conn, vocab))
        except VocabularyError as exc:
            self._json({"error": f"словарь: {exc}"}, 500)
        except graph_api.NotBuilt as exc:
            self._json({"api_version": graph_api.API_VERSION, "error": str(exc)}, 503)

    def _network(self, checked_only: bool) -> None:
        try:
            vocab = State.vocab()
            with db.connect(State.dsn) as conn:
                self._json(graph_api.network_response(conn, vocab, checked_only))
        except VocabularyError as exc:
            self._json({"error": f"словарь: {exc}"}, 500)
        except graph_api.NotBuilt as exc:
            self._json({"api_version": graph_api.API_VERSION, "error": str(exc)}, 503)

    def _graph(self, checked_only: bool) -> None:
        try:
            vocab = State.vocab()
            with db.connect(State.dsn) as conn:
                self._json(graph_api.graph_response(conn, vocab, checked_only))
        except VocabularyError as exc:
            self._json({"error": f"словарь: {exc}"}, 500)
        except graph_api.NotBuilt as exc:
            self._json({"api_version": graph_api.API_VERSION, "error": str(exc)}, 503)

    def _concept(self, payload: dict) -> None:
        try:
            vocab = State.vocab()
            request = graph_api.parse_request(vocab, payload)
            with db.connect(State.dsn) as conn:
                self._json(graph_api.concept_response(conn, vocab, request))
        except graph_api.RequestError as exc:
            self._json(exc.payload(), 400)
        except VocabularyError as exc:
            self._json({"error": f"словарь: {exc}"}, 500)
        except graph_api.NotBuilt as exc:
            self._json({"api_version": graph_api.API_VERSION, "error": str(exc)}, 503)

    # -- действия ---------------------------------------------------------- #
    def _stats(self) -> dict:
        with db.connect(State.dsn) as conn:
            info = db.stats(conn)
        lo, hi = info["years"]
        return {
            "documents": info["documents"],
            "chunks": info["chunks"],
            "year_from": lo,
            "year_to": hi,
        }

    def _search(self, query: dict) -> dict:
        text = (query.get("q") or [""])[0].strip()
        if not text:
            return {"hits": [], "query": ""}

        def number(name, default=None):
            raw = (query.get(name) or [""])[0]
            return int(raw) if raw.isdigit() else default

        def fraction(name, default=0.0):
            raw = (query.get(name) or [""])[0]
            try:
                return float(raw)
            except ValueError:
                return default

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
            )
        return {"query": text, "hits": [_hit_to_dict(h) for h in hits]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Веб-интерфейс к базе знаний ГеоRAG")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dsn", default=None, help="адрес базы, иначе GEORAG_DSN")
    parser.add_argument("--embedder", choices=["local", "ollama"], default="local")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = parser.parse_args(argv)

    State.dsn = args.dsn
    try:
        vocab = State.vocab()
        print(f"Словарь: {len(vocab.concepts)} понятий, {len(vocab.territories)} территорий, "
              f"{len(vocab.methods)} методов")
    except VocabularyError as exc:
        print(f"Словарь не читается: {exc}", file=sys.stderr)
        return 1

    # Проверяем базу до старта: лучше честно упасть здесь, чем показать пустую страницу.
    try:
        with db.connect(args.dsn) as conn:
            info = db.stats(conn)
            graph.init(conn)       # таблицы графа — один раз при старте, не на каждый запрос
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Поднимите её: docker compose up -d", file=sys.stderr)
        return 1

    print(f"В базе: {info['documents']} статей, {info['chunks']} чанков")
    print("Загружаю модель эмбеддингов (один раз на весь запуск)...")
    State.embedder = build_embedder(args.embedder, device=args.device)
    State.embedder.encode_one("прогрев")   # чтобы первый поиск не ждал загрузку весов

    address = f"http://localhost:{args.port}"
    print(f"\nГотово: {address}")
    print("Остановить — Ctrl+C\n")

    if not args.no_browser:
        import webbrowser

        webbrowser.open(address)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Остановлен.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
