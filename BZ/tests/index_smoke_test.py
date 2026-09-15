"""Проверка базы знаний без самой базы и без модели эмбеддингов.

Запуск:  python tests/index_smoke_test.py

Postgres здесь не нужен: подключение и модель заменены заглушками. Проверяется то,
что ломается тихо, — формат вектора, схема, раскладка чанка по колонкам,
арифметика RRF, порядок выдачи и пропуск уже проиндексированного.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.index import db  # noqa: E402
from georag.index.ingest import (  # noqa: E402
    _chunk_rows,
    _document_row,
    ingest_dir,
    is_reference_chunk,
)
from georag.index.embed import OllamaEmbedder, build_embedder  # noqa: E402
from georag.index.search import Hit, cap_per_doc, hybrid_search, rrf_merge  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        self.conn.executed.append((" ".join(sql.split()), params))
        self.conn.last_sql = sql

    def fetchall(self):
        return self.conn.rows.pop(0) if self.conn.rows else []

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    """Соединение-заглушка: запоминает SQL и отдаёт заранее заданные строки."""

    def __init__(self, rows=None):
        self.executed: list[tuple] = []
        self.rows = list(rows or [])
        self.commits = 0
        self.rollbacks = 0
        self.last_sql = ""

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _FakeEmbedder:
    """Вместо BGE-M3: вектор из трёх чисел, детерминированный по длине текста."""

    dim = 3

    def __init__(self):
        self.calls = 0

    def encode(self, texts, progress=False):
        self.calls += 1
        return [[len(t) / 100.0, 0.5, 0.25] for t in texts]

    def encode_one(self, text):
        return self.encode([text])[0]


def _row(chunk_id: int, doc_id: str = "d1", title: str = "Статья") -> tuple:
    #  id, doc_id, ord, text, headings, pages, title, year, journal, url, authors, score
    return (
        chunk_id, doc_id, chunk_id, f"текст {chunk_id}", ["Введение"], [chunk_id],
        title, 2024, "Руды и металлы", "https://example.org/a", ["Иванов И.И."], 0.5,
    )


# --------------------------------------------------------------------------- #
def test_vector_literal() -> None:
    print("\nФормат вектора для pgvector")
    check("скобки и запятые", db.vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]",
          db.vector_literal([0.1, 0.2, 0.3]))
    check("целые тоже проходят", db.vector_literal([1, 2]) == "[1,2]")
    check("пустой вектор не ломает", db.vector_literal([]) == "[]")


def test_schema() -> None:
    print("\nСхема базы")
    sql = db.SCHEMA % {"dim": db.VECTOR_DIM}
    check("размерность подставлена", "vector(1024)" in sql)
    check("расширение включается", "CREATE EXTENSION IF NOT EXISTS vector" in sql)
    check("полнотекст по-русски", "to_tsvector('russian'" in sql)
    check("чанк привязан к документу", "REFERENCES documents(doc_id) ON DELETE CASCADE" in sql)
    check("повторная нарезка не плодит дубли", "UNIQUE (doc_id, ord)" in sql)
    check("векторный индекс косинусный", "vector_cosine_ops" in db.HNSW)


def test_rrf() -> None:
    print("\nАрифметика RRF")
    scores = rrf_merge([[1, 2, 3], [3, 1]], k=60)
    check("первое место в двух списках весит больше",
          scores[1] > scores[3] > scores[2], str(scores))
    expected = 1 / 61 + 1 / 62
    check("формула 1/(k+rank)", abs(scores[1] - expected) < 1e-12, f"{scores[1]} vs {expected}")
    check("документ из одного списка тоже учтён", scores[2] == 1 / 62)
    check("пустой вход не ломает", rrf_merge([]) == {})


def test_hybrid_order() -> None:
    print("\nГибридный поиск: слияние двух списков")
    conn = _FakeConn(rows=[
        [_row(1), _row(2), _row(3)],   # вектор
        [_row(3), _row(4)],            # полнотекст
    ])
    hits = hybrid_search(conn, _FakeEmbedder(), "рудные узлы", limit=10, candidates=50)

    check("вернулись все найденные", len(hits) == 4, str(len(hits)))
    check("первым — найденный обоими", hits[0].chunk_id == 3, str(hits[0].chunk_id))
    check("подписано, кто нашёл", hits[0].found_by == "оба", hits[0].found_by)
    only_fts = [h for h in hits if h.chunk_id == 4][0]
    check("одиночка из полнотекста помечен", only_fts.found_by == "текст", only_fts.found_by)
    check("места из обоих списков сохранены", hits[0].vec_rank == 3 and hits[0].fts_rank == 1)
    check("оценки убывают", all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)))

    sql = " ".join(s for s, _ in conn.executed)
    check("вектор ищется косинусом", "<=>" in sql)
    check("полнотекст через websearch_to_tsquery", "websearch_to_tsquery('russian'" in sql)

    conn = _FakeConn(rows=[[_row(1)], [_row(2)]])
    hybrid_search(conn, _FakeEmbedder(), "x", year_from=2010, source="openalex")
    sql = " ".join(s for s, _ in conn.executed)
    check("фильтр по году добавлен", "d.year >= %(year_from)s" in sql)
    check("фильтр по источнику добавлен", "d.source = %(source)s" in sql)


def test_row_mapping() -> None:
    print("\nРаскладка по колонкам")
    meta = {
        "doc_id": "10.3390_min13050669", "title": "Prospectivity Mapping",
        "authors": ["Kai Zhou"], "year": 2023, "journal": "Minerals",
        "doi": "10.3390/min13050669", "url": "https://doi.org/10.3390/min13050669",
        "source": "mdpi", "status": "acquired", "pages": 20, "accuracy": 1.0,
    }
    row = _document_row(meta)
    check("год прочитан", row["year"] == 2023)
    check("точность стала числом", isinstance(row["accuracy"], float))
    check("отсутствующие поля не роняют", _document_row({})["title"] == "")

    chunks = [{"index": 0, "text": "видимый текст", "embed_text": "Введение\nвидимый текст",
               "headings": ["Введение"], "pages": [1, 2], "n_tokens": 42, "has_table": True}]
    rows = _chunk_rows("d1", chunks, [[0.1, 0.2, 0.3]])
    check("в базу идёт оригинал, не embed_text", rows[0]["text"] == "видимый текст")
    check("страницы стали числами", rows[0]["pages"] == [1, 2])
    check("вектор сериализован", rows[0]["embedding"] == "[0.1,0.2,0.3]")
    check("таблица отмечена", rows[0]["has_table"] is True)


def test_ingest() -> None:
    print("\nЗагрузка папки добычи")
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "d1.json").write_text(json.dumps(
            {"doc_id": "d1", "title": "Первая", "year": 2020, "status": "acquired"}
        ), encoding="utf-8")
        (tmp / "d1.chunks.json").write_text(json.dumps([
            {"index": 0, "text": "раз", "embed_text": "раз"},
            {"index": 1, "text": "два", "embed_text": "два"},
        ]), encoding="utf-8")
        # Документ без чанков: статья нашлась, но полного текста не было.
        (tmp / "d2.json").write_text(json.dumps({"doc_id": "d2", "title": "Вторая"}), encoding="utf-8")
        (tmp / "d2.chunks.json").write_text("[]", encoding="utf-8")

        conn = _FakeConn(rows=[[]])  # indexed_docs → пусто
        embedder = _FakeEmbedder()
        report = ingest_dir(conn, embedder, tmp, verbose=False)

        check("просмотрены оба документа", report.seen == 2, str(report.seen))
        check("проиндексирован один", report.indexed == 1, str(report.indexed))
        check("пустой посчитан отдельно", report.empty == 1)
        check("чанки записаны", report.chunks == 2, str(report.chunks))
        check("ошибок нет", not report.errors, "; ".join(report.errors))
        check("модель звали один раз на документ", embedder.calls == 1, str(embedder.calls))

        # Повторный запуск: число чанков совпало — документ пропускается.
        conn = _FakeConn(rows=[[("d1", 2)]])
        again = ingest_dir(conn, _FakeEmbedder(), tmp, verbose=False)
        check("уже проиндексированный пропущен", again.skipped == 1, str(again.skipped))
        check("заново ничего не писали", again.indexed == 0)

        # --force: переиндексируем, даже если совпало.
        conn = _FakeConn(rows=[[("d1", 2)]])
        forced = ingest_dir(conn, _FakeEmbedder(), tmp, force=True, verbose=False)
        check("--force переиндексирует", forced.indexed == 1)

        # Битый файл не должен ронять весь прогон.
        (tmp / "d3.chunks.json").write_text("{не json", encoding="utf-8")
        conn = _FakeConn(rows=[[]])
        broken = ingest_dir(conn, _FakeEmbedder(), tmp, verbose=False)
        check("битый файл только в ошибках", len(broken.errors) == 1 and broken.indexed == 1,
              "; ".join(broken.errors))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reference_chunks() -> None:
    print("\nОтсев списков литературы")
    check("заголовок References распознан",
          is_reference_chunk({"headings": ["References"], "text": "Smith J. 2020."}))
    check("русский заголовок распознан",
          is_reference_chunk({"headings": ["Введение", "Список литературы"], "text": "Иванов И."}))
    check("похожий заголовок не путается",
          not is_reference_chunk({"headings": ["Обзор литературы по району"],
                                  "text": "В районе выделены рудные узлы. " * 20}))

    refs = {
        "headings": ["РЕЗУЛЬТАТЫ"],
        "text": (
            "Pour A. B., Hashim M. Targeting hydrothermal alterations // International "
            "Archives. 2017. pp. 153-157. https://doi.org/10.5194/isprs-1 "
            "Safari M. Landsat-8 data in Iran // Remote Sensing. 2018. pp. 11-20. "
            "https://doi.org/10.5194/isprs-2 Hewson R. D. Seamless geological map // "
            "Remote Sensing of Environment. 2005. pp. 159-172. doi:10.1016/j.rse.2005 "
        ),
    }
    check("библиография без заголовка распознана по плотности ссылок", is_reference_chunk(refs))

    normal = {
        "headings": ["Результаты"],
        "text": ("В пределах Тырныаузского рудного узла выделены зоны окварцевания, "
                 "приуроченные к узлам пересечения разломов. " * 6),
    }
    check("обычный текст не отброшен", not is_reference_chunk(normal))
    check("короткий текст не отброшен", not is_reference_chunk({"text": "doi.org http://x"}))


def test_cap_per_doc() -> None:
    print("\nНе больше N кусков одной статьи")
    hits = [Hit(chunk_id=i, doc_id=doc, ord=i, text="t")
            for i, doc in enumerate(["a", "a", "a", "b", "a", "c"])]
    capped = cap_per_doc(hits, 2)
    top = [h.doc_id for h in capped[:4]]
    check("одна статья не занимает всю выдачу", top == ["a", "a", "b", "c"], str(top))
    check("лишние куски не потеряны, а сдвинуты в конец", len(capped) == len(hits))
    check("ноль означает без ограничения",
          [h.doc_id for h in cap_per_doc(hits, 0)] == [h.doc_id for h in hits])


class _FakeHttpResponse:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload or {}
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload


def _install_fake_ollama(handler) -> list:
    """Подменяет requests.post; возвращает список вызовов для проверки."""
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url, json))
        return handler(url, json or {})

    module = types.ModuleType("requests")
    module.post = post
    sys.modules["requests"] = module
    return calls


def test_ollama_embedder() -> None:
    print("\nВекторы через Ollama")

    def ok(url, payload):
        texts = payload.get("input") or []
        # Ненормированные векторы: Ollama так и отдаёт, нормируем мы сами.
        return _FakeHttpResponse({"embeddings": [[3.0] + [0.0] * 1023 for _ in texts]})

    calls = _install_fake_ollama(ok)
    emb = OllamaEmbedder(batch_size=2)
    vectors = emb.encode(["раз", "два", "три"])

    check("вернулось по вектору на текст", len(vectors) == 3, str(len(vectors)))
    check("размерность 1024", len(vectors[0]) == 1024, str(len(vectors[0])))
    length = math.sqrt(sum(v * v for v in vectors[0]))
    check("вектор нормирован", abs(length - 1.0) < 1e-9, str(length))
    check("тексты уходят пачками", len(calls) == 2, f"вызовов {len(calls)}")
    check("зовём /api/embed", calls[0][0].endswith("/api/embed"), calls[0][0])

    # Старая Ollama: /api/embed нет, есть /api/embeddings по одному тексту.
    def old(url, payload):
        if url.endswith("/api/embed"):
            return _FakeHttpResponse(status=404, text="not found")
        return _FakeHttpResponse({"embedding": [1.0] + [0.0] * 1023})

    calls = _install_fake_ollama(old)
    vectors = OllamaEmbedder(batch_size=4).encode(["раз", "два"])
    check("откат на старый endpoint сработал", len(vectors) == 2, str(len(vectors)))
    check("пошли по одному тексту",
          sum(1 for url, _ in calls if url.endswith("/api/embeddings")) == 2, str(calls))

    # Не та модель: размерность не та — это надо заметить сразу, а не после индексации.
    def wrong(url, payload):
        return _FakeHttpResponse({"embeddings": [[0.1] * 384]})

    _install_fake_ollama(wrong)
    try:
        OllamaEmbedder().encode(["раз"])
        check("чужая модель замечена", False, "ошибки не было")
    except RuntimeError as exc:
        check("чужая модель замечена", "1024" in str(exc), str(exc)[:80])

    check("пустой список не ходит в сеть", OllamaEmbedder().encode([]) == [])


def test_embedder_fallback() -> None:
    print("\nOllama молчит — переходим на локальную модель")

    def dead(url, payload):
        raise OSError("connection refused")

    _install_fake_ollama(dead)
    picked = build_embedder("ollama", quiet=True)
    check("выбрана локальная модель", picked.name == "local", picked.name)

    def alive(url, payload):
        texts = payload.get("input") or [payload.get("prompt")]
        return _FakeHttpResponse({"embeddings": [[1.0] + [0.0] * 1023 for _ in texts]})

    _install_fake_ollama(alive)
    picked = build_embedder("ollama", quiet=True)
    check("Ollama жива — берём её", picked.name == "ollama", picked.name)
    check("local выбирается явно", build_embedder("local").name == "local")


def test_citation() -> None:
    print("\nСсылка для цитирования")
    hit = Hit(chunk_id=1, doc_id="d1", ord=0, text="t", pages=[4, 5],
              title="Рудные узлы Анабарского щита", year=2021, url="https://example.org/a")
    line = hit.citation()
    check("есть год", "2021" in line)
    check("есть страницы", "с. 4–5" in line, line)
    check("есть ссылка", "https://example.org/a" in line)


def main() -> int:
    test_vector_literal()
    test_schema()
    test_rrf()
    test_hybrid_order()
    test_row_mapping()
    test_ingest()
    test_reference_chunks()
    test_cap_per_doc()
    test_ollama_embedder()
    test_embedder_fallback()
    test_citation()

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
