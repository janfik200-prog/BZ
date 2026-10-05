"""Прогон добычи без сети, без Ollama и без моделей docling.

Запуск:  python tests/acquire_smoke_test.py [путь_к.pdf]

Проверяем: сборку аннотации из инвертированного индекса OpenAlex, разбор ответа API,
отсечение HTML, выданного вместо PDF, срезание блока размышлений Qwen3, и главное —
полный проход добычи, где статья разбирается из памяти и PDF на диск не попадает.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.acquire import models as am  # noqa: E402
from georag.acquire.fetch import FetchResult, fetch_pdf  # noqa: E402
from georag.acquire.llm import _extract_json  # noqa: E402
from georag.acquire.providers.openalex import (  # noqa: E402
    OpenAlexProvider,
    abstract_from_inverted_index,
)
from georag.parse.config import Settings  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def test_abstract_index() -> None:
    print("\nАннотация из инвертированного индекса")
    index = {"Выделение": [0], "рудных": [1, 5], "узлов": [2, 6], "по": [3], "плотности": [4]}
    text = abstract_from_inverted_index(index)
    check(
        "слова расставлены по позициям",
        text == "Выделение рудных узлов по плотности рудных узлов",
        text,
    )
    check("пустой индекс не ломает", abstract_from_inverted_index(None) == "")


def test_openalex_parsing() -> None:
    print("\nРазбор ответа OpenAlex")
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123456789",
                "doi": "https://doi.org/10.24411/0869-7175-2020-10003",
                "title": "Опыт прогнозирования перспективных площадей",
                "publication_year": 2020,
                "language": "ru",
                "type": "article",
                "authorships": [
                    {"author": {"display_name": "Корчагина Д.А."}},
                    {"author": {"display_name": "Агибалов О.А."}},
                ],
                "abstract_inverted_index": {"рудно-россыпных": [0], "узлов": [1]},
                "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://x/1.pdf"},
                "best_oa_location": {
                    "pdf_url": "https://example.org/article.pdf",
                    "landing_page_url": "https://example.org/article",
                    "license": "cc-by",
                    "source": {"display_name": "Отечественная геология"},
                },
                "primary_location": {},
            }
        ]
    }
    _install_fake_requests(get_json=payload)
    provider = OpenAlexProvider("openalex", {"mailto": "a@b.c", "min_year": 2000})
    cands = provider.search("рудные узлы", 10)

    check("кандидат получен", len(cands) == 1)
    cand = cands[0]
    check("DOI без префикса", cand.doi == "10.24411/0869-7175-2020-10003", str(cand.doi))
    check("doc_id пригоден для имени файла", "/" not in cand.doc_id, cand.doc_id)
    check("аннотация собрана", cand.abstract == "рудно-россыпных узлов", cand.abstract)
    check("журнал вытащен", cand.journal == "Отечественная геология")
    check("ссылка на PDF найдена", cand.pdf_url == "https://example.org/article.pdf")
    check("ссылка для человека найдена", cand.url == "https://example.org/article")
    check("авторы собраны", cand.authors == ["Корчагина Д.А.", "Агибалов О.А."])
    check("фильтры собраны", "from_publication_date:2000-01-01" in provider._filters())


def test_crossref_parsing() -> None:
    print("\nРазбор ответа Crossref (MDPI)")
    from georag.acquire.providers.crossref import (
        CrossrefProvider,
        abstract_from_jats,
        pdf_urls_from_work,
    )

    check(
        "JATS-теги сняты с аннотации",
        abstract_from_jats(
            "<jats:p>Abstract: Mineral <jats:italic>prospectivity</jats:italic> mapping.</jats:p>"
        )
        == "Mineral prospectivity mapping.",
        abstract_from_jats(
            "<jats:p>Abstract: Mineral <jats:italic>prospectivity</jats:italic> mapping.</jats:p>"
        ),
    )
    check("пустая аннотация не ломает", abstract_from_jats(None) == "")

    check(
        "PDF берётся из link по content-type",
        pdf_urls_from_work(
            {
                "link": [
                    {"URL": "https://www.mdpi.com/x/1/2/3/pdf", "content-type": "application/pdf"}
                ]
            }
        )[0]
        == "https://www.mdpi.com/x/1/2/3/pdf",
    )
    check(
        "для mdpi ссылка достраивается суффиксом /pdf",
        "https://www.mdpi.com/2075-163X/13/5/669/pdf"
        in pdf_urls_from_work(
            {
                "link": [
                    {"URL": "https://www.mdpi.com/2075-163X/13/5/669", "content-type": "text/html"}
                ]
            }
        ),
    )
    check("без ссылок список пустой", pdf_urls_from_work({}) == [])

    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.3390/min13050669",
                    "title": ["Prospectivity Mapping of Tungsten Mineralization"],
                    "abstract": "<jats:p>We delineate exploration targets.</jats:p>",
                    "author": [
                        {"given": "Kai", "family": "Zhou"},
                        {"given": "Tao", "family": "Sun"},
                    ],
                    "issued": {"date-parts": [[2023, 5, 10]]},
                    "container-title": ["Minerals"],
                    "publisher": "MDPI AG",
                    "type": "journal-article",
                    "URL": "https://doi.org/10.3390/min13050669",
                    "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
                    "link": [
                        {
                            "URL": "https://www.mdpi.com/2075-163X/13/5/669/pdf",
                            "content-type": "application/pdf",
                            "intended-application": "text-mining",
                        }
                    ],
                }
            ]
        }
    }
    _install_fake_requests(get_json=payload)
    provider = CrossrefProvider("mdpi", {"member": 1968, "min_year": 2000, "mailto": "a@b.c"})
    cands = provider.search("ore cluster delineation", 10)

    check("кандидат получен", len(cands) == 1)
    cand = cands[0] if cands else None
    check(
        "издатель отфильтрован по member", "member:1968" in provider._filters(), provider._filters()
    )
    check("тип работ ограничен", "type:journal-article" in provider._filters())
    check("DOI прочитан", bool(cand and cand.doi == "10.3390/min13050669"))
    check("журнал прочитан", bool(cand and cand.journal == "Minerals"))
    check(
        "аннотация как текст", bool(cand and cand.abstract == "We delineate exploration targets.")
    )
    check("авторы собраны", bool(cand and cand.authors == ["Kai Zhou", "Tao Sun"]))
    check("год прочитан", bool(cand and cand.year == 2023))
    check("ссылка на PDF найдена", bool(cand and cand.pdf_url.endswith("/669/pdf")))
    check("лицензия прочитана", bool(cand and cand.license and "creativecommons" in cand.license))
    check(
        "doc_id тот же, что у OpenAlex для той же статьи — дубли схлопнутся",
        bool(
            cand
            and cand.doc_id
            == am.Candidate(
                source="openalex", external_id="W9", title="x", doi="10.3390/min13050669"
            ).doc_id
        ),
    )


def test_fetch_rejects_html() -> None:
    print("\nОтсечение не-PDF")
    _install_fake_requests(get_body=b"<!DOCTYPE html><html><head><title>article</title>")
    result = fetch_pdf("https://example.org/looks-like.pdf")
    check("html вместо PDF отклонён", not result.ok)
    check("причина названа", "не PDF" in result.error, result.error)

    _install_fake_requests(get_body=b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n")
    ok = fetch_pdf("https://example.org/real.pdf")
    check("настоящий PDF принят", ok.ok)
    check("sha256 посчитан", len(ok.sha256) == 64)
    check("размер посчитан", ok.bytes_len > 0)


def test_fetch_follows_citation_meta() -> None:
    print("\nСтраница статьи вместо PDF: идём по citation_pdf_url")
    page = (
        b"<!DOCTYPE html><html><head><title>Article</title>"
        b'<meta name="citation_pdf_url" content="https://example.org/a/1/pdf">'
        b"</head><body>text</body></html>"
    )
    _install_fake_requests(
        bodies={
            "https://example.org/a/1": page,
            "https://example.org/a/1/pdf": b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n",
        }
    )
    result = fetch_pdf("https://example.org/a/1")
    check("PDF найден через мета-тег", result.ok, result.error)
    check("взят адрес из мета-тега", result.url.endswith("/pdf"), result.url)

    _install_fake_requests(bodies={"https://example.org/a/2": b"<html><body>no meta</body></html>"})
    miss = fetch_pdf("https://example.org/a/2")
    check("без мета-тега честно признаём неудачу", not miss.ok)
    check("причина названа", "не PDF" in miss.error, miss.error)


def test_think_stripping() -> None:
    print("\nБлок размышлений Qwen3")
    raw = '<think>Надо вернуть JSON...</think>\n{"queries": ["рудные узлы", "ore clusters"]}'
    data = _extract_json(raw)
    check("JSON достаётся из-под <think>", data["queries"] == ["рудные узлы", "ore clusters"])
    raw2 = 'Вот ответ: {"decisions": [{"i": 1, "relevant": true}]} — готово'
    check("JSON достаётся из текста", _extract_json(raw2)["decisions"][0]["i"] == 1)


# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, json_data=None, body=b"", status=200, url=None):
        self._json = json_data
        self._body = body
        self.status_code = status
        self.url = url or "https://example.org/final.pdf"
        self.headers = {"Content-Type": "application/pdf"}

    def close(self):
        return None

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install_fake_requests(get_json=None, get_body=b"", bodies: dict | None = None) -> None:
    """Заглушка сети. bodies — ответ по каждому адресу отдельно, для многошаговых загрузок."""

    def respond(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if bodies is not None:
            return _FakeResponse(json_data=get_json, body=bodies.get(url, b""), url=url)
        return _FakeResponse(json_data=get_json, body=get_body, url=url)

    module = types.ModuleType("requests")
    module.get = respond
    module.post = lambda *a, **kw: _FakeResponse(json_data={})

    class _FakeSession:
        get = staticmethod(respond)

    module.Session = _FakeSession
    sys.modules["requests"] = module


# --------------------------------------------------------------------------- #
class _AcceptAll:
    """Заглушка отбора для сквозного теста: тема одним запросом, всё принимается."""

    model = "тест"

    def queries(self, topic, n=5):
        return [topic]

    def filter_batch(self, topic, batch):
        for cand in batch:
            cand.relevant = True
            cand.reason = "тест"


class _FakeProvider:
    """Источник без сети: отдаёт заранее заданные статьи."""

    name = "fake"

    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = 0

    def search(self, query, limit):
        self.calls += 1
        return [am.Candidate(**{**c, "query": query}) for c in self._candidates][:limit]


def test_acquire_end_to_end(pdf: Path) -> None:
    print(f"\nДобыча целиком (PDF в памяти): {pdf.name}")
    from georag.acquire import pipeline as ap

    out = Path("/tmp/georag-acquire-smoke")
    shutil.rmtree(out, ignore_errors=True)  # чисто с нуля, иначе индекс прошлого прогона мешает
    settings = Settings(
        out_dir=out / "parsed",
        log_dir=out / "logs",
        golden_dir=out / "golden",
        manual_dir=out / "manual",
        doc_timeout_sec=60,
    )
    cfg = ap.AcquireConfig(
        sources_path=out / "sources.json",
        out_dir=out,
        queries=1,
        per_query=10,
        max_docs=5,
        make_chunks=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    cfg.sources_path.write_text(
        json.dumps(
            {"sources": [{"name": "fake", "mode": "api", "provider": "fake", "enabled": True}]}
        ),
        encoding="utf-8",
    )

    candidates = [
        {
            "source": "fake",
            "external_id": "W1",
            "title": "Выделение рудных узлов по геофизическим данным",
            "abstract": "Рассмотрено выделение рудных узлов.",
            "doi": "10.1234/test-1",
            "year": 2024,
            "journal": "Тестовая геология",
            "landing_page_url": "https://example.org/w1",
            "pdf_urls": ["https://example.org/w1.pdf"],
            "is_oa": True,
        },
        {
            "source": "fake",
            "external_id": "W2",
            "title": "Статья без открытого текста",
            "abstract": "Только метаданные.",
            "doi": "10.1234/test-2",
            "year": 2023,
            "landing_page_url": "https://example.org/w2",
            "pdf_urls": [],
        },
        {  # дубль первой: должен схлопнуться внутри прогона
            "source": "fake",
            "external_id": "W1",
            "title": "Выделение рудных узлов по геофизическим данным",
            "doi": "10.1234/test-1",
            "landing_page_url": "https://example.org/w1",
            "pdf_urls": ["https://example.org/w1.pdf"],
        },
    ]
    provider = _FakeProvider(candidates)

    pdf_bytes = pdf.read_bytes()
    real_build, real_fetch = ap.build_providers, ap.fetch_pdf
    ap.build_providers = lambda sources: ([provider], [])
    ap.fetch_pdf = lambda url, **kw: FetchResult(
        ok=True,
        url=url,
        final_url=url,
        data=pdf_bytes,
        sha256="a" * 64,
        bytes_len=len(pdf_bytes),
        content_type="application/pdf",
        fetched_at="2026-09-12T00:00:00+00:00",
    )
    try:
        report = ap.acquire("выделение рудных узлов", _AcceptAll(), settings, cfg)
        second = ap.acquire("выделение рудных узлов", _AcceptAll(), settings, cfg)
    finally:
        ap.build_providers, ap.fetch_pdf = real_build, real_fetch

    check("дубль внутри прогона схлопнулся", report.after_dedup == 2, f"{report.after_dedup}")
    acquired = report.acquired
    check("статья разобрана из памяти", len(acquired) == 1, f"{[r.status for r in report.records]}")
    check(
        "статья без PDF помечена отдельно", any(r.status == am.NO_FULLTEXT for r in report.records)
    )
    check(
        "у пробного прогона свой статус, не путается с «нет текста»",
        am.NOT_FETCHED != am.NO_FULLTEXT,
    )

    rec = acquired[0] if acquired else None
    check("ссылка сохранена вместо файла", bool(rec and rec.url == "https://example.org/w1"))
    check("sha256 записан", bool(rec and len(rec.sha256) == 64))
    check("страницы посчитаны", bool(rec and rec.pages > 0), f"{rec.pages if rec else 0}")
    check(
        "парсер отработал по потоку байт",
        bool(rec and rec.parser == "pymupdf4llm"),
        rec.parser if rec else "",
    )

    pdfs_on_disk = list(out.rglob("*.pdf"))
    check("PDF на диск не попал", not pdfs_on_disk, f"{[p.name for p in pdfs_on_disk]}")
    check("текст сохранён", (out / f"{rec.doc_id}.md").exists() if rec else False)
    check("запись сохранена", (out / f"{rec.doc_id}.json").exists() if rec else False)
    check("индекс дописан", (out / "index.jsonl").exists())
    check("повторный прогон видит дубли", second.already_known >= 1, f"{second.already_known}")
    check("повторный прогон ничего не скачивал заново", len(second.acquired) == 0)


# --------------------------------------------------------------------------- #
def test_heuristic_filter() -> None:
    print("\nОтбор без модели")
    from georag.acquire.heuristic import HeuristicFilter, translate_topic  # noqa: E402

    topic = "выделение рудных узлов"
    check(
        "тема переводится по глоссарию",
        translate_topic(topic) == 'delineation "ore cluster"',
        str(translate_topic(topic)),
    )
    queries = HeuristicFilter().queries(topic, 5)
    check("запросы включают русский и английский", len(queries) >= 2, "; ".join(queries))
    free_form = "прогноз золотого оруденения на Анабарском щите по космоснимкам"
    english = translate_topic(free_form)
    check(
        "свободная формулировка переводится без смеси языков",
        bool(english) and not re.search(r"[а-я]", english),
        str(english),
    )
    check(
        "непереводимая тема не даёт мусорного запроса",
        translate_topic("стратиграфия кембрия") is None,
    )

    batch = [
        am.Candidate(
            source="s",
            external_id="1",
            title="Выделение рудных узлов по плотности линеаментов",
            abstract="Рассмотрены рудные узлы Анабарского щита.",
        ),
        am.Candidate(
            source="s",
            external_id="2",
            title="Ore cluster delineation using machine learning",
            abstract="We delineate ore clusters from geophysical data.",
        ),
        am.Candidate(
            source="s",
            external_id="3",
            title="Влияние удобрений на урожайность пшеницы",
            abstract="Агротехнические приёмы в севообороте.",
        ),
        am.Candidate(
            source="s", external_id="4", title="Геология рудного узла в пределах щита", abstract=""
        ),
    ]
    HeuristicFilter().filter_batch(topic, batch)
    check("русская статья по теме принята", batch[0].relevant is True, batch[0].reason)
    check("для статьи без аннотации порог мягче", batch[3].relevant is True, batch[3].reason)
    check(
        "в причине видно, что судили по названию",
        "только по названию" in batch[3].reason,
        batch[3].reason,
    )

    # И наоборот: пустая аннотация не должна пропускать что попало.
    off_topic_no_abstract = [
        am.Candidate(
            source="s",
            external_id="9",
            title="Клинические рекомендации по артериальной гипертензии",
            abstract="",
        ),
    ]
    HeuristicFilter().filter_batch(topic, off_topic_no_abstract)
    check(
        "без аннотации мусор всё равно отклонён",
        off_topic_no_abstract[0].relevant is False,
        off_topic_no_abstract[0].reason,
    )
    check("английская статья по теме принята", batch[1].relevant is True, batch[1].reason)
    check("статья мимо темы отклонена", batch[2].relevant is False, batch[2].reason)
    check("решение принято по каждой статье", all(c.relevant is not None for c in batch))


def test_domain_gate() -> None:
    print("\nДоменный фильтр: мусор из других наук")
    from georag.acquire.heuristic import HeuristicFilter, has_domain_term

    check("руда распознана", has_domain_term("выделение рудного узла"))
    check("склонение не мешает", has_domain_term("золоторудных месторождений"))
    check("английский распознан", has_domain_term("ore district delineation"))
    check("маркетинг — не наша тема", not has_domain_term("прогноз развития маркетинга в вузах"))
    check(
        "звёздные скопления — не наша тема",
        not has_domain_term("K-Means cluster analysis of mid-infrared absorption"),
    )

    flt = HeuristicFilter()

    def verdict(topic, title, abstract=""):
        cand = am.Candidate(source="t", external_id="1", title=title, abstract=abstract)
        flt.filter_batch(topic, [cand])
        return cand

    junk = verdict(
        "линеаментный анализ прогноз оруденения",
        "Прогноз развития маркетинговой парадигмы в высшем образовании",
        "Анализ развития маркетинга в вузах.",
    )
    check("статья из другой науки отклонена", junk.relevant is False)
    check("причина понятна", "рудной тематики" in junk.reason, junk.reason)

    good = verdict(
        "выделение рудных узлов",
        "Delineation of new mineral prospects using ZY1-02D",
        "We delineate mineral prospects and ore districts.",
    )
    check("статья по теме принята", good.relevant is True, good.reason)

    # Буквального совпадения терминов нет, но статья явно рудная и без аннотации:
    # такие раньше отсеивались в ноль, теперь проходят по доменной прибавке.
    near = verdict("прогноз рудных узлов", "Features of metallogenic zoning of Northern Vietnam")
    check("рудная статья без аннотации проходит", near.relevant is True, near.reason)

    # А вот с аннотацией и без единого термина темы — уже нет: доказательств хватает.
    off = verdict(
        "выделение рудных узлов",
        "Экономика горного предприятия",
        "Рассматривается рентабельность рудника и экономика предприятия.",
    )
    check("рудная по слову, но мимо темы — отклонена", off.relevant is False, off.reason)

    generic = verdict(
        "геофизические методы выделение перспективных площадей",
        "ПАЛЕОГЕОГРАФИЯ ТЮМЕНСКОЙ СВИТЫ",
        "Особенности палеогеографии, методы выделения площадей.",
    )
    check("совпадение по общим словам не спасает", generic.relevant is False, generic.reason)


def test_llm_fallback() -> None:
    print("\nПодстраховка эвристикой, когда модель молчит")
    from georag.acquire.heuristic import HeuristicFilter
    from georag.acquire.llm import filter_candidates

    class _DeadLLM:
        model = "qwen3:14b"

        def filter_batch(self, topic, batch):
            raise ConnectionError("Ollama не отвечает")

    batch = [
        am.Candidate(
            source="s",
            external_id="1",
            title="Прогноз рудных узлов Урала",
            abstract="Выделены рудные узлы.",
        ),
        am.Candidate(
            source="s",
            external_id="2",
            title="Кардиология и диета",
            abstract="Клиническое исследование.",
        ),
    ]
    errors = filter_candidates(
        _DeadLLM(), "выделение рудных узлов", batch, fallback=HeuristicFilter()
    )
    check("ошибка модели зафиксирована", len(errors) == 1, "; ".join(errors))
    check("никто не остался без решения", all(c.relevant is not None for c in batch))
    check("статья по теме всё равно принята", batch[0].relevant is True)
    check("мусор всё равно отклонён", batch[1].relevant is False)
    check("в записи видно, кто решил", batch[0].filtered_by == "эвристика", batch[0].filtered_by)

    # Без подстраховки статьи должны уйти человеку — это поведение тоже фиксируем.
    batch2 = [am.Candidate(source="s", external_id="3", title="Прогноз рудных узлов", abstract="")]
    filter_candidates(_DeadLLM(), "рудные узлы", batch2, fallback=None)
    check("без подстраховки статья ждёт человека", batch2[0].relevant is None)


def test_multiple_fulltext_urls() -> None:
    print("\nНесколько копий полного текста")
    from georag.acquire import pipeline as ap
    from georag.acquire.providers.crossref import pdf_urls_from_work

    cand = am.Candidate(
        source="s",
        external_id="1",
        title="t",
        pdf_urls=["https://publisher.example/blocked.pdf", "https://repo.example/copy.pdf"],
    )
    check(
        "адреса идут по порядку без повторов",
        cand.fetch_urls
        == ["https://publisher.example/blocked.pdf", "https://repo.example/copy.pdf"],
    )

    work = {
        "link": [
            {"URL": "https://www.mdpi.com/a/1/2/3", "content-type": "text/html"},
            {"URL": "https://repo.example/x.pdf", "content-type": "application/pdf"},
        ]
    }
    urls = pdf_urls_from_work(work)
    check("PDF из link идёт первым", urls[0] == "https://repo.example/x.pdf", str(urls))
    check(
        "достроенный адрес mdpi тоже в списке",
        "https://www.mdpi.com/a/1/2/3/pdf" in urls,
        str(urls),
    )

    # Первый адрес отдаёт заглушку, второй — файл: статья должна дойти до разбора.
    calls: list[str] = []

    def _fake_fetch(url, **kw):
        calls.append(url)
        if "blocked" in url:
            return FetchResult(ok=False, url=url, error="по ссылке не PDF (html-страница)")
        return FetchResult(
            ok=True,
            url=url,
            final_url=url,
            data=b"%PDF-1.4 fake",
            sha256="b" * 64,
            bytes_len=13,
            content_type="application/pdf",
            fetched_at="2026-09-12T00:00:00+00:00",
        )

    class _StubWorker:
        def parse(self, *a, **kw):
            raise AssertionError("не должно вызываться")

    real_fetch = ap.fetch_pdf
    real_process = ap.process_document
    ap.fetch_pdf = _fake_fetch
    ap.process_document = lambda inp, s, w, lg, **kw: (
        type(
            "R",
            (),
            {
                "status": "ok",
                "parser": "stub",
                "accuracy": 1.0,
                "pages": 1,
                "sections": 0,
                "tables": 0,
                "duration_sec": 0.1,
                "failed_checks": [],
                "attempts": ["stub"],
            },
        )(),
        type("P", (), {"text": "текст", "docling_doc": None})(),
        [],
    )
    try:
        out = Path("/tmp/georag-multiurl")
        shutil.rmtree(out, ignore_errors=True)
        cfg = ap.AcquireConfig(out_dir=out, make_chunks=False)
        logger = ap.StepLogger(out / "logs", prefix="test")
        record = ap._acquire_one(cand, Settings(log_dir=out / "logs"), cfg, _StubWorker(), logger)
    finally:
        ap.fetch_pdf = real_fetch
        ap.process_document = real_process

    check("второй адрес спас статью", record.status == am.ACQUIRED, record.status)
    check("оба адреса записаны в журнал", len(record.tried_urls) == 2, str(record.tried_urls))
    check("порядок попыток соблюдён", calls[0].endswith("blocked.pdf"))


def test_topics_file() -> None:
    print("\nСписок тем")
    from georag.acquire.cli import load_topics

    topics = load_topics(Path("config/topics.yaml"))
    check("темы читаются", len(topics) >= 5, f"{len(topics)} тем")
    check("есть русская тема", any("рудных узлов" in t for t in topics))
    check("есть английская тема", any("prospectivity" in t or "cluster" in t for t in topics))


def test_prepared_queries() -> None:
    print("\nЗапросы отдельно от поиска")
    from georag.acquire.cli import load_queries, save_queries
    from georag.acquire.pipeline import AcquireConfig, _queries_for

    path = Path("/tmp/georag-queries.yaml")
    prepared = {"выделение рудных узлов": ["рудные узлы прогноз", "ore cluster delineation"]}
    save_queries(path, prepared)
    check("файл запросов записан и читается", load_queries(path) == prepared)
    check(
        "в файле есть пояснение для человека",
        path.read_text(encoding="utf-8").lstrip().startswith("#"),
    )

    class _ExplodingLLM:
        model = "не должна вызываться"

        def queries(self, topic, n):
            raise AssertionError("модель не должна дёргаться, когда есть готовые запросы")

    class _Log:
        def __init__(self):
            self.details = []

        def log(self, *a, **kw):
            self.details.append(kw.get("detail", ""))

    logger = _Log()
    cfg = AcquireConfig(queries_map=prepared)
    got = _queries_for("выделение рудных узлов", _ExplodingLLM(), None, cfg, logger)
    check("готовые запросы берутся из файла", got == prepared["выделение рудных узлов"])
    check("в логе видно, что модель не дёргали", any("готовые" in d for d in logger.details))

    # А для темы, которой в файле нет, модель всё же спрашивается.
    class _Model:
        model = "stub"

        def queries(self, topic, n):
            return [f"запрос про {topic}"]

    got2 = _queries_for("другая тема", _Model(), None, cfg, _Log())
    check("для новой темы запросы просят у модели", got2 == ["запрос про другая тема"])


def test_json_api_provider() -> None:
    print("\nИсточник, описанный в каталоге, без кода")
    from georag.acquire.providers.json_api import JsonApiProvider, pick

    item = {
        "id": "P1",
        "title": ["Ore cluster delineation"],
        "externalIds": {"DOI": "https://doi.org/10.1234/abc"},
        "authors": [{"name": "Ivan Petrov"}, {"name": "Jane Doe"}],
        "openAccessPdf": {},
        "links": {"pdf": "https://repo.example/p1.pdf"},
    }
    check("простой путь", pick(item, "id") == "P1")
    check("вложенный путь", pick(item, "externalIds.DOI") == "https://doi.org/10.1234/abc")
    check("список объектов", pick(item, "authors[].name") == ["Ivan Petrov", "Jane Doe"])
    check(
        "альтернативные пути через |",
        pick(item, "openAccessPdf.url|links.pdf") == "https://repo.example/p1.pdf",
    )
    check("нет значения — None", pick(item, "nothing.here") is None)

    params = {
        "url": "https://api.example.org/search",
        "query_param": "query",
        "limit_param": "limit",
        "items_path": "data",
        "map": {
            "external_id": "id",
            "title": "title",
            "abstract": "abstract",
            "year": "year",
            "journal": "venue",
            "doi": "externalIds.DOI",
            "authors": "authors[].name",
            "pdf_url": "openAccessPdf.url|links.pdf",
        },
    }
    _install_fake_requests(
        get_json={"data": [dict(item, abstract="Про рудные узлы.", year=2024, venue="Minerals")]}
    )
    cands = JsonApiProvider("example", params).search("ore cluster", 5)
    check("кандидат собран по описанию", len(cands) == 1)
    cand = cands[0] if cands else None
    check("название вытащено из списка", bool(cand and cand.title == "Ore cluster delineation"))
    check(
        "DOI очищен от префикса",
        bool(cand and cand.doi == "10.1234/abc"),
        str(cand.doi if cand else ""),
    )
    check("авторы собраны", bool(cand and cand.authors == ["Ivan Petrov", "Jane Doe"]))
    check("год приведён к числу", bool(cand and cand.year == 2024))
    check(
        "ссылка на текст найдена по второму пути",
        bool(cand and cand.pdf_url == "https://repo.example/p1.pdf"),
    )
    check(
        "doc_id считается от DOI",
        bool(cand and cand.doc_id == "10.1234_abc"),
        cand.doc_id if cand else "",
    )

    # Ошибка в каталоге должна быть понятной, а не падать где-то в разборе ответа.
    try:
        JsonApiProvider("bad", {"url": "https://x", "map": {"titl": "title"}})
        check("опечатка в map замечена", False)
    except ValueError as exc:
        check("опечатка в map замечена", "titl" in str(exc), str(exc)[:80])
    try:
        JsonApiProvider("bad", {})
        check("отсутствующий url замечен", False)
    except ValueError:
        check("отсутствующий url замечен", True)


def test_real_catalog() -> None:
    print("\nКаталог источников проекта")
    from georag.acquire.sources import load_sources

    sources = load_sources(Path("config/sources.yaml"))
    by_name = {s.name: s for s in sources}
    check("каталог читается", len(sources) >= 2, f"{len(sources)} источников")
    enabled = [s.name for s in sources if s.automatable]
    check("включены три источника", enabled == ["openalex", "crossref", "mdpi"], str(enabled))
    check(
        "mdpi ищется через crossref с фильтром издателя",
        by_name["mdpi"].provider == "crossref" and by_name["mdpi"].params.get("member") == 1968,
    )
    check(
        "шаблон json_api лежит в каталоге выключенным",
        by_name["example_json_api"].provider == "json_api"
        and not by_name["example_json_api"].enabled,
    )
    check("lens помечен как ручной", by_name["lens"].mode == "manual")
    check("ручные источники в автоматику не идут", not by_name["lens"].automatable)


def test_questions_and_key() -> None:
    print("\nВопрос вместо темы и ключ OpenAlex")
    import os

    from georag.acquire.heuristic import HeuristicFilter, has_domain_term
    from georag.acquire.providers import openalex

    q = HeuristicFilter().queries(
        "Какие признаки на основе гравиметрии использовались в моделях перспективности?"
    )
    joined = " | ".join(q)
    check("вопрос целиком поисковику не уходит", all("?" not in x for x in q), joined)
    check(
        "вопросительных слов в запросах нет",
        not any(w in joined.lower() for w in ("какие", "использовались", "основе")),
        joined,
    )
    check(
        "есть английский запрос по глоссарию",
        any("gravity" in x and "prospectivity" in x for x in q),
        joined,
    )
    check(
        "ДЗЗ переводится",
        any(
            "remote sensing" in x
            for x in HeuristicFilter().queries(
                "Какие признаки ДЗЗ использовались в моделях перспективности?"
            )
        ),
    )
    check(
        "тема (не вопрос) ищется как есть",
        HeuristicFilter().queries("выделение рудных узлов")[0] == "выделение рудных узлов",
    )
    check(
        "«труд» — не «руда»",
        not has_domain_term("Признаки прекарности рабочего места и трудовые отношения"),
    )
    check("руда по-прежнему руда", has_domain_term("Руды и металлы: состав рудного тела"))

    old = os.environ.pop("GEORAG_OPENALEX_KEY", None)
    try:
        os.environ["GEORAG_OPENALEX_KEY"] = "k123"
        prov = openalex.OpenAlexProvider(params={})
        check(
            "ключ берётся из окружения и уходит в запрос",
            prov._params("gold", 5).get("api_key") == "k123",
        )
        os.environ.pop("GEORAG_OPENALEX_KEY")
        prov = openalex.OpenAlexProvider(params={"pause_sec": 0})
        if not openalex.KEY_FILE.exists():
            module = types.ModuleType("requests")
            module.get = lambda *a, **kw: _FakeResponse(json_data={}, status=409)
            sys.modules["requests"] = module
            try:
                prov.search("gold", 5)
                check("без ключа — понятная ошибка", False)
            except openalex.OpenAlexKeyError as exc:
                check(
                    "без ключа — понятная ошибка, где взять ключ",
                    "openalex.org/settings/api" in str(exc),
                )
            finally:
                sys.modules.pop("requests", None)
    finally:
        if old is not None:
            os.environ["GEORAG_OPENALEX_KEY"] = old


def test_filter_rules_and_gate() -> None:
    print("\nОтбор моделью: правила и доменный фильтр до модели")
    import tempfile

    from georag.acquire import pipeline as P
    from georag.acquire.llm import FILTER_SYSTEM, FILTER_USER, QUERY_SYSTEM, QUERY_USER
    from georag.acquire.models import Candidate
    from georag.parse.config import Settings

    check(
        "модель знает, о чём база",
        "прогноза рудных месторождений" in FILTER_SYSTEM
        and "прогноза рудных месторождений" in QUERY_SYSTEM,
    )
    check("запросы — с привязкой к рудной тематике", "привязка к рудной тематике" in QUERY_USER)
    check(
        "отбор: не нужно совпадение всех слов, чужой смысл — мимо",
        "Не нужно,\n   чтобы совпали все слова" in FILTER_USER and "GRACE" in FILTER_USER,
    )

    cands = [
        Candidate(
            source="x",
            external_id="1",
            title="Gravity data in mineral prospectivity mapping",
            abstract="Gravity anomalies as evidential layers for gold deposits.",
        ),
        Candidate(
            source="x",
            external_id="2",
            title="Monitoring groundwater with GRACE gravimetry",
            abstract="Satellite gravimetry of aquifers.",
        ),
        Candidate(
            source="x",
            external_id="3",
            title="Extravascular lung water measurement",
            abstract="Clinical study.",
        ),
    ]

    class Prov:
        name = "проба"

        def search(self, q, n):
            return list(cands)

    class Seen:
        model = "проба"

        def __init__(self):
            self.titles = []

        def queries(self, topic, n=5):
            return [topic]

        def filter_batch(self, topic, batch):
            for c in batch:
                self.titles.append(c.title)
                c.relevant = True
                c.reason = "проба"

    real = (P.load_sources, P.build_providers)
    tmp = Path(tempfile.mkdtemp())
    try:
        P.load_sources = lambda path: []
        P.build_providers = lambda sources: ([Prov()], [])
        llm = Seen()
        cfg = P.AcquireConfig(out_dir=tmp / "acq", dry_run=True, force=True)
        report = P.acquire("гравиметрия прогноз", llm, Settings(log_dir=tmp / "logs"), cfg)
        check(
            "до модели дошла только рудная статья", llm.titles == [cands[0].title], str(llm.titles)
        )
        check(
            "отброшенные фильтром — в отчёте как мимо темы",
            report.rejected == 2,
            str(report.rejected),
        )
    finally:
        P.load_sources, P.build_providers = real
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_abstract_index()
    test_openalex_parsing()
    test_crossref_parsing()
    test_fetch_rejects_html()
    test_fetch_follows_citation_meta()
    test_think_stripping()
    test_heuristic_filter()
    test_domain_gate()
    test_llm_fallback()
    test_multiple_fulltext_urls()
    test_topics_file()
    test_prepared_queries()
    test_json_api_provider()
    test_real_catalog()
    test_questions_and_key()
    test_filter_rules_and_gate()

    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if pdf_arg and Path(pdf_arg).exists():
        sys.modules.pop("requests", None)  # дальше нужен настоящий импорт, если понадобится
        test_acquire_end_to_end(Path(pdf_arg))
    else:
        print("\n(PDF не передан — сквозной прогон добычи пропущен)")

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
