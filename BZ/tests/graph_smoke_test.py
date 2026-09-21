"""Проверка графа: названия правилами, словарь, разметка понятий, контракт. Без базы.

Запуск:  python tests/graph_smoke_test.py

Тексты взяты из настоящих статей корпуса — на выдуманных предложениях правила
выглядят лучше, чем есть. База и модели подменяются: проверяется логика,
а не то, что лежит у вас в PostgreSQL.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.graph import api, concepts, store  # noqa: E402
from georag.graph import vocabulary as V  # noqa: E402
from georag.graph.extract import COMMODITY, METHOD, OBJECT, extract  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def names(text: str, kind: str | None = None) -> set[str]:
    return {e.name for e in extract(text) if kind is None or e.kind == kind}


# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        self.conn.executed.append((" ".join(sql.split()), params))

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
    def __init__(self, rows=None):
        self.executed: list[tuple] = []
        self.rows = list(rows or [])
        self.commits = 0

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


# --------------------------------------------------------------------------- #
def test_objects() -> None:
    print("\nНазвания объектов")
    found = names("В пределах Тырныаузского рудного узла выделены зоны окварцевания.")
    check("падеж снят, имя в именительном", "Тырныаузский рудный узел" in found, str(found))

    check("средний род от опорного слова",
          "Крайчиковское рудопроявление" in names("Крайчиковское рудопроявление изучено бурением."))
    check("женский род от опорного слова",
          "Зыгыркольская зона" in names("Зыгыркольская зона опробована по сети 100х20 м."))
    check("двойное имя через дефис",
          "Далдыно-Алакитский район" in names("Далдыно-Алакитский район наиболее изучен."))
    check("прилагательное между именем и словом",
          "Северо-Енисейский рудный район" in
          names("в пределах Северо-Енисейского рудного района"))

    # То, что ломало правила на живом тексте.
    junk = names("Золото приурочено к узлам пересечения разнонаправленных разломов.")
    check("строчные слова не становятся названиями", not junk & {"Пересечения разлом"}, str(junk))
    check("существительное в начале предложения не имя",
          "Металлогения массив" not in names("Металлогения массива изучена слабо."),
          str(names("Металлогения массива изучена слабо.")))
    check("общее прилагательное не имя",
          not names("Рудные узлы выделяются по совокупности признаков.", OBJECT),
          str(names("Рудные узлы выделяются по совокупности признаков.", OBJECT)))

    # Более точное опорное слово выигрывает у общего.
    trubka = names("Кимберлитовые трубки Далдыно-Алакитского района вскрыты бурением.")
    check("«кимберлитовые трубки» не порождают ложное имя",
          "Кимберлитовская трубка" not in trubka, str(trubka))
    check("район при этом найден", "Далдыно-Алакитский район" in trubka, str(trubka))


def test_commodities_and_methods() -> None:
    print("\nПолезные ископаемые и методы")
    text = ("Прогноз золоторудных месторождений выполнен по данным ASTER "
            "с применением метода главных компонент и машинного обучения. "
            "Оценена алмазоносность и содержания меди.")
    commodities = names(text, COMMODITY)
    check("золото найдено по «золоторудных»", "золото" in commodities, str(commodities))
    check("алмазы найдены по «алмазоносность»", "алмазы" in commodities)
    check("медь найдена", "медь" in commodities)

    methods = names(text, METHOD)
    check("ASTER найден", "ASTER" in methods, str(methods))
    check("метод главных компонент найден", "метод главных компонент" in methods)
    check("машинное обучение найдено", "машинное обучение" in methods)

    english = names("Machine learning for mineral prospectivity using Landsat-8 and gold deposits.")
    check("английский текст тоже разбирается",
          {"машинное обучение", "Landsat", "золото"} <= english, str(english))


def test_dedup() -> None:
    print("\nПовторы и ключи")
    twice = extract("Тырныаузского рудного узла ... Тырныаузский рудный узел ... золото, золота")
    check("повтор внутри куска схлопнут", len(twice) == 2, str([e.name for e in twice]))
    keys = {e.key for e in twice}
    check("ключи различают вид сущности", len(keys) == 2, str(keys))
    check("в ключе нет падежа",
          extract("Тырныаузскому рудному узлу")[0].key == extract("Тырныаузский рудный узел")[0].key)
    check("пустой текст не ломает", extract("") == [])


def test_queries() -> None:
    print("\nЗапросы по названиям")
    conn = _FakeConn(rows=[[("k", "Анабарский щит", "объект", 3, 21)]])
    rows = store.top_entities(conn, kind="объект")
    check("названия считаются по статьям", rows[0]["docs"] == 3, str(rows))
    sql = conn.executed[0][0]
    check("сортировка по числу статей", "ORDER BY docs DESC" in sql)
    check("фильтр по виду подставлен", "e.kind = %(kind)s" in sql)
    check("совместной встречаемости больше нет", not hasattr(store, "related"))


# --------------------------------------------------------------------------- #
#  Словарь
# --------------------------------------------------------------------------- #
def _vocab() -> V.Vocabulary:
    return V.load()


def test_vocabulary_file() -> None:
    print("\nСловарь vocabulary.yaml")
    vocab = _vocab()
    check("восемь понятий, как в kb.concept", len(vocab.concepts) == 8, str(list(vocab.concepts)))
    expected = {
        "структурный контроль", "гидротермальные изменения", "литологический контроль",
        "магматизм", "палеодолины и впадины", "плотностная и магнитная неоднородность",
        "рельеф и геоморфология", "обнажённость и растительный покров",
    }
    check("коды совпадают с kb.concept буква в букву", set(vocab.concepts) == expected,
          str(set(vocab.concepts) ^ expected))
    kinds = {c.kind for c in vocab.concepts.values()}
    check("два рода понятий", kinds == set(V.CONCEPT_KINDS), str(kinds))
    check("у каждого понятия есть определение",
          all(len(c.definition) > 40 for c in vocab.concepts.values()))
    check("три типа связей", set(vocab.relations) == {"описывает", "проявлено_на", "измеряет"})
    check("отпечаток словаря посчитан", len(vocab.sha) == 16)
    check("предупреждений нет", not vocab.warnings, str(vocab.warnings))


def test_vocabulary_errors() -> None:
    print("\nОшибки в словаре называются по месту")
    base = {"гидротермальные изменения": {
        "род": "рудоконтролирующий фактор",
        "определение": "Околорудная переработка пород горячими растворами.",
        "синонимы": ["окварцевание"]}}

    def error(data) -> str:
        try:
            V.parse(data)
        except V.VocabularyError as exc:
            return str(exc)
        return ""

    check("пустые понятия", "пуст" in error({"понятия": {}}))
    check("неизвестный род", "род" in error({"понятия": {"х": {**base["гидротермальные изменения"],
                                                                "род": "фактор"}}}))
    check("нет определения", "определения" in error(
        {"понятия": {"х": {"род": "условие наблюдения", "определение": ""}}}))
    check("связь, которую код не строит",
          "не умеет" in error({"понятия": base,
                               "связи": {"х": {"от": "территория", "к": "территория"}}}))
    check("неизвестный раздел", "неизвестные" in error({"понятия": base, "признаки": {}}))
    check("синоним строкой, а не списком, принимается",
          error({"понятия": {"х": {**base["гидротермальные изменения"], "синонимы": "метасоматоз"}}})
          == "")
    dup = V.parse({"понятия": {
        "а": {**base["гидротермальные изменения"], "синонимы": ["разлом"]},
        "б": {**base["гидротермальные изменения"], "синонимы": ["разлом"]}}})
    check("общий синоним — предупреждение, не ошибка", dup.warnings, str(dup.warnings))


def test_patterns() -> None:
    print("\nСинонимы находят слово в любом падеже")
    vocab = _vocab()
    hydro = vocab.concepts["гидротермальные изменения"].matcher
    check("окварцевание → «окварцеванием»",
          "окварцевание" in hydro.found("сопровождается окварцеванием пород"))
    check("метасоматоз → «метасоматоза»",
          "метасоматоз" in hydro.found("зоны калиевого метасоматоза"))
    check("английское во множественном числе",
          "alteration zone" in hydro.found("hydrothermal alteration zones were mapped"))
    struct = vocab.concepts["структурный контроль"].matcher
    check("фраза из двух слов с падежами",
          "пересечение разломов" in struct.found("к узлам пересечения разломов"))
    check("сложное слово не цепляется",
          not struct.found("процессы разломообразования не изучены"))
    cover = vocab.concepts["обнажённость и растительный покров"].matcher
    check("«ё» и «е» — одно", "обнажённость" in cover.found("низкая обнаженность территории"))
    relief = vocab.concepts["рельеф и геоморфология"].matcher
    check("аббревиатура с учётом регистра", relief.found("по ЦМР SRTM") and
          not relief.found("dem и srtm строчными"))
    tags = vocab.tags("В пределах Анабарского щита по данным ASTER и АГСМ")
    check("территория в родительном падеже", ("территория", "Анабарский щит") in tags, str(tags))
    check("метод по синониму", ("метод", "аэрогамма-спектрометрия") in tags, str(tags))
    check("поиск имени по синониму", vocab.territory("Anabar shield").name == "Анабарский щит")
    check("понятие без учёта «ё»",
          vocab.concept("обнаженность и растительный покров").code
          == "обнажённость и растительный покров")


# --------------------------------------------------------------------------- #
#  Предложения и цитаты
# --------------------------------------------------------------------------- #
def test_sentences() -> None:
    print("\nПредложения")
    text = ("Работы выполнены А. П. Степановым и др. в 2019 г. на площади. "
            "Выделены зоны окварцевания (см. рис. 3). Оруденение связано с разломами!")
    sentences = concepts.split_sentences(text)
    check("инициалы и сокращения не рвут предложение", len(sentences) == 3, str(sentences))
    check("пустой текст", concepts.split_sentences("") == [])


def test_quotes() -> None:
    print("\nЦитата сверяется с текстом")
    text = ("Для Хаптасыннахской зоны описаны две стадии:\nранний калиевый метасоматизм "
            "и позднее «окварцевание» с серицитизацией. Далее — о структуре.")
    found = concepts.locate_quote("ранний калиевый метасоматизм и позднее \"окварцевание\"", text)
    check("находит при других кавычках и переносе строки",
          found == "ранний калиевый метасоматизм и позднее «окварцевание»", str(found))
    found = concepts.locate_quote("РАННИЙ КАЛИЕВЫЙ МЕТАСОМАТИЗМ", text)
    check("регистр не мешает, а цитата возвращается как в статье",
          found == "ранний калиевый метасоматизм", str(found))
    found = concepts.locate_quote("описаны две стадии … с серицитизацией", text)
    check("сокращение многоточием: части по порядку",
          found is not None and found.startswith("описаны") and found.endswith("серицитизацией"),
          str(found))
    check("выдуманная цитата не проходит",
          concepts.locate_quote("описаны три стадии березитизации", text) is None)
    check("слишком короткая цитата не проходит", concepts.locate_quote("две", text) is None)
    check("части в обратном порядке не проходят",
          concepts.locate_quote("с серицитизацией … описаны две стадии", text) is None)

    vocab = _vocab()
    q = concepts.quote_by_terms(concepts.split_sentences(text),
                                vocab.concepts["гидротермальные изменения"].matcher)
    check("цитата — предложение с большинством слов словаря",
          q is not None and "серицитизацией" in q, str(q))


def test_judge() -> None:
    print("\nРешение по ответу модели")
    text = "Выделены зоны интенсивного окварцевания и серицитизации вдоль разломов."
    verdict, quote, _ = concepts.judge(
        {"описывает": True, "цитата": "зоны интенсивного окварцевания и серицитизации"}, text)
    check("подтверждение с цитатой", verdict == store.CONFIRMED and quote, str(verdict))
    verdict, _, _ = concepts.judge({"описывает": "да", "цитата": "зоны березитизации пород"}, text)
    check("«описывает», но цитаты нет в тексте — не подтверждение",
          verdict == store.UNCHECKED, verdict)
    verdict, _, reason = concepts.judge({"описывает": False, "причина": "упомянуто мимоходом"}, text)
    check("отказ с причиной", verdict == store.REJECTED and "мимоходом" in reason)


class _ScriptedLLM:
    model = "qwen3:14b"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def chat_json(self, system, user):
        self.prompts.append(user)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_verify() -> None:
    print("\nПроверка моделью")
    text = "Выделены зоны интенсивного окварцевания и серицитизации вдоль разломов."
    rows = [(1, "гидротермальные изменения", text),
            (2, "гидротермальные изменения", text),
            (3, "гидротермальные изменения", text)]
    llm = _ScriptedLLM([
        {"описывает": True, "цитата": "зоны интенсивного окварцевания и серицитизации"},
        {"описывает": True, "цитата": "выдуманная цитата про березиты"},   # переспросить
        {"описывает": True, "цитата": "снова выдумка про березиты и пиритизацию"},
        {"описывает": False, "причина": "не о том"},
    ])
    conn = _FakeConn(rows=[rows])
    stats = concepts.verify(conn, llm, _vocab(), log=lambda *a: None)
    check("все три просмотрены", stats["checked"] == 3, str(stats))
    check("одно подтверждено", stats["confirmed"] == 1, str(stats))
    check("выдуманная дважды цитата отклонена", stats["no_quote"] == 1, str(stats))
    check("переспрос содержит замечание", "не найдена" in llm.prompts[2])
    updates = [p for s, p in conn.executed if s.startswith("UPDATE evidence")]
    check("решение записано по каждому", len(updates) == 3, str(len(updates)))
    check("фиксация после каждого решения", conn.commits == 3, str(conn.commits))

    dead = _ScriptedLLM([ConnectionError("нет Ollama")] * 5)
    conn = _FakeConn(rows=[rows])
    stats = concepts.verify(conn, dead, _vocab(), log=lambda *a: None)
    check("молчащая модель — остановка, решения не приняты",
          stats["stopped"] and stats["checked"] == 0, str(stats))
    check("ничего не записано", not any(s.startswith("UPDATE") for s, _ in conn.executed))


# --------------------------------------------------------------------------- #
#  Кандидаты шага 1
# --------------------------------------------------------------------------- #
class _Embedder:
    name = "проба"

    def encode(self, texts):
        # Вектор «про изменения», если в тексте есть «окварц», иначе ортогональный.
        return [[1.0, 0.0] if "окварц" in t.lower() or "гидротерм" in t.lower() else [0.0, 1.0]
                for t in texts]


def test_candidates() -> None:
    print("\nКандидаты: вектор, словарь, оба")
    vocab = V.parse({"понятия": {"гидротермальные изменения": {
        "род": "рудоконтролирующий фактор",
        "определение": "Околорудная переработка пород горячими растворами, окварцевание.",
        "синонимы": ["окварцевание", "разлом"]}}})
    chunks = [
        (1, "d1", "Интенсивное окварцевание вдоль контакта. Вторая фраза."),
        (2, "d1", "Без слов словаря здесь. Гидротермальный процесс описан подробно."),
        (3, "d2", "Разлом северо-западного простирания. Ничего о растворах."),
        (4, "d2", "Разлом, и совсем мимо."),
    ]
    # Похожесть на определение: 1 — близко, 2 — близко, 3 — умеренно, 4 — мимо.
    conn = _FakeConn(rows=[[(1, 0.71), (2, 0.60), (3, 0.47)]])
    found, report = concepts.find_candidates(conn, _Embedder(), vocab, chunks,
                                             log=lambda *a: None)
    by = {c.chunk_id: c for c in found}
    check("близко и со словом — «оба»", by[1].found_by == "оба", str(by.get(1)))
    check("близко без слов — «вектор»", by[2].found_by == "вектор", str(by.get(2)))
    check("слово при умеренной близости — «словарь»", by[3].found_by == "словарь",
          str(by.get(3)))
    check("слово без близости не проходит", 4 not in by)
    check("цитата — предложение со словом", by[1].quote.startswith("Интенсивное окварцевание"),
          by[1].quote)
    check("без слов словаря цитата — предложение, ближайшее к определению",
          by[2].quote == "Гидротермальный процесс описан подробно.", by[2].quote)
    check("сводка по понятию", report["гидротермальные изменения"]["total"] == 3, str(report))

    # Много кандидатов без слов словаря: цитаты подбираются пачками, и ни одна
    # не должна съехать на чужой фрагмент на границе пачки.
    many = [(100 + i, f"d{i}", " ".join(f"Фраза номер {k} без смысла." for k in range(9))
             + f" Гидротермальный признак {i}.") for i in range(70)]
    conn = _FakeConn(rows=[[(cid, 0.9) for cid, _, _ in many]])
    found, _ = concepts.find_candidates(conn, _Embedder(), vocab, many, log=lambda *a: None)
    check("пачки по 512 предложений не перепутали цитаты",
          len(found) == 70 and all(c.quote == f"Гидротермальный признак {c.chunk_id - 100}."
                                   for c in found), str([c.quote for c in found[:3]]))
    check("порог передан в запрос",
          any("1 - (embedding <=> %(q)s::vector) >= %(floor)s" in s for s, _ in conn.executed))


# --------------------------------------------------------------------------- #
#  Контракт с базой данных проекта
# --------------------------------------------------------------------------- #
def test_request() -> None:
    print("\nЗапрос по шаблону")
    vocab = _vocab()
    req = api.parse_request(vocab, {"concept": "гидротермальные изменения",
                                    "territory": "Billyakh", "method": "cokriging",
                                    "limit": "7", "checked_only": "true"})
    check("имена приведены к словарю",
          req["territory"] == "Билляхская зона" and req["method"] == "кокригинг", str(req))
    check("число из строки", req["limit"] == 7)
    check("флаг из строки", req["checked_only"] is True)
    check("умолчания", api.parse_request(vocab, {"concept": "магматизм"})["limit"] == 5)

    def error(payload) -> api.RequestError | None:
        try:
            api.parse_request(vocab, payload)
        except api.RequestError as exc:
            return exc
        return None

    exc = error({"concept": "рудный фактор"})
    check("неизвестное понятие — с перечнем известных", exc and len(exc.known) == 8)
    check("без понятия", error({}) is not None)
    check("неизвестная территория", error({"concept": "магматизм", "territory": "Луна"}))
    check("лимит за пределами", error({"concept": "магматизм", "limit": 1000}))
    check("лишнее поле — ошибка, а не молчание", error({"concept": "магматизм", "q": "х"}))
    check("чужая версия контракта", error({"concept": "магматизм", "api_version": 2}))
    payload = error({"concept": "х"}).payload()
    check("ошибка отдаётся с версией", payload["api_version"] == 1 and payload["known"])


def test_not_built() -> None:
    print("\nОтвет, пока разметка не построена")
    conn = _FakeConn(rows=[[]])   # graph_meta пуста
    try:
        api.concepts_response(conn, _vocab())
        ok = False
    except api.NotBuilt:
        ok = True
    check("«не построено» вместо пустого списка", ok)


def test_graph() -> None:
    print("\nГраф связей для картинки")
    rows = [
        [("built_at", "2026-09-21T10:00:00+00:00"), ("vocabulary_sha", _vocab().sha)],  # meta
        [(12,)], [(40, 38)], [(1990, 2024)],                                           # stats
        [("гидротермальные изменения", "территория", "Билляхская зона", 5, 3),
         ("гидротермальные изменения", "метод", "ASTER", 2, 2),
         ("выброшенное понятие", "метод", "ASTER", 1, 1)],                               # рёбра
        [("гидротермальные изменения", 9, 4)],                                          # понятия
        [("территория", "Билляхская зона", 3), ("метод", "ASTER", 2)],                  # узлы
    ]
    data = api.graph_response(_FakeConn(rows=rows), _vocab())
    nodes = {n["id"]: n for n in data["nodes"]}
    vocab = _vocab()
    expected = len(vocab.concepts) + len(vocab.territories) + len(vocab.methods)
    check("все узлы словаря, даже пустые", len(nodes) == expected, f"{len(nodes)} из {expected}")
    check("вес понятия — статьи", nodes["понятие:гидротермальные изменения"]["documents"] == 4)
    check("пустое понятие с нулём", nodes["понятие:магматизм"]["documents"] == 0)
    types = {(e["term"], e["type"]) for e in data["edges"]}
    check("территория — «проявлено на», метод — «изучается методом»",
          types == {("территория:Билляхская зона", "проявлено_на"), ("метод:ASTER", "измеряет")},
          str(types))
    check("понятие не из словаря в граф не попадает", len(data["edges"]) == 2)
    check("подписи связей из словаря",
          {e["label"] for e in data["edges"]} == {"проявлено на", "изучается методом"})


def test_network() -> None:
    print("\nСеть знаний (как в Obsidian)")
    rows = [
        [("built_at", "2026-09-21T10:00:00+00:00"), ("vocabulary_sha", _vocab().sha)],
        [(2,)], [(10, 10)], [(2019, 2021)],
        [("d1", "Статья один", 2021, "https://x/1", "10.1/1"),
         ("d2", "Статья два", 2019, "https://x/2", None),
         ("d3", "Статья без связей", 2018, "https://x/3", None)],               # статьи
        [("d1", "гидротермальные изменения", 3, True),
         ("d2", "гидротермальные изменения", 1, False)],                        # описывает
        [("d1", "территория", "Анабарский щит", 2), ("d2", "метод", "ASTER", 1)],  # метки
        [("d1", "объект:анабарск:щит", "Анабарский щит", "объект", 5),
         ("d2", "объект:тырныауз:рудный-узел", "Тырныаузский рудный узел", "объект", 1),
         ("d2", "ископаемое:золото", "золото", "ископаемое", 2)],                 # названия
    ]
    data = api.network_response(_FakeConn(rows=rows), _vocab())
    nodes = {n["id"]: n for n in data["nodes"]}
    kinds = {n["kind"] for n in data["nodes"]}
    check("шесть видов узлов", kinds == {"статья", "понятие", "территория", "метод",
                                         "объект", "ископаемое"}, str(kinds))
    check("статья без связей тоже узел", "статья:d3" in nodes)
    check("«Анабарский щит» правил слит с территорией словаря",
          "название:объект:анабарск:щит" not in nodes
          and nodes["территория:Анабарский щит"]["degree"] == 1, str(nodes.get("территория:Анабарский щит")))
    pair = [e for e in data["edges"] if e["target"] == "территория:Анабарский щит"]
    check("дубль ребра схлопнут, вес — наибольший", len(pair) == 1 and pair[0]["weight"] == 5,
          str(pair))
    described = [e for e in data["edges"] if e["type"] == "описывает"]
    check("рёбра «описывает» с пометкой проверки",
          len(described) == 2 and {e["confirmed"] for e in described} == {True, False})
    check("все рёбра идут от статьи", all(e["source"].startswith("статья:") for e in data["edges"]))
    check("степень понятия — число статей", nodes["понятие:гидротермальные изменения"]["degree"] == 2)
    check("ссылка DOI у статьи", nodes["статья:d1"]["doi_url"] == "https://doi.org/10.1/1")


def main() -> int:
    test_objects()
    test_commodities_and_methods()
    test_dedup()
    test_queries()
    test_vocabulary_file()
    test_vocabulary_errors()
    test_patterns()
    test_sentences()
    test_quotes()
    test_judge()
    test_verify()
    test_candidates()
    test_request()
    test_not_built()
    test_graph()
    test_network()

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
