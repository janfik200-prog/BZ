"""Проверка извлечения сущностей и запросов графа. Без базы.

Запуск:  python tests/graph_smoke_test.py

Тексты взяты из настоящих статей корпуса — на выдуманных предложениях правила
выглядят лучше, чем есть.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.graph import store  # noqa: E402
from georag.graph.cli import build  # noqa: E402
from georag.graph.extract import COMMODITY, METHOD, OBJECT, extract  # noqa: E402
from georag.graph.llm_extract import extract_with_llm, parse_response  # noqa: E402

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


def test_build() -> None:
    print("\nПостроение графа")
    conn = _FakeConn(rows=[
        [  # chunks_for_graph
            (1, "d1", "В пределах Тырныаузского рудного узла выявлено золото."),
            (2, "d1", "Применён метод главных компонент по данным ASTER."),
            (3, "d2", "Анабарский щит изучен слабо; оценена алмазоносность."),
        ],
        [("объект", 3), ("ископаемое", 2), ("метод", 2)],   # counts by kind
        [(9,)],                                              # mentions
        [(0,)],                                              # relations
    ])
    info = build(conn, verbose=False)
    check("просмотрены все фрагменты", info["chunks"] == 3, str(info))
    check("статьи посчитаны", info["documents"] == 2, str(info))

    sql = " ".join(s for s, _ in conn.executed)
    check("граф чистится перед сборкой", "DELETE FROM mentions" in sql and "DELETE FROM entities" in sql)
    check("сущности пишутся без дублей", "ON CONFLICT (key) DO NOTHING" in sql)
    inserted = [p for s, p in conn.executed if s.startswith("INSERT INTO entities")]
    written = {p[1] for p in inserted}
    check("найден узел", "Тырныаузский рудный узел" in written, str(written))
    check("найден щит", "Анабарский щит" in written, str(written))
    check("упоминание привязано к статье и фрагменту",
          any(s.startswith("INSERT INTO mentions") and p[1] == "d1" and p[2] == 1
              for s, p in conn.executed))


def test_llm_pass() -> None:
    print("\nРазбор моделью поверх правил")
    answer = {
        "entities": [
            {"name": "трубка Удачная", "kind": "объект"},
            {"name": "Далдыно-Алакитский район", "kind": "объект"},
            {"name": "алмазы", "kind": "ископаемое"},
            {"name": "", "kind": "объект"},
            {"name": "нечто", "kind": "выдумка"},
        ],
        "relations": [
            {"from": "трубка Удачная", "to": "Далдыно-Алакитский район", "type": "находится в"},
            {"from": "трубка Удачная", "to": "алмазы", "type": "содержит"},
            {"from": "трубка Удачная", "to": "Мирный", "type": "рядом с"},
            {"from": "трубка Удачная", "to": "трубка Удачная", "type": "равно"},
            "мусор",
        ],
    }
    entities, relations = parse_response(answer)
    found = {e.name for e in entities}
    check("имя с обратным порядком слов взято", "трубка Удачная" in found, str(found))
    check("пустое имя отброшено", "" not in found)
    check("неизвестный вид отброшен", "нечто" not in found)
    check("связи названы", {r.type for r in relations} == {"находится в", "содержит"},
          str([r.type for r in relations]))
    check("связь на невыписанную сущность отброшена",
          all("мирн" not in r.to_key for r in relations))
    check("связь с самой собой отброшена", all(r.from_key != r.to_key for r in relations))
    check("не-словарь в списке не роняет разбор", len(relations) == 2)

    # Ключи общие с правилами — значит найденное обоими способами схлопнется.
    from georag.graph.extract import _key
    check("ключ считается так же, как у правил",
          [e for e in entities if e.name == "алмазы"][0].key == _key("алмазы", COMMODITY))

    class _Dead:
        def chat_json(self, system, user):
            raise ConnectionError("Ollama не отвечает")

    check("молчащая модель не роняет сборку", extract_with_llm(_Dead(), "текст") == ([], []))


def test_build_with_llm() -> None:
    print("\nСборка графа с моделью")

    class _Model:
        def chat_json(self, system, user):
            return {
                "entities": [{"name": "трубка Удачная", "kind": "объект"},
                             {"name": "алмазы", "kind": "ископаемое"}],
                "relations": [{"from": "трубка Удачная", "to": "алмазы", "type": "содержит"}],
            }

    conn = _FakeConn(rows=[
        [(1, "d1", "Далдыно-Алакитский район изучен бурением.")],
        [("объект", 2), ("ископаемое", 1)],
        [(3,)],
        [(1,)],
    ])
    info = build(conn, verbose=False, llm=_Model())
    sql = " ".join(s for s, _ in conn.executed)
    check("связь записана", "INSERT INTO relations" in sql)
    check("модель дополнила правила, а не заменила",
          {p[1] for s, p in conn.executed if s.startswith("INSERT INTO entities")}
          >= {"Далдыно-Алакитский район", "трубка Удачная"},
          str({p[1] for s, p in conn.executed if s.startswith("INSERT INTO entities")}))
    check("число связей попало в отчёт", info.get("relations") == 1, str(info))


def test_queries() -> None:
    print("\nЗапросы графа")
    conn = _FakeConn(rows=[[("k", "Анабарский щит", "объект", 3, 21)]])
    rows = store.top_entities(conn, kind="объект")
    check("сущности считаются по статьям", rows[0]["docs"] == 3, str(rows))
    sql = conn.executed[0][0]
    check("сортировка по числу статей", "ORDER BY docs DESC" in sql)
    check("фильтр по виду подставлен", "e.kind = %(kind)s" in sql)

    conn = _FakeConn(rows=[[("k2", "золото", "ископаемое", 4)]])
    rel = store.related(conn, "k")
    check("связь — это общая статья", "other.doc_id = mine.doc_id" in conn.executed[0][0])
    check("сама с собой не связывается", "other.entity_key <> mine.entity_key" in conn.executed[0][0])
    check("возвращает число общих статей", rel[0]["together"] == 4, str(rel))


def main() -> int:
    test_objects()
    test_commodities_and_methods()
    test_dedup()
    test_build()
    test_llm_pass()
    test_build_with_llm()
    test_queries()

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
