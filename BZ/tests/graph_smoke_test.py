"""Проверка графа знаний: факты из статей, словарь синонимов, датасеты. Без базы и модели.

Запуск:  python tests/graph_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.graph import api, facts  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.text import locate_quote, mentions, name_key  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        self.conn.executed.append((" ".join(sql.split()), params))

    def executemany(self, sql, rows):
        self.conn.executed.append((" ".join(sql.split()), list(rows)))

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


def row(i, src, rel, dst, doc="d1", quote="цитата", chunk=None):
    return {"id": i, "chunk_id": chunk or i, "doc_id": doc, "src": src, "relation": rel,
            "dst": dst, "quote": quote, "title": f"Статья {doc}", "year": 2021, "url": "",
            "doi": "10.1/" + doc}


# --------------------------------------------------------------------------- #
def test_text() -> None:
    print("\nИмена и цитаты")
    check("падеж не важен: «Донецкого бассейна» = «Донецкий бассейн»",
          name_key("Донецкого бассейна") == name_key("Донецкий бассейн"))
    check("беглая гласная: «рудного узла» = «рудный узел»",
          name_key("рудного узла") == name_key("рудный узел"))
    check("английское число: basins = basin", name_key("Donetsk basins") == name_key("Donetsk basin"))
    check("разные языки ключ не сводит — это делает словарь",
          name_key("Donetsk basin") != name_key("Донецкий бассейн"))
    check("упоминание в другом падеже", mentions("Анабарский щит", "в пределах Анабарского щита"))
    check("пропущено одно общее слово — упоминание",
          mentions("Leimengou porphyry system", "The Leimengou porphyry Mo system is hosted"))
    check("имя собственное пропускать нельзя: «Онежский рудный район» ≠ «рудного района»",
          not mentions("Онежский рудный район", "В пределах рудного района"))
    check("короткое имя — только целиком", not mentions("Билляхская зона", "зона разломов"))
    q = locate_quote("оруденение приурочено к «узлам» пересечения разломов",
                     "Здесь Оруденение приурочено к “узлам” пересечения   разломов. Далее.")
    check("цитата находится при других кавычках и пробелах, возвращается как в статье",
          q == "Оруденение приурочено к “узлам” пересечения разломов", str(q))
    check("выдуманная цитата не находится",
          locate_quote("золото связано с дайками гранитоидов", "Текст про другое, совсем другое.") is None)


def test_synonyms() -> None:
    print("\nСловарь синонимов")
    syn = S.parse({"имена": {"Донецкий бассейн": ["Donetsk basin", "Донбасс"],
                             "метод главных компонент": ["PCA", "principal component analysis"],
                             "Анабарский щит": None},
                   "связи": {"приурочено к": ["приурочен к", "associated with"]}})
    check("синоним → каноническое имя", syn.name("Donetsk basins") == "Донецкий бассейн")
    check("в любом падеже", syn.name("Донецкого бассейна") == "Донецкий бассейн")
    check("чего нет в словаре — None", syn.name("Eastern Donbass") is None)
    check("связь по словарю", syn.relation("Associated with") == "приурочено к")
    check("связи нет в словаре — как написано, строчными", syn.relation("Контролирует ") == "контролирует")
    check("имя без синонимов — тоже в словаре", "Анабарский щит" in syn.groups)
    dup = S.parse({"имена": {"А": ["общее"], "Б": ["общее"]}})
    check("одно написание у двух имён — предупреждение, не ошибка",
          bool(dup.warnings) and dup.name("общее") == "А")
    for bad, why in (({"имена": ["список"]}, "не словарь"), ({"прочее": {}}, "неизвестный раздел"),
                     ({"имена": {"А": 5}}, "синонимы не списком")):
        try:
            S.parse(bad)
            check(f"ошибка: {why}", False)
        except S.SynonymsError:
            check(f"ошибка: {why}", True)
    real = S.load()
    check("файл config/synonyms.yaml читается",
          real.name("Anabar shield") == "Анабарский щит" and not real.warnings, str(real.warnings))

    names = {"a": ("Донецкий бассейн", 5), "b": ("Donetsk basin", 2), "c": ("Eastern Donbass", 1),
             "d": ("окварцевание", 3)}
    vec = {"Донецкий бассейн": [1, 0, 0], "Donetsk basin": [0.95, 0.31, 0],
           "Eastern Donbass": [0.9, 0, 0.43], "окварцевание": [0, 0, 1]}
    asked = []

    def judge(pairs):
        asked.extend(pairs)
        return [set(p) == {"Донецкий бассейн", "Donetsk basin"} for p in pairs]
    groups = S.suggest(names, S.Synonyms(), lambda xs: [vec[x] for x in xs], judge, threshold=0.8)
    check("похожие имена спрашиваются у модели, непохожие — нет",
          any(set(p) == {"Донецкий бассейн", "Donetsk basin"} for p in asked)
          and not any("окварцевание" in p for p in asked), str(asked))
    check("часть и целое модель не сводит — в группе только одно и то же",
          groups == [{"canon": "Донецкий бассейн", "variants": ["Donetsk basin"], "count": 7,
                      "known": False}], str(groups))
    text = S.proposals_text(groups, [])
    check("предложения — в виде словаря, с пояснением",
          "Донецкий бассейн: [Donetsk basin]" in text and "перенесите нужные строки" in text)


def test_facts_check() -> None:
    print("\nФакты от Qwen3: проверка кодом")
    rules = facts.load_rules()
    check("правила читаются, комментарии не уходят модели",
          "Бери только то, что написано" in rules and "Правится" not in rules)
    system, user = facts.messages(rules, "Текст фрагмента.")
    check("модели — правила и фрагмент, без словаря", "«от»" in system
          and "Текст фрагмента." in user and "факты" in user)
    text = ("Тырныаузский рудный узел расположен в пределах Анабарского щита. Оруденение "
            "Тырныаузского рудного узла приурочено к зонам дробления северо-западного "
            "простирания. В пределах рудного района выявлены зоны окварцевания. "
            "Работы 2019 г. выполнены А. П. Степановым.")
    answer = {"факты": [
        {"от": "Тырныаузский рудный узел", "связь": "Входит в", "к": "Анабарский щит",
         "цитата": "Тырныаузский рудный узел расположен в пределах Анабарского щита."},
        {"от": "оруденение", "связь": "приурочено к", "к": "зоны дробления",
         "цитата": "Оруденение Тырныаузского рудного узла приурочено к зонам дробления"},
        {"от": "Онежский рудный район", "связь": "содержит", "к": "зоны окварцевания",
         "цитата": "В пределах рудного района выявлены зоны окварцевания."},
        {"от": "золото", "связь": "связано с", "к": "дайки",
         "цитата": "Золото связано с дайками гранитоидов."},
        {"от": "2019", "связь": "год работ", "к": "Степанов",
         "цитата": "Работы 2019 г. выполнены А. П. Степановым."},
        {"от": "рудный узел", "связь": "входит в", "к": "рудного узла", "цитата": "…"},
        {"от": "оруденение", "связь": "очень длинная связь из многих слов подряд",
         "к": "зоны дробления",
         "цитата": "Оруденение Тырныаузского рудного узла приурочено к зонам дробления"},
        {"от": "Оруденение", "связь": "приурочено к", "к": "зонам дробления",
         "цитата": "Оруденение Тырныаузского рудного узла приурочено к зонам дробления"},
        "мусор",
    ]}
    got = facts.check(answer, text)
    triples = [(f.src, f.relation, f.dst) for f in got.facts]
    reasons = " | ".join(got.rejected)
    check("верные факты приняты, связь строчными",
          triples == [("Тырныаузский рудный узел", "входит в", "Анабарский щит"),
                      ("оруденение", "приурочено к", "зоны дробления")], str(triples))
    check("дописанное имя («Онежский рудный район») — отброшено",
          "в цитате нет «Онежский рудный район»" in reasons, reasons)
    check("выдуманная цитата — отброшена", "цитаты нет во фрагменте" in reasons)
    check("год вместо предмета — отброшен", "число или год" in reasons)
    check("связь с самим собой — отброшена", "с самим собой" in reasons)
    check("слишком длинная связь — отброшена", "длиннее 6 слов" in reasons)
    check("повтор того же факта в другом падеже не задваивается", len(got.facts) == 2)
    check("цитата — как в статье", got.facts[1].quote.startswith("Оруденение Тырныаузского"))
    check("мусор вместо JSON — ничего", not facts.check("не json", text).facts)


def test_facts_run() -> None:
    print("\nПроход модели по фрагментам")
    text = "Тырныаузский рудный узел расположен в пределах Анабарского щита, это важно."
    good = {"факты": [{"от": "Тырныаузский рудный узел", "связь": "входит в",
                       "к": "Анабарский щит",
                       "цитата": "Тырныаузский рудный узел расположен в пределах Анабарского щита"}]}

    class FakeLLM:
        model = "qwen3:14b-проба"

        def __init__(self, answers):
            self.answers = list(answers)

        def chat_json(self, system, user):
            item = self.answers.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    conn = _FakeConn(rows=[[(1, "d1", text), (2, "d2", text)]])
    stats = facts.run(conn, FakeLLM([good, {"факты": []}]), log=lambda *a: None)
    check("оба фрагмента пройдены, факт принят", stats["done"] == 2 and stats["facts"] == 1,
          str(stats))
    sql = " ".join(q for q, _ in conn.executed)
    check("по фрагменту: прошлое заменяется, отметка о проходе",
          "DELETE FROM facts WHERE chunk_id" in sql and "INSERT INTO facts_pass" in sql)
    inserted = next(p for q, p in conn.executed if q.startswith("INSERT INTO facts ("))
    check("в базу — имя как написано и ключ без падежей",
          inserted[0][2] == "Тырныаузский рудный узел"
          and inserted[0][5] == name_key("Тырныаузский рудный узел"))
    check("после каждого фрагмента — сохранение (можно прервать)", conn.commits == 2)
    conn = _FakeConn(rows=[[(i, "d", text) for i in range(1, 5)]])
    stats = facts.run(conn, FakeLLM([RuntimeError("нет")] * 3), log=lambda *a: None)
    check("модель молчит трижды — остановка без записи", stats["stopped"] and stats["done"] == 0)


def test_graph() -> None:
    print("\nГраф: имена сведены, связи с цитатами, датасеты")
    syn = S.parse({"имена": {"Донецкий бассейн": ["Donetsk basin"]},
                   "связи": {"входит в": ["part of"], "приурочено к": ["associated with"]}})
    rows = [
        row(1, "Eastern Donbass", "part of", "Donetsk basin", "d1",
            "Eastern Donbass is part of the Donetsk basin."),
        row(2, "Nesvetaevsky complex", "расположено в", "Eastern Donbass", "d1", "q2"),
        row(3, "оруденение", "приурочено к", "Nesvetaevsky complex", "d1", "q3"),
        row(4, "золото", "associated with", "Донецкого бассейна", "d2", "q4"),
        row(5, "золото", "приурочено к", "Донецкий бассейн", "d3", "q5"),
        row(6, "Донецкий бассейн", "входит в", "Donetsk basin", "d3", "сам с собой после словаря"),
    ]
    g = api.Graph(rows, syn)
    check("словарь и падежи свели написания в одно имя",
          set(g.entities) == {"Донецкий бассейн", "Eastern Donbass", "Nesvetaevsky complex",
                              "оруденение", "золото"}, str(sorted(g.entities)))
    link = g.links[("золото", "приурочено к", "Донецкий бассейн")]
    check("одинаковые факты из разных статей — одна связь, статей 2",
          link.documents == 2 and len(link.quotes()) == 2)
    check("факт сам с собой после словаря — не связь",
          ("Донецкий бассейн", "входит в", "Донецкий бассейн") not in g.links)
    check("что входит — по фактам «входит в»",
          g.parts("Донецкий бассейн") == ["Донецкий бассейн", "Eastern Donbass"])
    check("узнаёт имя, как его написал человек", g.resolve("Donetsk basins") == "Донецкий бассейн"
          and g.resolve("золота") == "золото" and g.resolve("Марс") is None)

    data = api.dataset(None, syn, "Донецкий бассейн", g)
    lines = [(r["about"], r["direction"], r["relation"], r["other"], r["documents"])
             for r in data["facts"]]
    check("датасет — факты о нём и о том, что в него входит; состав — не факт",
          lines == [("Донецкий бассейн", "←", "приурочено к", "золото", 2),
                    ("Eastern Donbass", "←", "расположено в", "Nesvetaevsky complex", 1)],
          str(lines))
    check("у факта — цитаты и статьи", data["facts"][0]["quotes"][0]["title"] == "Статья d2"
          and data["facts"][0]["quotes"][0]["url"] == "https://doi.org/10.1/d2")
    csv = api.dataset_csv(data)
    check("CSV для Excel: BOM, точка с запятой, цитата",
          csv.startswith("﻿") and "Донецкий бассейн;←;приурочено к;золото;2;2;q4" in csv,
          csv[:200])

    conn = _FakeConn(rows=[[(r["id"], r["chunk_id"], r["doc_id"], r["src"], r["relation"],
                             r["dst"], r["quote"], r["title"], r["year"], r["url"], r["doi"])
                            for r in rows],
                           [(5, 7, 0, None)], [(40,)]])
    net = api.network_response(conn, syn)
    kinds = {n["kind"] for n in net["nodes"]}
    fact_edges = [e for e in net["edges"] if e["type"] == "факт"]
    check("сеть: сущности и статьи, у связей — подпись и цитаты",
          kinds == {"сущность", "статья"} and all(e["label"] and e["quotes"] for e in fact_edges))
    check("сеть: из какой статьи узел", any(e["type"] == "из статьи" and e["source"] == "статья:d2"
                                           for e in net["edges"]))
    check("сеть: сколько прошла модель", net["pass"]["passed"] == 5 and net["pass"]["chunks"] == 40)

    def fake(rs):
        return _FakeConn(rows=[[(r["id"], r["chunk_id"], r["doc_id"], r["src"], r["relation"],
                                 r["dst"], r["quote"], r["title"], r["year"], r["url"], r["doi"])
                                for r in rs], [(5, 7, 0, None)], [(40,)]])

    listed = {d["name"]: d for d in api.datasets_response(fake(rows), syn)["datasets"]}
    check("список датасетов — те же числа, что в карточке",
          listed["Донецкий бассейн"]["facts"] == len(data["facts"])
          and listed["Донецкий бассейн"]["documents"] == data["documents"], str(listed))
    check("готовых датасетов нет: пусто в фактах — пусто в списке",
          api.datasets_response(fake([]), syn)["datasets"] == [])
    more = rows + [row(7, "Ростовский выступ", "содержит", "золото", "d4", "q7")]
    listed2 = {d["name"]: d for d in api.datasets_response(fake(more), syn)["datasets"]}
    check("новый факт после прохода модели — новый датасет и обновлённый старый",
          "Ростовский выступ" in listed2 and "Ростовский выступ" not in listed
          and listed2["золото"]["facts"] == listed["золото"]["facts"] + 1, str(listed2))

    try:
        api.dataset_response(_FakeConn(rows=[[]]), syn, {"name": "Марс"})
        check("чужое имя — понятная ошибка", False)
    except api.RequestError as exc:
        check("чужое имя — понятная ошибка", "нет" in exc.message)


def test_review_fixes() -> None:
    print("\nИсправления: ключи имён, границы слов, обратные связи, имена внутри других")
    check("«разлом» и «разломы» — один ключ",
          len({name_key(w) for w in ("разлом", "разломы", "разлома", "разломов")}) == 1)
    check("«покров» и «покрова» — один ключ", name_key("покров") == name_key("покрова"))
    check("разное не сливается: соль ≠ сель, мел ≠ мол",
          name_key("соль") != name_key("сель") and name_key("мел") != name_key("мол"))
    check("processes = process", name_key("processes") == name_key("process"))
    check("слово не находится внутри другого: «щит» — не в «защиты»",
          not mentions("щит", "меры защиты от выветривания")
          and not mentions("магнетит", "встречается титаномагнетит"))
    check("короткое слово — в любом падеже: «зона» в «зоны», «руда» в «руды»",
          mentions("зона", "в пределах этой зоны") and mentions("руда", "состав руды"))

    ans = {"факты": [
        {"от": "Возраст браннерита", "связь": "составляет", "к": "385 ± 2 млн лет",
         "цитата": "Возраст браннерита составляет 385 ± 2 млн лет по данным датирования."},
        {"от": "Binotto (2015)", "связь": "предпочитает", "к": "SAM",
         "цитата": "Binotto (2015) предпочитает SAM для картирования изменений."},
        {"от": "в аномалиях Bi", "связь": "выражен", "к": "Верхний скарн",
         "цитата": "Верхний скарн слабее всего выражен в аномалиях Bi на площади."},
        {"от": "южнее Персияновского разлома", "связь": "расположены", "к": "дайки",
         "цитата": "Дайки расположены южнее Персияновского разлома в западной части."},
    ]}
    text = " ".join(f["цитата"] for f in ans["факты"])
    got = facts.check(ans, text)
    reasons = " | ".join(got.rejected)
    check("количество, ссылка на автора и положение — не предметы",
          "количество" in reasons and "ссылка на автора" in reasons and "положение" in reasons,
          reasons)
    check("предлог в начале имени снят, факт принят",
          [(f.src, f.dst) for f in got.facts] == [("аномалиях Bi", "Верхний скарн")],
          str([(f.src, f.dst) for f in got.facts]))

    syn = S.parse({"связи": {"входит в": ["part of"]},
                   "обратные": {"входит в": ["включает"], "контролирует": ["контролируется"]}})
    rows = [row(1, "Анабарский щит", "включает", "Билляхская зона", "d1"),
            row(2, "оруденение", "контролируется", "зоны разломов", "d1"),
            row(3, "золото", "приурочено к", "зоне Персияновского разлома", "d2"),
            row(4, "дайки", "расположены вдоль", "Персияновского разлома в западной части", "d3"),
            row(5, "Персияновский", "упомянут", "рудные тела", "d3")]
    g = api.Graph(rows, syn)
    check("«включает» развёрнут в «входит в»",
          ("Билляхская зона", "входит в", "Анабарский щит") in g.links
          and g.parts("Анабарский щит") == ["Анабарский щит", "Билляхская зона"])
    check("«контролируется» развёрнут в «контролирует»",
          ("зоны разломов", "контролирует", "оруденение") in g.links)
    check("имени нет, но оно внутри других — узнаётся",
          g.resolve("Персияновский разлом") == "Персияновский разлом"
          and sorted(g.related("Персияновский разлом")) ==
          ["Персияновского разлома в западной части", "зоне Персияновского разлома"])
    data = api.dataset(None, syn, "Персияновский разлом", g)
    check("датасет имени собирает факты тех, где оно внутри",
          data["documents"] == 2 and len(data["facts"]) == 2, str(data["facts"]))
    check("у общих слов «родственников» нет: «золото» не тянет всё золотое",
          g.related("золото") == [])
    try:
        S.parse({"обратные": {"входит в": ["включает"]}, "прочее": {}})
        check("неизвестный раздел — ошибка", False)
    except S.SynonymsError as exc:
        check("неизвестный раздел — ошибка, в подсказке есть «обратные»", "обратные" in str(exc))


def test_graph_uses() -> None:
    print("\nГраф в деле: синонимы в поиске, вопросы оценки, темы по пробелам")
    import yaml
    from georag.graph import gaps, questions

    syn = S.parse({"имена": {"Донецкий бассейн": ["Donetsk basin", "Донбасс"],
                             "метод спектрального угла": ["SAM"],
                             "Анабарский щит": ["Anabar shield"]}})
    alts = S.synonym_variants("золото Донбасса", syn)
    check("запрос с другими написаниями из словаря",
          "золото donetsk basin" in alts and "золото донецкий бассейн" in alts, str(alts))
    check("сокращение — только целым словом: SAM не в «sample»",
          S.synonym_variants("sample preparation", syn) == []
          and "метод спектрального угла карта" in S.synonym_variants("SAM карта", syn))
    check("нет названий из словаря — вариантов нет", S.synonym_variants("плотность линеаментов", syn) == [])

    rows = [row(1, "оруденение", "приурочено к", "зоны дробления", "d1",
                "Оруденение приурочено к зонам дробления вдоль разлома."),
            row(2, "Персияновский разлом", "контролирует", "дайки", "d2", "q2"),
            row(3, "Персияновский разлом", "сопровождается", "магнитные аномалии", "d2", "q3"),
            row(4, "Персияновский разлом", "пересекает", "Южная антиклиналь", "d2", "q4"),
            row(5, "385 ± 2 млн лет", "возраст", "браннерит", "d3", "q5")]
    g = api.Graph(rows, syn)
    items = questions.build(g, None, limit=10, log=None)
    check("вопросы — из разных статей по очереди", [i["статьи"][0] for i in items[:2]] == ["d1", "d2"],
          str([i["статьи"] for i in items]))
    check("в вопросе нет ответа, ответ — в ключевых",
          all(i["ключевые"][0].split("|")[0] not in i["вопрос"] for i in items)
          and items[0]["ключевые"][0].startswith("зоны дробления"), str(items[0]))
    check("факты с числами в вопросы не идут", all("385" not in i["факт"] for i in items))

    class Talky:
        def __init__(self, text):
            self.text = text

        def chat_json(self, system, user):
            return {"вопрос": self.text}
    link = g.links[("оруденение", "приурочено к", "зоны дробления")]
    check("модель переформулировала — берётся её вопрос",
          questions.phrase(Talky("К чему приурочено оруденение"), link) == "К чему приурочено оруденение?")
    check("модель выдала ответ в вопросе — шаблон",
          questions.phrase(Talky("Приурочено ли оруденение к зонам дробления?"), link)
          == questions.template(link))
    loaded = yaml.safe_load(questions.to_yaml(items))
    check("файл вопросов читается оценкой", loaded["вопросы"][0]["статьи"] == ["d1"])

    found = {r["name"]: r for r in gaps.find(g, syn)}
    check("пробел: в словаре есть, фактов нет — с английской темой",
          found.get("Анабарский щит", {}).get("topics") == ["Анабарский щит", "Anabar shield"], str(found))
    check("пробел: много фактов из одной статьи", "Персияновский разлом" in found
          and "одной статьи" in found["Персияновский разлом"]["why"])
    check("есть факты — не пробел", "Донецкий бассейн" in found and "оруденение" not in found)
    topics = yaml.safe_load(gaps.to_yaml(list(found.values())))["topics"]
    check("темы — в виде topics.yaml", "Anabar shield" in topics and "Персияновский разлом" in topics)


def main() -> int:
    test_text()
    test_synonyms()
    test_facts_check()
    test_facts_run()
    test_graph()
    test_review_fixes()
    test_graph_uses()

    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
