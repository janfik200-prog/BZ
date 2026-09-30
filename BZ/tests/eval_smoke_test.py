"""Проверка оценки: вопросы читаются, ключевые слова ищутся, отчёт собирается.

Запуск:  python tests/eval_smoke_test.py

Ни базы, ни моделей не нужно: поиск и ответ подменены.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag import evaluation as E  # noqa: E402
from georag.chat import answer as A  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def test_questions_file() -> None:
    print("\nФайл вопросов")
    qs = E.load_questions()
    check("вопросов не меньше двадцати", len(qs) >= 20, str(len(qs)))
    check("есть контрольные не по теме", sum(q.expect == E.NOT_IN_BASE for q in qs) >= 3)
    check("у вопросов по теме есть ключевые",
          all(q.keywords for q in qs if q.expect == E.IN_BASE))
    check("у вопросов есть тема для сводки", sum(bool(q.concept) for q in qs) >= 10)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "q.yaml"
        path.write_text("вопросы:\n  - {id: a, вопрос: x}\n  - {id: a, вопрос: y}\n",
                        encoding="utf-8")
        try:
            E.load_questions(path)
            check("повтор id — ошибка", False)
        except E.QuestionsError as exc:
            check("повтор id — ошибка", "повторяется" in str(exc))
        path.write_text("вопросы:\n  - {id: a, вопрос: x, ожидание: может быть}\n",
                        encoding="utf-8")
        try:
            E.load_questions(path)
            check("неизвестное ожидание — ошибка", False)
        except E.QuestionsError:
            check("неизвестное ожидание — ошибка", True)


def test_keywords() -> None:
    print("\nКлючевые слова")
    groups = [["рудный узел", "ore cluster"], ["линеамент", "lineament"], ["ROC", "AUC"]]
    found, missing = E.keyword_hits(groups, "В пределах рудного узла плотность линеаментов выше.")
    check("падежи и беглая гласная", found == 2, str(missing))
    check("чего нет — названо", missing == ["ROC"])
    found, _ = E.keyword_hits([["ROC"]], "The rock is altered; the process is slow.")
    check("аббревиатура не ловится внутри слова", found == 0)
    found, _ = E.keyword_hits([["ROC"]], "Качество по ROC-кривой — 0,86.")
    check("аббревиатура ловится целиком", found == 1)
    found, _ = E.keyword_hits([["ore cluster"]], "Delineation of ore clusters in Yakutia")
    check("английское множественное число", found == 1)


def _q(expect=E.IN_BASE, keywords=(("линеамент",),)) -> E.Question:
    return E.Question("t1", "Как выделяют рудные узлы?", expect, [list(k) for k in keywords],
                      "структурный контроль")


def _src(n, text):
    return {"n": n, "title": "Статья", "text": text, "year": 2020, "similarity": 0.6,
            "found_by": "вектор"}


def test_checks() -> None:
    print("\nПроверка поиска и ответа")
    s = E.check_search(_q(), [_src(1, "Плотность линеаментов.")])
    check("поиск: нашлось и есть ключевое — прошёл", s["ok"] and s["keywords"] == 1.0)
    s = E.check_search(_q(), [_src(1, "Про другое.")])
    check("поиск: нет ключевого — не прошёл", not s["ok"] and s["missing"] == ["линеамент"])
    check("поиск: пусто — не прошёл", not E.check_search(_q(), [])["ok"])
    check("контрольный: пусто — прошёл", E.check_search(_q(E.NOT_IN_BASE, ()), [])["ok"])
    check("контрольный: что-то нашлось — не прошёл",
          not E.check_search(_q(E.NOT_IN_BASE, ()), [_src(1, "x")])["ok"])
    q = _q()
    q.docs = ["d1", "d2", "d3"]
    hit = dict(_src(1, "Плотность линеаментов."), doc_id="d1")
    s = E.check_search(q, [hit])
    check("нужные статьи: найдена одна из трёх — не прошёл, названы ненайденные",
          not s["ok"] and s["docs"] == 0.33 and s["docs_missing"] == ["d2", "d3"], str(s["docs"]))
    s = E.check_search(q, [hit, dict(hit, doc_id="d2")])
    check("нужные статьи: две из трёх — прошёл", s["ok"] and s["docs"] == 0.67)

    long_answer = ("Линеаменты сгущаются у рудных узлов, и это главный признак [1]. " * 12)
    events = [{"type": "token", "text": long_answer},
              {"type": "done", "mode": "база", "used": [1], "unknown": [], "coverage": [12, 12],
               "seconds": 3.0}]
    a = E.check_answer(_q(), events)
    check("ответ по базе, длинный, со ссылками — прошёл", a["ok"], "; ".join(a["problems"]))
    short = [{"type": "token", "text": "Линеаменты у узлов [1]."},
             {"type": "done", "mode": "база", "used": [1], "unknown": [], "coverage": [1, 1]}]
    a = E.check_answer(_q(), short)
    check("отписка в одну фразу — не прошла", not a["ok"] and "короткий" in a["problems"][0])
    middle = [{"type": "token", "text": "Линеаменты сгущаются у рудных узлов [1]. " * 6},
              {"type": "done", "mode": "база", "used": [1], "unknown": [], "coverage": [6, 6]}]
    check("ответ по существу (не обязательно длинный) — проходит по длине",
          not any("короткий" in p for p in E.check_answer(_q(), middle)["problems"]))
    bad = [{"type": "token", "text": long_answer},
           {"type": "done", "mode": "база", "used": [1], "unknown": [7], "coverage": [3, 12]}]
    problems = E.check_answer(_q(), bad)["problems"]
    check("мало ссылок и выдуманный номер — оба замечены",
          any("мало ссылок" in p for p in problems) and any("несуществующие" in p for p in problems),
          "; ".join(problems))
    general = [{"type": "general", "text": "нет"}, {"type": "token", "text": "что-то"},
               {"type": "done", "mode": "без базы", "used": [], "unknown": [], "coverage": [0, 0]}]
    check("по теме, а ответил без базы — не прошёл",
          not E.check_answer(_q(), general)["ok"])
    check("не по теме и ответил без базы — прошёл",
          E.check_answer(_q(E.NOT_IN_BASE, ()), general)["ok"])
    err = [{"type": "error", "error": "Ollama не отвечает"}]
    check("ошибка модели — не прошёл, причина видна",
          "Ollama" in E.check_answer(_q(), err)["problems"][0])


def test_run_and_report() -> None:
    print("\nПрогон и отчёт")
    real_find, real_answer = A.find_sources, A.answer
    try:
        A.find_sources = lambda conn, emb, queries, settings, **kw: [_src(1, "Плотность линеаментов.")]
        seen = {}

        def fake_answer(conn, emb, question, history, settings, llm=None, prepared=None):
            seen["prepared"] = prepared
            yield {"type": "token", "text": "Линеаменты сгущаются у узлов [1]. " * 30}
            yield {"type": "done", "mode": "база", "used": [1], "unknown": [],
                   "coverage": [30, 30], "seconds": 2.0}
        A.answer = fake_answer
        planner = lambda q, h, s: ["lineament density", "плотность линеаментов"]  # noqa: E731
        r = E.run_question(None, None, _q(), A.Settings(), with_answer=True, planner=planner,
                           judge=None)
        check("запросы: вопрос и запросы модели", r["queries"][0] == "Как выделяют рудные узлы?"
              and len(r["queries"]) == 3, str(r["queries"]))
        check("ищется один раз — найденное передано в ответ готовым",
              isinstance(seen["prepared"], A.Prepared)
              and seen["prepared"].plan.queries == r["queries"]
              and seen["prepared"].sources[0]["text"] == "Плотность линеаментов.")
        check("видно, отбирала ли модель и сколько кругов", r["judged"] is False and r["rounds"] == 1)
        check("ответ оценён", r["answer"]["ok"] and r["answer"]["mode"] == "база")
        control = E.run_question(None, None, E.Question("x1", "Как приготовить борщ?",
                                                        E.NOT_IN_BASE), A.Settings(),
                                 with_answer=False, planner=lambda *a: [], judge=None)
    finally:
        A.find_sources, A.answer = real_find, real_answer

    results = [r, control]
    summary = E.summarize(results)
    check("сводка: поиск по теме нашёл", summary["search_found"] == 1.0)
    check("сводка: контрольный — поиск кое-что подсунул", summary["control_clean"] == 0.0)
    check("сводка: ответы посчитаны", summary["answer_in_base"] == 1.0
          and summary["answer_chars"] > 600)
    meta = {"when": "01.10.2026 10:00", "documents": 10, "chunks": 300, "model": "qwen3:14b",
            "with_answer": True, "planner": True}
    md = E.report(results, summary, {"search_ok": 1.0, "answer_chars": 500}, meta)
    check("отчёт: таблица итога", "| вопросов прошло по поиску |" in md)
    check("отчёт: сравнение с прошлым", "к прошлому" in md)
    check("отчёт: разбор неудачного", "## Что не так" in md and "x1" in md)

    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp)
        (logs / "eval-20260101-1000.json").write_text(
            json.dumps({"summary": {"search_ok": 0.5}}), encoding="utf-8")
        (logs / "eval-20260201-1000.json").write_text(
            json.dumps({"summary": {"search_ok": 0.7}}), encoding="utf-8")
        check("прошлый прогон — самый поздний до текущего",
              E.previous_run(logs, "20260301-1000") == {"search_ok": 0.7})
        check("первый прогон — сравнивать не с чем", E.previous_run(logs, "20250101-1000") is None)


def main() -> int:
    test_questions_file()
    test_keywords()
    test_checks()
    test_run_and_report()
    print(f"\nИтого: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    if FAILED:
        print("Провалены: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
