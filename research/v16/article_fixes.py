"""Правки научной редакции статьи (Статья_ИПУЗ_2026_Егоров_научная_редакция.docx) по
результатам v16: замены по фрагментам оформления, без пересборки документа. Каждая замена
должна найтись ровно один раз. Числа — из results/*.json (см. docs/article_updates.md).

    article_fixes.py <исходный.docx> <итоговый.docx>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def n(x: float) -> str:
    return str(x) if isinstance(x, int) else f"{x:.2f}".replace(".", ",")  # в статье — без пробелов в числах


def replacements() -> list[tuple[str, str]]:
    a = json.loads((RES / "relation_pipeline_summary.json").read_text("utf-8"))
    t = a["typing"]
    nongeo = t["без типа: не о геологии"]
    kinds = t["без типа: виды концов не подходят к типу связи"]
    other = a["excluded_typing"] - nongeo - kinds
    assert a["stored"] - a["graph"] - a["candidates"] == a["excluded_typing"] == 3585
    out = [
        ("Предметный словарь содержит 435 русско- и англоязычных терминов",
         "Предметный словарь содержит 410 русско- и англоязычных терминов"),
        ("Дополнительные обращения к модели служат проверкой согласованности с фрагментом.",
         "Дополнительные обращения к модели служат проверкой согласованности с фрагментом: отношение "
         "принимается, если выдержало не менее трёх из четырёх контрольных вопросов по предложению-основанию "
         "либо его предметы связаны и в другой статье. Затем модель назначает принятому отношению тип из "
         "закрытого списка; отношение без допустимого типа в граф не включается, но сохраняется в базе."),
        ("Всего языковая модель предложила 22218 отношений. Базовые программные проверки исключили 8479,",
         "Всего языковая модель предложила не менее 22218 отношений. Базовые программные проверки исключили "
         "не менее 8479 (журнал хранит до 40 отклонений на фрагмент),"),
        ("Для остальных 3585 записей из числа прошедших базовую проверку в имеющейся сводке не приведён "
         "отдельный итоговый статус; поэтому представленные числа не составляют полной воронки решений, и "
         "причины дальнейшего исключения этих записей требуют сверки с журналом обработки.",
         f"Остальные 3585 записей прошли подтверждение, но не получили типа из закрытого списка: {n(nongeo)} "
         f"модель отнесла к негеологическим, у {n(kinds)} виды предметов не соответствуют типу отношения, "
         f"{n(other)} исключены из-за неназванного объекта, конца связи, не являющегося предметом, или типа "
         "вне списка. В граф они не входят, но сохраняются в базе: 13739 = 3818 + 6336 + 3585. Решения "
         "этапа типизации принимает модель, их точность вручную пока не оценена."),
        ("Из 16 ошибок семь связаны с потерей оговорки о предположительном характере вывода, шесть — с "
         "неверным типом отношения,",
         "Из 16 ошибок шесть связаны с потерей оговорки о предположительном характере вывода, семь — с "
         "неверным типом отношения,"),
        ("Во-вторых, по итоговой статистике не установлены причины, по которым 3585 формально прошедших "
         "кандидатов не попали ни в граф, ни в набор неподтверждённых; необходим полный журнал решений.",
         "Во-вторых, 3585 подтверждённых отношений исключены из графа решением модели на этапе типизации; "
         "доля напрасно исключённых геологических отношений не измерена."),
        ("Приоритеты продолжения — полный аудит статусов отношений,",
         "Приоритеты продолжения — ручная проверка решений типизации,"),
        ("Cross-Script Name Matching", "Cross-Script Toponym Matching"),
    ]
    return out + retrieval_replacements()


def retrieval_replacements() -> list[tuple[str, str]]:
    """Сравнение с простым территориальным фильтром и графовыми кандидатами — только если
    прогон retrieval_eval.py завершён (results/retrieval_summary.json)."""
    path = RES / "retrieval_summary.json"
    if not path.exists():
        return []
    from article_retrieval import replacements as rr  # noqa: PLC0415 — текст зависит от итогов

    return rr(json.loads(path.read_text("utf-8")))


def replace_in_paragraph(p, old: str, new: str) -> bool:
    if old not in p.text:
        return False
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    # Фрагмент проходит через несколько участков оформления: текст абзаца — в первый участок.
    text = p.text.replace(old, new)
    for r in p.runs[1:]:
        r.text = ""
    p.runs[0].text = text
    return True


def main(src: str, out: str) -> None:
    sys.path.insert(0, str(HERE))
    doc = Document(src)
    paras = list(doc.paragraphs) + [p for t in doc.tables for row in t.rows for c in row.cells for p in c.paragraphs]
    for old, new in replacements():
        hits = sum(replace_in_paragraph(p, old, new) for p in paras)
        print(hits, "|", old[:70])
        assert hits == 1, old
    if (RES / "retrieval_summary.json").exists():
        import article_retrieval

        article_retrieval.insert(doc, json.loads((RES / "retrieval_summary.json").read_text("utf-8")))
    doc.save(out)
    text = "\n".join(p.text for p in doc.paragraphs) + "".join(
        c.text for t in doc.tables for row in t.rows for c in row.cells)
    print("сохранено", out, "| знаков:", len(text))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
