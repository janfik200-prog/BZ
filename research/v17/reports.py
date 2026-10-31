"""docs/prefilter_sampling_audit.md по results/corpus_structure.json и results/labeling_scores.json
(числа не вписываются руками: документ пересобирается после каждой разметки).

    reports.py → docs/prefilter_sampling_audit.md,
                 docs/relation_pipeline_summary.md + results/relation_pipeline_summary.md (раздел v17)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES, DOCS = HERE / "results", HERE / "docs"


def d(x, nd: int = 3) -> str:
    return "—" if x in ("", None) else f"{float(x):.{nd}f}".replace(".", ",")


def pct(x) -> str:
    return "—" if x in ("", None) else f"{100 * float(x):.1f} %".replace(".", ",")


def main() -> None:
    cs = json.loads((RES / "corpus_structure.json").read_text("utf-8"))
    lab = json.loads((RES / "labeling_scores.json").read_text("utf-8"))
    pop, fs, by = lab["filter_population"], lab["filter_sample"], lab["filter_by_terms"]
    rows = lab["filter_thresholds"]
    frame = {r["threshold"]: r for r in rows if r["population"] == "frame_ge200"}
    dom = {r["threshold"]: r for r in rows if r["population"] == "domain_all"}
    v16 = lab["filter_recall_v16_original"]
    names = {(False, False, False): "обычные", (False, False, True): "короткие (< 200 знаков)",
             (False, True, False): "подписи и легенды", (False, True, True): "подписи, короткие",
             (True, False, False): "дубликаты", (True, False, True): "дубликаты, короткие",
             (True, True, False): "дубликаты-подписи", (True, True, True): "дубликаты-подписи, короткие"}
    un = fs["unlabelled"]
    done = pop["short_complete"]
    L = [
        "# Выборка для оценки словарного фильтра: совокупность, отбор, исключения",
        "",
        "Документ собирается скриптом `reports.py` из `results/corpus_structure.json` и `results/labeling_scores.json`.",
        "",
        "## Что было не так в v16",
        "",
        "Выборку для разметки `labeling_sheet.py` отбирал только из фрагментов длиной от 200 знаков "
        "(`len(c[\"text\"]) >= 200`). При этом доли переносились на все фрагменты, кроме дубликатов и подписей, "
        "без ограничения длины (`filter_population`). Короткие фрагменты не могли попасть в разметку, но "
        "участвовали в расчёте полноты. В коде системы (`facts.run`) правила о длине нет: короткий фрагмент "
        "проходит фильтр так же, как длинный.",
        "",
        "## Структура корпуса",
        "",
        "Признаки пересекаются, поэтому фрагменты разложены по комбинациям: каждый фрагмент считается один раз.",
        "",
        "| Комбинация признаков | Фрагментов |",
        "|---|---|",
    ]
    for c in cs["combinations"]:
        L.append(f"| {names[(c['duplicate'], c['caption'], c['short'])]} | {c['chunks']} |")
    L += [
        f"| **всего** | **{cs['total']}** |",
        "",
        f"- Дубликатов (MinHash, `dedup.py`): {cs['duplicates']}. Подписей и легенд (`captions.py`): {cs['captions']}. "
        f"Короче 200 знаков: {cs['short_lt_200']}, из них {cs['short_lt_200'] - pop['short_lt200']} — подписи.",
        f"- **Область применения фильтра** — не дубликаты и не подписи: {pop['domain']} фрагментов. Дубликаты и "
        "подписи отсеиваются другими правилами при любом пороге.",
        f"- **Рамка выборки v16** — область без коротких: {pop['frame_ge200']} фрагментов.",
        f"- **Короткие фрагменты области**: {pop['short_lt200']} ({pct(pop['short_lt200'] / pop['domain'])} "
        f"фрагментов, {pct(cs['short_tokens'] / cs['domain_tokens'])} токенов). "
        f"Связи графа есть в {cs['short_with_graph_facts']} из них, записанные связи — в {cs['short_with_stored_facts']}.",
        "",
        "## Способ отбора и оценки",
        "",
        f"1. **Рамка.** Из {pop['frame_ge200']} фрагментов отобрано {fs['selected']}: по 25 из каждой группы по числу "
        "терминов словаря (0, 1, 2, 3, 4, 5 и больше), зерно 20261031. Разметчик не видел числа терминов и "
        "извлечённых связей.",
        f"2. **Размечено {fs['labelled']} из {fs['selected']}.** "
        + (f"Строка {un[0]['row']} (фрагмент {un[0]['chunk_id']}, терминов {un[0]['terms']}) оставлена автором пустой "
           f"с пометкой «{un[0]['note'].rstrip('.')}». Оценка неоднозначна, поэтому строка в расчёт не входит, ни 0, ни 1 ей "
           "не присваивается. В группе «3 термина» — 24 ответа." if un else "Пустых строк нет."),
        "3. **Веса страт исправлены.** Доля фрагментов со связями в группе переносится на численность группы "
        "**в рамке**, а не во всей области. Так вывод относится к той совокупности, из которой бралась выборка.",
        f"4. **Короткие фрагменты** (все {pop['short_lt200']}) размечаются сплошь, по тому же правилу: лист «Короткие» "
        "в `Разметка_v17.xlsx`. У сплошной разметки нет ошибки выборки, и оценка для всей области — сумма "
        "двух страт. "
        + (f"Размечено {pop['short_labelled']} из {pop['short_lt200']}, со связями — {pop['short_with_relations']}."
           if done else f"**Пока размечено {pop['short_labelled']} из {pop['short_lt200']}**, поэтому для области "
           "даны только границы точечной оценки: все короткие без связей / все со связями. Неразмеченный "
           "фрагмент не считается ни 0, ни 1."),
        "5. **Интервалы** — бутстрэп внутри страт, 10 000 повторов, зерно 1, перцентили 2,5–97,5 %.",
        "",
        "## Разметка по стратам (рамка)",
        "",
        "| Терминов | Размечено | Со связями | Доля | 95 % (Уилсон) | В рамке | Коротких |",
        "|---|---|---|---|---|---|---|",
    ]
    for b, v in by.items():
        L.append(f"| {b} | {v['labelled']} | {v['with_relations']} | {d(v['share'], 2)} | "
                 f"{d(v['ci'][0], 2)}–{d(v['ci'][1], 2)} | {v['population_frame']} | {v['short']}"
                 + (f" (со связями {v['short_with_relations']})" if v["short_with_relations"] is not None else "")
                 + " |")
    L += [
        "",
        "## Пороги 1–4",
        "",
        "Порог t: отсеиваются фрагменты, где терминов меньше t.",
        "",
        "| Порог | Совокупность | Отсеяно фрагментов | Экономия токенов | Сохраняется фрагментов со связями (полнота) | 95 % интервал | "
        "Теряется фрагментов со связями | Доля фрагментов со связями среди отсеянных | Отсеяно фрагментов со связями (оценка) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in (1, 2, 3, 4):
        f = frame[t]
        L.append(f"| {t} | рамка, {f['chunks']} | {f['filtered_chunks']} ({pct(f['filtered_share'])}) | "
                 f"{pct(f['token_saving'])} | {d(f['relation_recall'])} | {d(f['recall_ci_low'])}–{d(f['recall_ci_high'])} | "
                 f"{pct(f['lost_relation_share'])} | {pct(f['relation_share_among_filtered'])} | {f['missed_relations']} |")
        g = dom[t]
        if done:
            L.append(f"| {t} | вся область, {g['chunks']} | {g['filtered_chunks']} ({pct(g['filtered_share'])}) | "
                     f"{pct(g['token_saving'])} | {d(g['relation_recall'])} | {d(g['recall_ci_low'])}–{d(g['recall_ci_high'])} | "
                     f"{pct(g['lost_relation_share'])} | {pct(g['relation_share_among_filtered'])} | {g['missed_relations']} |")
        else:
            L.append(f"| {t} | вся область, {g['chunks']} | {g['filtered_chunks']} ({pct(g['filtered_share'])}) | "
                     f"{pct(g['token_saving'])} | границы {d(g['recall_ci_low'])}–{d(g['recall_ci_high'])} | — | — | — | — |")
    L += [
        "",
        "**Исходная оценка v16** (доли рамки перенесены на всю область) сохранена в "
        "`labeling_scores.json` → `filter_recall_v16_original`. Полнота: "
        + "; ".join(f"порог {t} — {d(v16[str(t)])}" for t in (1, 2, 3, 4))
        + ". Файлы v16 не менялись (`research/v16/results`).",
        "",
        "## Примеры потерянных связей (размеченные фрагменты, где терминов меньше 4)",
        "",
        "Связи записаны разметчиком в столбце «Какие связи».",
        "",
    ]
    for e in lab["filter_missed_examples"]:
        if e["relations"]:
            L.append(f"- терминов {e['terms']}, фрагмент {e['chunk_id']}: {e['relations']}")
    L += [
        "",
        "## Два разных показателя полноты",
        "",
        "| | Ретроспективная проверка (v15) | Ручная разметка (v16–v17) |",
        "|---|---|---|",
        "| Что считается | доля **связей уже построенного графа**, лежащих во фрагментах, которые порог 3 "
        "отсеял бы | доля **фрагментов, где по разметке есть геологическая связь**, которые порог 3 отсеял бы |",
        "| Знаменатель | связи графа (3 818) | фрагменты со связями (оценка по выборке) |",
        f"| Значение | 1,8 % | {pct(frame[3]['lost_relation_share'])} (рамка) |",
        "",
        "Эти числа не противоречат друг другу и не опровергают одно другое: у них разные знаменатели. Граф "
        "строился по фрагментам, где модель смогла извлечь и подтвердить связь. Во фрагментах с малым числом "
        "терминов таких связей мало, даже если сама связь в тексте есть. Поэтому ретроспективная проверка по "
        "графу показывает, сколько уже найденного потерял бы фильтр, но не измеряет пропуск всех связей, "
        "которые можно было бы извлечь. Это видно и по охвату: связи графа есть примерно в "
        f"{pct(lab['graph_coverage_of_relation_chunks']['point'])} фрагментов, где по разметке есть связи.",
        "",
    ]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "prefilter_sampling_audit.md").write_text("\n".join(L), "utf-8")
    print("ok")


MARK = "## Уточнения v17"


def relation_summary() -> None:
    """Дописать к сводке relation_audit.py раздел v17 (нижние границы, обход типизации)."""
    un = json.loads((RES / "untyped_53_summary.json").read_text("utf-8"))
    va = un["variants"]
    path = DOCS / "relation_pipeline_summary.md"
    base = path.read_text("utf-8").split(MARK)[0].rstrip()
    add = [
        "", "", MARK, "",
        "Определения всех чисел — в `relation_count_definitions.md`.",
        "",
        "- **22 218 и 8 479 — нижние границы**, причин три. (1) Журнал отклонений хранит не больше 40 строк на "
        "фрагмент. (2) Повторы внутри фрагмента отбрасываются ещё в `facts.check` (`seen`) и нигде не "
        "записываются; ограничение UNIQUE в таблице `facts` — лишь вторая защита, до него такие связи не доходят. (3) Разбор ответа модели прекращается после 20 принятых связей (`MAX_FACTS`); "
        "этот предел достигнут у 193 фрагментов. Потери по (2) и (3) по базе не восстановить.",
        f"- **Обход типизации.** Из {va['A']['relations']} связей графа типизированы {va['B']['relations']}. "
        f"Остальные {un['untyped_in_graph']} приняты только по повтору в другой статье: `kinds.todo` берёт лишь "
        f"связи с `votes ≥ 3`. Из них {un['repeat_only_from_untyped']} подтверждают только друг друга, "
        f"{un['repeat_same_work_versions']} — повтор между препринтом и статьёй одной работы. Подробно — в "
        "`graph_typing_invariant.md`.",
        f"- **Уникальные тройки.** {va['A']['unique_triples']} = {va['B']['unique_triples']} типизированных + "
        f"{va['A']['unique_triples'] - va['B']['unique_triples']} троек нетипизированных связей. У последних вместо "
        f"типа стоит слово статьи. В статье v17 указано {va['B']['unique_triples']}.",
        "",
    ]
    text = base + "\n".join(add)
    path.write_text(text, "utf-8")
    (RES / "relation_pipeline_summary.md").write_text(text, "utf-8")


if __name__ == "__main__":
    main()
    relation_summary()
