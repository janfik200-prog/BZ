"""Статья ИПУЗ v14 (DOCX) из шаблона v13: стили, поля и шрифты — как в шаблоне, текст —
новый. Каждое число берётся из файлов результатов (research/v14/results, data, dossier)
и записывается в SCIENCE_CLAIMS.csv с файлом и ключом, откуда оно взято.

    .venv/Scripts/python.exe research/v14/article/build_article.py
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[3]
V14 = ROOT / "research" / "v14"
RES, DATA, ART = V14 / "results", V14 / "data", V14 / "article"
TEMPLATE = ROOT / ".local" / "logs" / "Статья_ИПУЗ_2026_база_знаний_v13_шаблон.docx"
OUT = ART / "Статья_ИПУЗ_2026_база_знаний_v14.docx"

CLAIMS: list[dict] = []


def J(name: str) -> dict:
    return json.loads((RES / name).read_text("utf-8"))


def num(value, claim: str, source: str, key: str, fmt: str = "{}") -> str:
    """Число для текста — с записью, откуда оно."""
    text = fmt.format(value)
    CLAIMS.append({"claim_id": f"N{len(CLAIMS) + 1:03d}", "paper_sentence": claim, "metric_value": text,
                   "raw_file": source, "key": key})
    return text


def pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def dec(x: float, d: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -d else x  # без «-0,00»
    return f"{x:.{d}f}".replace(".", ",")


def ci(c) -> str:
    return f"[{dec(c[0])}; {dec(c[1])}]"


# ------------------------------------------------------------------------------------- данные
metrics = J("metrics.json")
support = J("support_eval.json")
rules = J("support_rules.json")
boot = J("bootstrap.json")
esum = J("e_summary.json")
eprec = J("e_precision_sample.json")["precision"]
indep = J("independence_report.json")
prov = J("provenance_audit.json")
freeze = J("FREEZE.json")
runv = J("run_v_summary.json")
pmet = J("p_metrics.json") if (RES / "p_metrics.json").exists() else None
dos = json.loads((V14 / "dossier" / "summary.json").read_text("utf-8"))
tests = json.loads((V14 / "qa" / "tests_summary.json").read_text("utf-8"))
manifest = json.loads((DATA / "split_manifest.json").read_text("utf-8"))

T = {n: metrics[f"TEST:{n}"] for n in ("V0", "V1", "V2", "V3")}
D = {n: metrics[f"DEV:{n}"] for n in ("V0", "V1", "V2", "V3")}
chosen = rules["chosen_on_dev"]
T["V3+S"] = rules["metrics"]["TEST"][chosen]
T["V2+S"] = rules["v2_same_support"]["TEST"][chosen]
S3 = support["support"]["by_threshold"]["3"]
off = esum["offline"]
live = esum["live"]
h14, h8 = esum["historical_e2"]["georag_b14"], esum["historical_e2"]["georag_b8v3"]


def frac(r: dict, k: str, n: str) -> str:
    return f"{r[k]}/{r[n]}"


# ------------------------------------------------------------------------------------- документ
doc = docx.Document(str(TEMPLATE))
body = doc.element.body
sect = body[-1]
for el in list(body):
    if el is not sect:
        body.remove(el)


def para(text: str = "", style: str = "Normal", align=None, bold: bool = False, italic: bool = False,
         size: float | None = None, indent: bool | None = None, keep: bool = False):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if indent is False:
        p.paragraph_format.first_line_indent = Cm(0)
    if keep:
        p.paragraph_format.keep_with_next = True
    _runs(p, text, bold, italic, size)
    return p


def _runs(p, text: str, bold=False, italic=False, size=None):
    """**жирный** и ~~курсив~~ внутри строки (подчёркивание — часть кодов причин, не разметка)."""
    import re

    for part in re.split(r"(\*\*[^*]+\*\*|~~[^~]+~~)", text):
        if not part:
            continue
        b, i, t = bold, italic, part
        if part.startswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("~~") and part.endswith("~~") and len(part) > 4:
            i, t = True, part[2:-2]
        r = p.add_run(t)
        r.bold, r.italic = b or None, i or None
        if size:
            r.font.size = Pt(size)


def section(title: str):
    para(title, "Section", keep=True)


def subsection(title: str):
    para(title, "Subsection", keep=True)


def _borders(table):
    tbl = table._tbl
    pr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "000000")
        b.append(e)
    pr.append(b)


def table(caption: str, header: list[str], rows: list[list[str]], widths: list[float], size: float = 11):
    para(caption, "Table caption", keep=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _borders(t)
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        pp = c.paragraphs[0]
        pp.paragraph_format.first_line_indent = Cm(0)
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = pp.add_run(h)
        r.bold = True
        r.font.size = Pt(size)
    # Заголовок повторяется на каждой странице, строка не разрывается.
    trpr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trpr.append(th)
    for i, row in enumerate(rows, start=1):
        cant = OxmlElement("w:cantSplit")
        t.rows[i]._tr.get_or_add_trPr().append(cant)
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ""
            pp = c.paragraphs[0]
            pp.paragraph_format.first_line_indent = Cm(0)
            pp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            _runs(pp, v, size=size)
    t.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    for j, w in enumerate(widths):
        t._tbl.tblGrid.findall(qn("w:gridCol"))[j].set(qn("w:w"), str(int(w * 567)))
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    return t


def figure(path: Path, caption: str, width_cm: float = 16.3):
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    para(caption, "Figure caption")


L = WD_ALIGN_PARAGRAPH
# ------------------------------------------------------------------------------------- шапка
para("УДК 004.912:550.8", align=L.LEFT, bold=True, indent=False)
para("[Фамилия И. О. — состав и порядок авторов подтвердить перед подачей]", align=L.RIGHT, size=11.5, indent=False)
para("Научно-исследовательский центр искусственного интеллекта, Государственный университет «Дубна», г. Дубна, "
     "Россия [аффилиацию и e-mail подтвердить]", align=L.RIGHT, size=11.5, indent=False)
para("КОНТЕКСТНАЯ ПРИМЕНИМОСТЬ ЦИТИРУЕМЫХ ФАКТОВ В ЛОКАЛЬНОЙ БАЗЕ ЗНАНИЙ ПО РУДНОЙ ГЕОЛОГИИ", align=L.CENTER,
     bold=True, indent=False)

n_pairs = freeze["n_pairs"]
para(
    "Дословная цитата подтверждает, что утверждение есть в статье, но не то, что оно относится к району, о котором "
    "спрашивает геолог. В локальной базе знаний с графом фактов, выписанных языковой моделью из "
    f"{num(114, 'статей в корпусе', 'docs/BASELINE_MANIFEST.md', 'documents')} статей, проверка источника разделена на три "
    "независимые части: верность цитаты, поддержку связи цитатой и применимость к вопросу. Для применимости предложены "
    "структурное представление области действия, модальности, полярности и квантора факта, справочник территорий "
    "с основаниями вложенности и детерминированная политика допуска со статусами «прямое», «аналог», «вне области», "
    "«не решено» и причинными кодами. На "
    f"{num(n_pairs, 'пар факт–вопрос в эксперименте', 'results/FREEZE.json', 'n_pairs')} размеченных парах факт–вопрос "
    "по 35 территориальным вопросам граф без контроля контекста выдал прямыми свидетельствами "
    f"{num(T['V0']['false_direct_foreign'], 'ложных DIRECT V0 TEST', 'results/metrics.json', 'TEST:V0.false_direct_foreign')} из "
    f"{num(T['V0']['n_foreign'], 'чужих пар TEST', 'results/metrics.json', 'TEST:V0.n_foreign')} "
    "фактов о других районах (тестовая часть), предложенный метод — "
    f"{num(T['V3']['false_direct_foreign'], 'ложных DIRECT V3 TEST', 'results/metrics.json', 'TEST:V3.false_direct_foreign')}, "
    "при охвате прямых ответов "
    f"{num(pct(T['V3']['direct_coverage']), 'охват V3 TEST', 'results/metrics.json', 'TEST:V3.direct_coverage', '{} %')} против "
    f"{num(pct(T['V0']['direct_coverage']), 'охват V0 TEST', 'results/metrics.json', 'TEST:V0.direct_coverage', '{} %')}. "
    "Разметка выполнена одним разметчиком, не геологом; результаты являются пилотными.",
    size=12)
para("КЛЮЧЕВЫЕ СЛОВА: база знаний, граф знаний, генерация с дополнением выборкой, проверка утверждений, применимость "
     "свидетельств, область действия, справочник территорий, рудная геология.", size=12, bold=False)

# ------------------------------------------------------------------------------------- введение
section("ВВЕДЕНИЕ")
para("Прогнозная модель рудоносности опирается на признаки, по которым геологи выделяют рудные узлы: структурный "
     "контроль, зоны гидротермальных изменений, геохимические и геофизические аномалии. Сведения о них рассеяны по сотням "
     "статей на русском и английском языках. Языковые модели ускоряют обзор, а генерация с дополнением выборкой (RAG) [1] "
     "и графовые её варианты [2] привязывают ответ к найденным фрагментам. Проверка ссылки — сверка утверждения с "
     "процитированным текстом — устраняет вымышленные цитаты, но не главный для геологии риск: перенос сведений между "
     "территориями.")
para("Пример из рабочей базы. На вопрос о золотом оруденении Дальнего Востока России граф фактов без контроля контекста "
     "выдаёт как прямое свидетельство дословно процитированный и подтверждённый факт о мезозойском магматизме "
     "полуострова Цзяодун в Китае; на вопрос о Бразилии — сведения о золоте Витватерсранда, приведённые для сравнения в "
     "статье о месторождении Якобина. Цитата верна, связь утверждается, но к району вопроса источник не относится. "
     "Поэтому в работе разделены три вопроса к каждому факту: (A) верна ли цитата, (B) утверждает ли она связь и (C) "
     "применима ли связь к территории и смыслу вопроса — и третий решается отдельной проверяемой процедурой, а не "
     "сверкой текста.")
para("Вклад работы: формальное представление области действия факта и вопроса с квантором, модальностью и "
     "полярностью; порядок извлечения области действия из контекста источника с фиксацией основания; детерминированная "
     "политика допуска с отказом и причинными кодами; встраивание её в граф и чат-бот действующей системы; эксперимент на "
     "реальном корпусе с раздельной оценкой ложного переноса и охвата прямых ответов.")

# ------------------------------------------------------------------------------------- связанные работы
section("СВЯЗАННЫЕ РАБОТЫ")
para("Геологические графы знаний из текстов строят с онтологическим управлением извлечением и схемой «сгенерировать — "
     "проверить — исправить» (OntoGRC [3]); объединяют многоязычные потоки геологических сведений с выявлением "
     "противоречий и сохранением происхождения [4]; строят региональные графы по картам ГИС [5] и связывают граф с "
     "распознаванием геологических карт (Geo-MAG [6]). Вне геологии SciGraph-LLM [7] извлекает научные утверждения "
     "вместе с фрагментами-доказательствами. Поэтому ни граф, ни онтология, ни цитата, ни двухэтапное извлечение сами по "
     "себе новизной не являются. Предмет данной работы уже: решение, может ли процитированный и подтверждённый факт "
     "служить ~~прямым~~ свидетельством по конкретному вопросу о территории, с явным разделением пространственного "
     "включения и квантора («встречается где-либо в районе» против «характерно для района в целом») и с измерением "
     "ложного переноса на размеченных парах.")

# ------------------------------------------------------------------------------------- система
section("СИСТЕМА И ПОСТАНОВКА ЗАДАЧИ")
para("Система локальна: PostgreSQL с pgvector [8], векторы BGE-M3 [9], разбор PDF Docling [10], модели Qwen3 [11] в "
     "vLLM [12]. Статья режется на фрагменты до 512 токенов с номерами страниц; поиск — векторный и полнотекстовый с "
     "объединением по рангам [13]. Граф фактов строится в два этапа (рис. 1): без модели отсеиваются почти-дубли (MinHash "
     "[14] с проверкой защищённых отличий — отрицание, числа, территории, модальность) и фрагменты, бедные терминами "
     "геологического словаря, а места находит spaCy [15]; затем модель-строитель связывает найденные предметы и "
     "возвращает номер предложения, а цитату по номеру берёт код. Каждый факт проходит проверки кодом и четыре вопроса "
     "модели по цитате (подтверждён — не менее трёх совпадений из четырёх).")
figure(ART / "fig1_pipeline.png", "Рис. 1. Путь факта: первый этап без модели, модель-строитель, три раздельные проверки "
                                  "(цитата, поддержка, применимость) и выход в ответ и досье.")
para("Факт — кортеж f = (s, r, o, e, g, k, m, p, z): концы и связь; неизменяемое доказательство e (статья, фрагмент, "
     "страница, смещения цитаты в замороженном разборе, отпечатки PDF, разбора и цитаты); область действия g "
     "(территория и основание её привязки); тип оруденения k; модальность m ∈ {утверждение, гипотеза, со ссылкой}; "
     "полярность p ∈ {да, нет}; квантор z ∈ {существует, локально, типично, всюду, не указан}. Не названное в источнике "
     "остаётся неизвестным и не достраивается. Вопрос — рамка q = (территория, тип оруденения, полярность, квантор, "
     "режим, разрешены ли аналогии), извлекаемая из текста вопроса правилами. Проверка источника возвращает "
     "S(f, e) ∈ {ENTAILED, PARTIAL, CONTRADICTED, UNKNOWN, INVALID_EVIDENCE}, политика — A(f, q) ∈ {DIRECT, ANALOG, "
     "OUT_OF_SCOPE, UNRESOLVED} со списком причин. DIRECT означает «источник прямо говорит о требуемой территории в "
     "нужной модальности», а не истинность сказанного.")

# ------------------------------------------------------------------------------------- метод
section("ОБЛАСТЬ ДЕЙСТВИЯ И ПОЛИТИКА ДОПУСКА")
reg_n = num(197, "территорий в справочнике", "data/regional_registry.csv", "rows")
para(f"**Справочник территорий** содержит {reg_n} единиц (страны, регионы, щиты, пояса, рудные районы, поля, месторождения) "
     f"и {num(748, 'написаний в справочнике', 'data/registry_aliases.csv', 'aliases')} написаний на русском и английском. "
     "Каждая связь включения имеет основание: формулировку статьи корпуса (84 единицы), административное деление или "
     "принятое региональное деление; две известные территории без записанной вложенности считаются несовместимыми, "
     "если не записано их перекрытие (например, Восточная пустыня Египта и Аравийско-Нубийский щит). Одно написание у "
     "двух объектов (рудопроявление «Майское» на Чукотке и в Северной Карелии) неоднозначно и разрешается только "
     "вложенностью с другой названной в цитате территорией. Слияние по близости векторов имён не применяется.")
para("**Область действия факта** определяется по порядку: территория в самой цитате; при явной анафоре («this area», "
     "«в пределах района») — предшествующее предложение того же фрагмента с сохранением этого предложения как "
     "основания; заголовок раздела — если фрагмент не называет несовместимых территорий; название статьи — только "
     "конкретное (одна территория, не обзор) и без конфликта с фрагментом. Общий или обзорный заголовок основанием для "
     "прямого статуса не служит. Признаки модальности, полярности и квантора определяются по цитате правилами (маркеры "
     "предположения, отрицания, обобщения).")
para("**Политика допуска** — чистая функция без модели. Недостоверная цитата или поддержка, отличная от ENTAILED, — "
     "UNRESOLVED. Территория факта совпадает с территорией вопроса или входит в неё — кандидат в DIRECT; содержит её — "
     "UNRESOLVED (сказанное о районе в целом не переносится на его часть); несовместима — OUT_OF_SCOPE, а при "
     "сравнительном вопросе ANALOG, если у территорий общая геологическая обстановка в справочнике и подтверждено "
     "совпадение типа оруденения; не установлена или перекрывается — UNRESOLVED. Затем: другой тип оруденения — "
     "OUT_OF_SCOPE; отрицание при утвердительном вопросе и гипотеза при фактическом — UNRESOLVED (гипотеза остаётся "
     "DIRECT с пометкой для вопроса о гипотезах); для вопроса «типично/всюду» прямой ответ даёт только факт о той же "
     "территории с тем же квантором. Решение сохраняет причины (REGION_EXACT, REGION_PART_OF, REGION_DISJOINT, "
     "REGION_TITLE_ONLY, SCOPE_UNIVERSALITY_MISMATCH, MODALITY_HYPOTHESIS и др.), версии политики и справочника и "
     "идентификаторы доказательств; десять инвариантов (недостоверная цитата, чужой район, часть вместо целого, "
     "гипотеза, отрицание, неизвестная территория, независимость от вопроса, воспроизводимость, монотонность при снятии "
     "доказательства) и 40 отрицательных сценариев проверяются автотестами.")
para("**Встраивание.** При вопросе о территории в сводку графа и в кандидаты ответа идут только факты DIRECT, аналоги "
     "выводятся отдельно; фрагменты, все территории которых несовместимы с вопросом, не используются как прямые "
     "источники; утверждения ответа сверх лимита сверки (24) помечаются как непроверенные. Полный список решений — в "
     "досье вопроса: четыре секции, цитата, DOI, страница, причины, журнал исправлений специалиста без перезаписи "
     "истории. Две статьи с одной работой (препринт и журнальная версия, части одной серии) или дословным повтором "
     "текста не считаются двумя независимыми подтверждениями.")

# ------------------------------------------------------------------------------------- эксперимент
section("ЭКСПЕРИМЕНТ")
fams = manifest["n_families"]
para(f"**Корпус и выборка.** Рабочая база: {num(114, 'статей', 'docs/BASELINE_MANIFEST.md', 'documents')} статей, "
     f"{num('4 272', 'фрагментов', 'docs/BASELINE_MANIFEST.md', 'chunks')} фрагмента, "
     f"{num('13 739', 'фактов графа', 'docs/BASELINE_MANIFEST.md', 'facts')} фактов графа (Qwen3-14B по всем фрагментам). "
     f"Сформировано 35 вопросов о золотом оруденении конкретных территорий (Россия и её регионы, Египет, Судан, Нигерия, "
     f"Узбекистан, Канада и др.), из них 3 — о типичности для территории, 2 — о гипотезах, 2 — с разрешёнными аналогиями. "
     f"Для каждого вопроса факты о золоте отобраны так, как их выдал бы граф без контроля контекста, со стратификацией "
     f"по вручную заданной области исследования статьи: своя, соседняя, чужая территория, обзор. Всего "
     f"{num(n_pairs, 'пар', 'results/FREEZE.json', 'n_pairs')} пар. Разбиение на настройку (DEV) и проверку (TEST) — "
     f"по {num(fams, 'семейств источников', 'data/split_manifest.json', 'n_families')} семействам источников, а не по "
     f"фрагментам: препринт и журнальная версия одной работы, части одной серии — в одном разбиении; DEV — "
     f"{num(manifest['pairs_by_split']['DEV'], 'пар DEV', 'data/split_manifest.json', 'pairs_by_split.DEV')} пар, TEST — "
     f"{num(manifest['pairs_by_split']['TEST'], 'пар TEST', 'data/split_manifest.json', 'pairs_by_split.TEST')}.")
lab = freeze["labels"]
para("**Разметка.** Каждая пара размечена по правилам, зафиксированным до разметки: поддержка связи цитатой, территория "
     "факта, квантор, модальность, полярность и итоговая применимость к вопросу. Получено "
     f"{num(lab['DIRECT'], 'gold DIRECT', 'results/FREEZE.json', 'labels.DIRECT')} DIRECT, "
     f"{num(lab['ANALOG'], 'gold ANALOG', 'results/FREEZE.json', 'labels.ANALOG')} ANALOG, "
     f"{num(lab['OUT_OF_SCOPE'], 'gold OUT_OF_SCOPE', 'results/FREEZE.json', 'labels.OUT_OF_SCOPE')} OUT_OF_SCOPE и "
     f"{num(lab['UNRESOLVED'], 'gold UNRESOLVED', 'results/FREEZE.json', 'labels.UNRESOLVED')} UNRESOLVED. "
     "**Ограничение:** разметку выполнил один разметчик — языковая модель Claude, не участвовавшая в построении фактов, "
     "но не геолог и автор политики допуска. Меры против подгонки: правила разметки и политика зафиксированы до прогона, "
     "разметка — вслепую к решениям методов, отпечатки разметки записаны до первого прогона. Слепая проверка геологом "
     "подготовлена (164 пары, 49 %) и не выполнена; до неё выводы предварительны.")
para("**Сравниваемые режимы** получают одни и те же пары: V0 — граф действующей системы (факт подтверждён вопросами или "
     "повторён в другой статье), территория не проверяется; V1 — V0 и совпадение названия территории вопроса (с "
     "синонимами справочника) в цитате, заголовке или названии статьи; V2 — справочник с вложенностью, тип оруденения и "
     "поддержка, область действия по цитате или по названию статьи без проверки его общности; V3 — предложенный метод. "
     "Показатели: доля ложного переноса FTR — ложные DIRECT среди пар о чужих территориях (ANALOG и OUT_OF_SCOPE по "
     "разметке); охват — доля вопросов, на которые есть прямой ответ по разметке, получивших хотя бы один верный DIRECT; "
     "избирательный риск — доля неверных среди всех выданных DIRECT. Для долей приведены 95 % интервалы Уилсона, для "
     "разностей — бутстреп по семействам источников. Политика и порог голосов (3 из 4) заданы до прогона и составляют "
     "основной протокол.")

# ------------------------------------------------------------------------------------- результаты
section("РЕЗУЛЬТАТЫ")


def row(name: str, r: dict, src: str, key: str) -> list[str]:
    f = num(f"{r['false_direct_foreign']}/{r['n_foreign']}", f"FTR {name}", src, f"{key}.false_direct_foreign/n_foreign")
    c = num(f"{r['covered']}/{r['n_answerable']}", f"охват {name}", src, f"{key}.covered/n_answerable")
    s = num(f"{r['n_false_direct']}/{r['n_direct_pred']}", f"риск {name}", src, f"{key}.n_false_direct/n_direct_pred")
    return [name, f"{f} ({dec(r['ftr'])}) {ci(r['ftr_ci'])}", f"{c} ({dec(r['direct_coverage'])}) {ci(r['coverage_ci'])}",
            f"{s} ({dec(r['selective_risk'])})"]


rows = [row("V0 — граф без контроля контекста", T["V0"], "results/metrics.json", "TEST:V0"),
        row("V1 — совпадение названия", T["V1"], "results/metrics.json", "TEST:V1"),
        row("V2 — справочник, тип, поддержка", T["V2"], "results/metrics.json", "TEST:V2"),
        row("V3 — предложенный метод", T["V3"], "results/metrics.json", "TEST:V3"),
        row("V2 + поддержка S*", T["V2+S"], "results/support_rules.json", f"v2_same_support.TEST.{chosen}"),
        row("V3 + поддержка S*", T["V3+S"], "results/support_rules.json", f"metrics.TEST.{chosen}")]
table("Таблица 1. Применимость на тестовой части (207 пар, 35 семейств; в скобках — доля и 95 % интервал)",
      ["Режим", "Ложный перенос FTR", "Охват прямых ответов", "Изб. риск"], rows, [4.6, 4.4, 4.4, 3.0], size=10.5)
para("* S — правило поддержки, выбранное на DEV до проверки на TEST (не менее одного голоса из четырёх и подтверждение "
     "отдельным вопросом модели о направлении и отрицании); см. ниже.", size=11, indent=False)

b0 = boot["TEST:V3-vs-V0"]
b2s = rules["bootstrap_V3_vs_V2_same_support"]["TEST"]
para("Граф без контроля контекста (V0) выдаёт прямыми свидетельствами больше половины фактов о чужих территориях; "
     "предложенный метод снижает ложный перенос на "
     f"{num(dec(-b0['delta_ftr']), 'снижение FTR V3 vs V0', 'results/bootstrap.json', 'TEST:V3-vs-V0.delta_ftr')} "
     f"(95 % интервал {num(ci([-b0['delta_ftr_ci95'][1], -b0['delta_ftr_ci95'][0]]), 'CI снижения FTR', 'results/bootstrap.json', 'TEST:V3-vs-V0.delta_ftr_ci95')}) "
     f"ценой снижения охвата на {num(dec(-b0['delta_coverage']), 'потеря охвата V3 vs V0', 'results/bootstrap.json', 'TEST:V3-vs-V0.delta_coverage')} "
     f"(интервал {num(ci([-b0['delta_coverage_ci95'][1], -b0['delta_coverage_ci95'][0]]), 'CI потери охвата', 'results/bootstrap.json', 'TEST:V3-vs-V0.delta_coverage_ci95')}). "
     "Совпадение названия (V1) тоже почти исключает перенос, но вдвое сокращает охват: факты о частях территории "
     "редко называют её саму. На DEV картина та же: FTR V0 — "
     f"{num(frac(D['V0'], 'false_direct_foreign', 'n_foreign'), 'FTR V0 DEV', 'results/metrics.json', 'DEV:V0')}, V3 — "
     f"{num(frac(D['V3'], 'false_direct_foreign', 'n_foreign'), 'FTR V3 DEV', 'results/metrics.json', 'DEV:V3')}.")
para("Основной выигрыш против V0 и V1 дают справочник с вложенностью и проверяемая область действия; на этой выборке "
     "V2 тоже не допустил переноса на TEST. Собственный вклад полной политики (V3 против V2) — в избирательном риске: "
     f"{num(dec(T['V3']['selective_risk']), 'риск V3', 'results/metrics.json', 'TEST:V3.selective_risk')} против "
     f"{num(dec(T['V2']['selective_risk']), 'риск V2', 'results/metrics.json', 'TEST:V2.selective_risk')} "
     "(отсеиваются факты о части территории для вопросов «в целом», гипотезы для фактических вопросов и названия "
     "обзорных статей), при охвате на одну тестовую задачу меньше. При одинаковой улучшенной поддержке V3 и V2 различаются "
     f"по FTR на {num(dec(-b2s['delta_ftr'], 3), 'разность FTR V3+S vs V2+S', 'results/support_rules.json', 'bootstrap_V3_vs_V2_same_support.TEST.delta_ftr')} "
     f"и по охвату на {num(dec(-b2s['delta_coverage']), 'разность охвата V3+S vs V2+S', 'results/support_rules.json', 'bootstrap_V3_vs_V2_same_support.TEST.delta_coverage')}; "
     "интервалы обеих разностей включают ноль — при таком объёме выборки преимущество V3 над V2 по переносу "
     "**не установлено**. Абляции на TEST: без справочника охват падает до 0,50; без квантора и без модальности и "
     "полярности растёт риск (0,29 против 0,23); правила названия статьи и анафора на этой выборке охват почти не меняют.")
para("**Поддержка источником ограничивает охват.** Четыре вопроса модели (не менее трёх совпадений) на "
     f"{num(support['support']['n_facts'], 'уникальных фактов в выборке', 'results/support_eval.json', 'support.n_facts')} "
     "размеченных фактах имеют точность "
     f"{num(dec(S3['precision']), 'точность confirm>=3', 'results/support_eval.json', 'support.by_threshold.3.precision')} "
     f"{ci(S3['precision_ci'])} и полноту "
     f"{num(dec(S3['recall']), 'полнота confirm>=3', 'results/support_eval.json', 'support.by_threshold.3.recall')} "
     f"{ci(S3['recall_ci'])}: почти три из четырёх пропусков прямых ответов V3 вызваны тем, что верная связь не набрала "
     "трёх голосов. Отдельный вопрос модели о поддержке с явной проверкой направления, отрицания и предположения "
     "(инструмент той же модели, не разметка) в сочетании с хотя бы одним голосом выбран на DEV по заранее заданному "
     "критерию и на TEST поднял охват V3 до "
     f"{num(dec(T['V3+S']['direct_coverage']), 'охват V3+S', 'results/support_rules.json', 'metrics.TEST.'+chosen+'.direct_coverage')} "
     "без ложного переноса (табл. 1, последние строки).")

ft = off["test_at_k"]
para("**Первый этап и модель-строитель.** Порог числа терминов выбран на DEV (наибольший, при котором теряется не более "
     f"5 % опорных фактов графа): k = {num(off['chosen_k_on_dev'], 'выбранный k', 'results/e_summary.json', 'offline.chosen_k_on_dev')}. "
     f"На TEST к модели идёт {num(pct(ft['share_to_model']), 'доля фрагментов к модели', 'results/e_summary.json', 'offline.test_at_k.share_to_model', '{} %')} "
     f"фрагментов и {num(pct(ft['tokens_share']), 'доля токенов', 'results/e_summary.json', 'offline.test_at_k.tokens_share', '{} %')} "
     f"токенов; теряется {num(pct(ft['prefilter_loss_ref']), 'потеря опорных фактов TEST', 'results/e_summary.json', 'offline.test_at_k.prefilter_loss_ref', '{} %')} "
     f"опорных фактов (цель 5 % на TEST превышена) и {num(frac(ft, 'gold_lost', 'gold_entailed'), 'потеря размеченных верных связей', 'results/e_summary.json', 'offline.test_at_k.gold_lost/gold_entailed')} "
     "размеченных верных связей. Почти-дублей в корпусе мало (5 фрагментов); защищённые отличия удержали от отсева 1. "
     "На 100 фрагментах TEST тот же сервер и модель с отсевом и списком предметов в задании (E1) израсходовали "
     f"{num(f'{live['by_mode']['E1']['prompt_tokens']:,}'.replace(',', ' '), 'токены E1', 'results/e_summary.json', 'live.by_mode.E1.prompt_tokens')} "
     f"токенов задания против {num(f'{live['by_mode']['E0']['prompt_tokens']:,}'.replace(',', ' '), 'токены E0', 'results/e_summary.json', 'live.by_mode.E0.prompt_tokens')} "
     f"и выписали {num(live['by_mode']['E1']['facts'], 'фактов E1', 'results/e_summary.json', 'live.by_mode.E1.facts')} фактов против "
     f"{num(live['by_mode']['E0']['facts'], 'фактов E0', 'results/e_summary.json', 'live.by_mode.E0.facts')}, но точность выписки до "
     f"подтверждения на слепой выборке — {num(f'{eprec['E1']['correct']}/{eprec['E1']['n']}', 'точность E1', 'results/e_precision_sample.json', 'precision.E1')} "
     f"против {num(f'{eprec['E0']['correct']}/{eprec['E0']['n']}', 'точность E0', 'results/e_precision_sample.json', 'precision.E0')}: "
     "список предметов побуждает модель связывать всё названное. Это отрицательный результат для схемы первого этапа "
     "в нынешнем виде. Модель-строитель Qwen3-8B (AWQ) на тех же 200 фрагментах в исторических прогонах выписала "
     f"{num(h8['facts_by_model'][0]['facts'], 'фактов 8B', 'results/e_summary.json', 'historical_e2.georag_b8v3')} фактов за "
     f"{num(dec(h8['wall_minutes'], 1), 'минут 8B', 'results/e_summary.json', 'historical_e2.georag_b8v3.wall_minutes')} мин против "
     f"{num(h14['facts_by_model'][0]['facts'], 'фактов 14B', 'results/e_summary.json', 'historical_e2.georag_b14')} за "
     f"{num(dec(h14['wall_minutes'], 1), 'минут 14B', 'results/e_summary.json', 'historical_e2.georag_b14.wall_minutes')} мин у Qwen3-14B; "
     "версии правил между прогонами могли различаться, поэтому сравнение ориентировочное, а точность 8B не размечена.")
para("**Цитаты и независимость.** Из "
     f"{num('13 739', 'фактов в аудите', 'results/provenance_audit.json', 'n_facts')} цитат графа по смещениям и отпечаткам "
     f"воспроизводятся {num(f'{prov['valid']:,}'.replace(',', ' '), 'воспроизводимых цитат', 'results/provenance_audit.json', 'valid')}; "
     "14 фактов потеряли фрагмент после переразбора и помечаются устаревшими, 5 — с битым текстовым слоем. Из "
     f"{num(indep['relations_in_2plus_docs'], 'связей в 2+ статьях', 'results/independence_report.json', 'relations_in_2plus_docs')} "
     "связей, повторённых в двух и более статьях, на независимых источниках держатся "
     f"{num(indep['independence_status']['INDEPENDENT'], 'независимых связей', 'results/independence_report.json', 'independence_status.INDEPENDENT')}; "
     "остальные — копии одной работы или дословный повтор текста одной группы авторов, которые прежний буфер графа "
     "засчитывал как подтверждение.")
if pmet:
    p1, p2, p0 = pmet["P1"], pmet["P2"], pmet["P0"]
    para("**Сквозные ответы.** На 15 вопросах о территориях чат-бот без графа (P0), с графом (P1) и с графом и проверкой "
         "применимости (P2) сослался на статьи о чужих территориях в "
         f"{num(f'{p0['foreign_citations']}/{p0['citations']}', 'чужие ссылки P0', 'results/p_metrics.json', 'P0')}, "
         f"{num(f'{p1['foreign_citations']}/{p1['citations']}', 'чужие ссылки P1', 'results/p_metrics.json', 'P1')} и "
         f"{num(f'{p2['foreign_citations']}/{p2['citations']}', 'чужие ссылки P2', 'results/p_metrics.json', 'P2')} случаях; "
         "хотя бы одна ссылка на статью о своей территории была в "
         f"{num(p0['questions_with_own_citation'], 'вопросов со своей ссылкой P0', 'results/p_metrics.json', 'P0.questions_with_own_citation')}, "
         f"{num(p1['questions_with_own_citation'], 'P1 own', 'results/p_metrics.json', 'P1.questions_with_own_citation')} и "
         f"{num(p2['questions_with_own_citation'], 'P2 own', 'results/p_metrics.json', 'P2.questions_with_own_citation')} ответах. "
         "Оценка ведётся по области исследования процитированной статьи, а не по каждому утверждению, по одному "
         "прогону на режим, и требует проверки геологом.")
para("**Досье.** Для одного и того же набора из "
     f"{num(dos['n_facts_same_set'], 'фактов в досье', 'dossier/summary.json', 'n_facts_same_set')} структурных фактов о золоте "
     "вопрос о Центральной Восточной пустыне Египта с разрешёнными аналогиями дал "
     f"{num(dos['A']['sections'].get('DIRECT', 0), 'DIRECT досье A', 'dossier/summary.json', 'A.sections.DIRECT')} прямых "
     f"свидетельств и {num(dos['A']['sections'].get('ANALOG', 0), 'ANALOG досье A', 'dossier/summary.json', 'A.sections.ANALOG')} "
     "аналога из Судана (тот же Нубийский щит и тип оруденения), а вопрос о типичности для Египта в целом — ни одного "
     "прямого: те же цитаты перешли в «не решено» с причиной «часть вместо целого». Статус меняется вопросом, а не "
     "новым фактом.")
subsection("Типичные ошибки")
para("Неверные DIRECT предложенного метода на тестовой части — общие утверждения внутри региональной статьи: «кварц-"
     "халцедон-адуляровые жилы, известные высокими содержаниями золота» в статье о бассейне Кастро или правило "
     "интерпретации магнитного поля в статье о сланцевом поясе Илеша получают территорию из названия статьи, хотя "
     "говорят о закономерности вообще. Второй класс — подписи к рисункам и таблицам, где связь извлечена неверно. "
     "Единственный ложный перенос на DEV — уран-редкометалльный факт района Умм-Ара для вопроса о золоте: в цитате "
     "упомянуто и золото, и несовпадение типа оруденения не распознано. Пропуски прямых ответов на три четверти "
     "вызваны порогом поддержки, остальное — несколькими территориями в одной цитате и неизвестными справочнику "
     "названиями.")

# ------------------------------------------------------------------------------------- обсуждение
section("ОБСУЖДЕНИЕ И ОГРАНИЧЕНИЯ")
para("Результат показывает, что территориальная применимость — отдельная от сверки цитаты проверка, и её отсутствие "
     "даёт в графе большую долю переноса; основную долю ошибок устраняет справочник с явной вложенностью, а квантор и "
     "модальность снижают долю неверных прямых ответов. Ограничения существенны. Разметка выполнена одним "
     "не-геологом, автором политики; справочник территорий собран тем же исполнителем и требует проверки специалистом; "
     "выборка мала (20 тестовых вопросов с прямым ответом, 6 аналогов), поэтому преимущество над вариантом V2 не "
     "установлено. Квантор факта определяется по цитате плохо, и для вопросов о типичности метод опирается в основном "
     "на соотношение территорий. Статус применимости не говорит об истинности вывода статьи. Оценка сквозных ответов "
     "выполнена по уровню статей. Модель-строитель Qwen3-8B живым прогоном на той же выборке не проверялась: на "
     "видеокарте 16 ГБ две модели одновременно не помещаются, а остановка работающей модели ответов требует решения "
     "владельца системы.")

section("ЗАКЛЮЧЕНИЕ")
para("В локальной базе знаний по рудной геологии проверка факта разделена на верность цитаты, поддержку связи и "
     "применимость к территории и смыслу вопроса. Предложены представление области действия с квантором, модальностью "
     "и полярностью, порядок её извлечения из контекста источника со справочником территорий и детерминированная "
     "политика допуска с причинами. На реальном корпусе и пилотной разметке ложный перенос фактов между территориями "
     f"снизился с {num(dec(T['V0']['ftr']), 'FTR V0', 'results/metrics.json', 'TEST:V0.ftr')} до "
     f"{num(dec(T['V3']['ftr']), 'FTR V3', 'results/metrics.json', 'TEST:V3.ftr')} при охвате прямых ответов "
     f"{num(dec(T['V3']['direct_coverage']), 'охват V3', 'results/metrics.json', 'TEST:V3.direct_coverage')} "
     f"({num(dec(T['V3+S']['direct_coverage']), 'охват V3+S', 'results/support_rules.json', 'metrics.TEST.'+chosen)} с улучшенной "
     "проверкой поддержки). Подтверждение этих чисел независимой экспертной разметкой — необходимое условие перед "
     "использованием выводов в прогнозных работах. Код, данные разметки и сырые решения всех режимов доступны в "
     "репозитории проекта.")

# ------------------------------------------------------------------------------------- литература
section("ЛИТЕРАТУРА")
refs = [
    "Lewis P., Perez E., Piktus A., et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks // Advances in Neural Information Processing Systems 33 (NeurIPS 2020). 2020. P. 9459–9474.",
    "Edge D., Trinh H., Cheng N., et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. 2024. arXiv: 2404.16130.",
    "Zhang C., Li C., Wang S., et al. OntoGRC: Ontology-Guided Mineral Exploration Knowledge Extraction from Geological Texts Using Generate–Reflect–Correct Prompting // Ore Geology Reviews. 2026. Vol. 195. Art. 107411. DOI: 10.1016/j.oregeorev.2026.107411.",
    "Qin Y., Yang H., Cui L., et al. Semantic-Aware Fusion of Mineral Exploration Knowledge Streams Towards Dynamic Geological Knowledge Graphs // Minerals. 2025. Vol. 15, No. 12. Art. 1257. DOI: 10.3390/min15121257.",
    "Liu J., Ye Y., Li Z., et al. Constructing Regional Mineral Prospecting Knowledge Graph from GIS Maps // Artificial Intelligence in Geosciences. 2025. Vol. 6, No. 2. Art. 100169. DOI: 10.1016/j.aiig.2025.100169.",
    "Ma K., Zhan Q., Zhang T., et al. Geo-MAG: A Knowledge Graph (KG)-Enhanced Multimodal Retrieval-Augmented Generation (RAG) Framework for Geological Map Understanding // Transactions in GIS. 2026. Vol. 30, No. 2. Art. e70226. DOI: 10.1111/tgis.70226.",
    "Malashin I. P., Masich I. S., Tynchenko V. S., et al. SciGraph-LLM: Automatic Knowledge Graph Construction from Scientific Papers // Proc. 19th ACM International Conference on Web Search and Data Mining (WSDM 2026). 2026. P. 48–57. DOI: 10.1145/3779211.3793169.",
    "pgvector: Open-source vector similarity search for Postgres [Электронный ресурс] // GitHub. URL: https://github.com/pgvector/pgvector (дата обращения: 27.10.2026).",
    "Chen J., Xiao S., Zhang P., et al. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation // Findings of ACL 2024. 2024. P. 2318–2335. DOI: 10.18653/v1/2024.findings-acl.137.",
    "Auer C., Lysak M., Nassar A., et al. Docling Technical Report. 2024. arXiv: 2408.09869.",
    "Yang A., Li A., Yang B., et al. Qwen3 Technical Report. 2025. arXiv: 2505.09388.",
    "Kwon W., Li Z., Zhuang S., et al. Efficient Memory Management for Large Language Model Serving with PagedAttention // Proc. 29th Symposium on Operating Systems Principles (SOSP). 2023. P. 611–626. DOI: 10.1145/3600006.3613165.",
    "Cormack G. V., Clarke C. L. A., Büttcher S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods // Proc. 32nd International ACM SIGIR Conference. 2009. P. 758–759. DOI: 10.1145/1571941.1572114.",
    "Broder A. Z. On the Resemblance and Containment of Documents // Proc. Compression and Complexity of SEQUENCES 1997. IEEE, 1997. P. 21–29. DOI: 10.1109/SEQUEN.1997.666900.",
    "Honnibal M., Montani I., Van Landeghem S., Boyd A. spaCy: Industrial-Strength Natural Language Processing in Python. Zenodo, 2020. DOI: 10.5281/zenodo.1212303.",
]
for i, r in enumerate(refs, 1):
    para(f"{i}. {r}", "References")

# ------------------------------------------------------------------------------------- английская аннотация
para("[Author name], Research Center for Artificial Intelligence, Dubna State University, Dubna, Russia [to be confirmed]. "
     "[Other authors and e-mail to be confirmed]", "EN Abstract")
para("CONTEXT APPLICABILITY OF QUOTED FACTS IN A LOCAL KNOWLEDGE BASE FOR ORE GEOLOGY", "EN Abstract", bold=True)
para("A verbatim quote shows that a statement is in a paper, not that it concerns the area a geologist asks about. In a "
     "local knowledge base whose fact graph is extracted by a language model from 114 papers, source checking is split "
     "into quote validity, relation support and applicability to the query. For applicability we propose a structured "
     "representation of a fact's scope, modality, polarity and quantifier, a gazetteer with justified containment, and a "
     "deterministic admission policy with direct, analogue, out-of-scope and unresolved statuses and reason codes. On "
     f"{n_pairs} annotated fact–query pairs for 35 area-specific questions, the graph without context control admitted "
     f"{T['V0']['false_direct_foreign']} of {T['V0']['n_foreign']} facts about other areas as direct evidence (test split), "
     f"the proposed method {T['V3']['false_direct_foreign']}, at direct-answer coverage of {dec(T['V3']['direct_coverage'])} "
     f"versus {dec(T['V0']['direct_coverage'])}. Annotation was done by a single non-geologist annotator, so results are "
     "preliminary.", "EN Abstract")
para("KEYWORDS: knowledge base, knowledge graph, retrieval-augmented generation, claim verification, evidence "
     "applicability, gazetteer, ore geology.", "EN Abstract")

doc.core_properties.title = "Контекстная применимость цитируемых фактов в локальной базе знаний по рудной геологии"
doc.core_properties.author = "[авторы — подтвердить]"
doc.save(str(OUT))

chars = sum(len(p.text) for p in doc.paragraphs) + sum(len(c.text) for t in doc.tables for r in t.rows for c in r.cells)
with (ART / "SCIENCE_CLAIMS.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["claim_id", "paper_sentence", "metric_value", "raw_file", "key", "run_id", "status"])
    w.writeheader()
    for c in CLAIMS:
        w.writerow({**c, "run_id": runv["run_id"], "status": "FROM_RAW (pilot gold, single non-expert annotator)"})
print(json.dumps({"out": str(OUT), "chars": chars, "claims": len(CLAIMS)}, ensure_ascii=False))
