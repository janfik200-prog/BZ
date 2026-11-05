"""Сравнение промтов ответа на одних и тех же фрагментах — без нового поиска.

Прогон report сохраняет на листе «Фрагменты» всё, что модель видела при ответе. replay берёт
эту таблицу и по каждому вопросу заново пишет ответ разными промтами (GEORAG_PROMPT: «full» —
полный, «short» — короткий с правилами по случаю) на тех же фрагментах, с той же обработкой
ответа: отказ и второй заход, ссылки, сверка утверждений с фрагментами. Так видно, что меняет
сам промт, а не поиск, который от прогона к прогону находит немного разное.

    python georag.py replay .local\\reports\\вопросы_все-….xlsx
    python georag.py replay таблица.xlsx --only m01,f04 --prompts full,short

Таблица — в .local/reports (внутри папки проекта): ответы базы наружу не уходят.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .report import REPORTS

PROMPTS = ("full", "short")
LABELS = {"full": "полный промт", "short": "короткий промт"}
# Для сравнения промтов — без случайности: при температуре чата (0,1) один и тот же вопрос на
# тех же фрагментах давал от прогона к прогону разные ответы (f17, 05.11.2026), и по одному-двум
# изменившимся ответам из 78 нельзя было отличить действие промта от шума.
TEMPERATURE = 0.0


def system_prompt(style: str) -> str:
    """Точный текст системного промта, который уходит модели."""
    from .messages import SHORT_SYSTEM, SYSTEM

    return SHORT_SYSTEM if style == "short" else SYSTEM


def fingerprint(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def rule_names(question: str, sources: list[dict[str, Any]], style: str) -> str:
    """Какие правила «по случаю» короткий промт добавил к этому вопросу."""
    if style != "short":
        return ""
    from . import messages as M

    names = {M.FORMULA_RULE: "формулы", M.SUPERLATIVE_RULE: "«основной/лучший»",
             M.GENERAL_RULE: "общий вопрос"}
    return ", ".join(names.get(r, "другое") for r in M.situational_rules(question, sources))


def _s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def read_report(path: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Ответы и фрагменты из таблицы прогона report (листы «Ответы» и «Фрагменты»)."""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = {ws.title: list(ws.iter_rows(values_only=True)) for ws in wb.worksheets}
    finally:
        wb.close()
    if "Ответы" not in sheets or "Фрагменты" not in sheets:
        raise ValueError("в таблице нет листов «Ответы» и «Фрагменты» — это не таблица report")
    head = sheets["Ответы"][0]
    answers = [dict(zip(head, r)) for r in sheets["Ответы"][1:] if r and r[0]]
    fhead = sheets["Фрагменты"][0]
    frags: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in sheets["Фрагменты"][1:]:
        if r and r[0]:
            row = dict(zip(fhead, r))
            frags[(_s(row.get("№ вопроса")), _s(row.get("Словарь")))].append(row)
    return answers, frags


def _pages(value: Any) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", _s(value))][:6]


def to_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки листа «Фрагменты» → источники ответа, с прежними номерами [n]."""
    out = []
    for i, r in enumerate(sorted(rows, key=lambda r: int(re.sub(r"\D", "", _s(r.get("[n]"))) or 0))):
        n = int(re.sub(r"\D", "", _s(r.get("[n]"))) or i + 1)
        title = _s(r.get("Название статьи / отчёта"))
        dictionary = not _s(r.get("Файл")) and title.lower().startswith("геологический словарь")
        year = _s(r.get("Год"))
        out.append({
            "n": n,
            "doc_id": _s(r.get("Файл")) or title or f"фрагмент-{n}",
            "ord": i,
            "title": title,
            "year": int(year) if year.isdigit() else None,
            "journal": None,
            "authors": [],
            "pages": _pages(r.get("Страницы")),
            "headings": [_s(r.get("Раздел"))] if _s(r.get("Раздел")) else [],
            "url": "",
            "text": _s(r.get("Текст фрагмента")),
            "similarity": None,
            "found_by": "словарь" if dictionary else "",
            "part": None,
            **({"kind": "словарь"} if dictionary else {}),
        })
    return out


def answer_on(question: str, sources: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    """Ответ модели на готовых фрагментах — той же дорогой, что в чате (answer._answer_from):
    отказ → второй заход с напоминанием, ссылки, сверка утверждений."""
    from .answer import _answer_from, stream_llm
    from .messages import fit_budget
    from .plan import Part, Plan

    plan = Plan([Part(question, [])])
    # Как в чате: то, что не влезает в окно модели, отбрасывается с конца (слабейшие
    # фрагменты). Без этого длинный набор (m11, 05.11.2026) упирался в лимит контекста.
    articles = [s for s in sources if s.get("kind") != "словарь"]
    glossary = [s for s in sources if s.get("kind") == "словарь"]
    total = len(articles)
    import inspect

    # tail — статьи словаря, которые не урезаются; в копиях без этого параметра — без него.
    more = {"tail": glossary} if "tail" in inspect.signature(fit_budget).parameters else {}
    # Окно — по полному (длинному) промту для обоих: иначе короткому влезало больше фрагментов
    # и входы различались (m11: 13 против 15, 05.11.2026).
    articles, _ = fit_budget(question, articles, [], replace(settings, prompt="full"), plan, **more)
    sources = articles + glossary
    started = time.monotonic()
    text, used, refused = "", [], True
    for insist in (False, True):
        text = ""
        gen = _answer_from(stream_llm, question, sources, [], plan, [], settings, {}, started, insist)
        try:
            while True:
                ev = next(gen)
                kind = ev.get("type")
                if kind == "token":
                    text += ev["text"]
                elif kind == "revised":
                    text = ev["text"]
                elif kind == "done":
                    used = ev.get("used") or []
                elif kind == "error":
                    text = f"[ошибка] {ev.get('error')}"
        except StopIteration as stop:
            refused = not stop.value
        if not refused:
            break
    return {
        "answer": text.strip() or "В базе знаний нет ответа на этот вопрос.",
        "used": ", ".join(f"[{n}]" for n in used),
        "seconds": round(time.monotonic() - started, 1),
        "rules": rule_names(question, sources, settings.prompt),
        "window": f"{len(articles)} из {total}" + (f" + словарь {len(glossary)}" if glossary else ""),
    }


def write(rows: list[dict[str, Any]], frag_rows: list[list[Any]], frag_head: list[str],
          prompts: list[str], out: Path, source: Path, temperature: float = TEMPERATURE) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Сравнение"
    head = ["№", "Вид", "Вопрос"]
    for p in prompts:
        head += [f"Ответ ({LABELS[p]})", f"Ссылки ({LABELS[p]})", f"Время, с ({LABELS[p]})",
                 f"Правила по случаю ({LABELS[p]})", f"Фрагментов в окне ({LABELS[p]})"]
    ws.append(head)
    for r in rows:
        line = [r["id"], r["kind"], r["question"]]
        for p in prompts:
            got = r[p]
            line += [got["answer"], got["used"], got["seconds"], got.get("rules", ""),
                     got.get("window", "")]
        ws.append(line)
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    widths = [7, 9, 40] + [80, 14, 9, 18, 14] * len(prompts)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    ws.freeze_panes = "D2"
    fs = wb.create_sheet("Фрагменты")
    fs.append(frag_head)
    for r in frag_rows:
        fs.append(r)
    ps = wb.create_sheet("Промты")
    ps.append(["Промт", "Отпечаток (sha256, 12 знаков)", "Точный текст системного промта"])
    for p in prompts:
        text = system_prompt(p)
        ps.append([LABELS[p], fingerprint(text), text])
    ps.column_dimensions["A"].width = 16
    ps.column_dimensions["B"].width = 18
    ps.column_dimensions["C"].width = 120
    for row in ps.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    info = wb.create_sheet("О прогоне")
    for line in (f"Повтор на фрагментах из: {source.name}",
                 "Промты: " + ", ".join(f"{LABELS[p]} ({fingerprint(system_prompt(p))})"
                                         for p in prompts),
                 f"Температура модели: {temperature}",
                 f"Вопросов: {len(rows)}; дата: {datetime.now():%d.%m.%Y %H:%M}",
                 "Фрагменты те же, что в исходном прогоне: новый поиск не делался."):
        info.append([line])
    info.column_dimensions["A"].width = 100
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    from .settings import Settings

    p = argparse.ArgumentParser(prog="python georag.py replay")
    p.add_argument("table", help="таблица прогона report (.xlsx)")
    p.add_argument("--prompts", default="full,short", help="какие промты: full, short")
    p.add_argument("--only", default="", help="только эти вопросы: m01,f04")
    p.add_argument("--out", default="", help="куда записать .xlsx")
    p.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help="температура модели (для сравнения — 0, без случайности)")
    args = p.parse_args(argv)
    prompts = [x.strip() for x in args.prompts.split(",") if x.strip() in PROMPTS]
    if not prompts:
        print("Промты — full и/или short.", file=sys.stderr)
        return 1
    path = Path(args.table)
    try:
        answers, frags = read_report(path)
    except (OSError, ValueError) as exc:
        print(f"Не читается {path}: {exc}", file=sys.stderr)
        return 1
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        answers = [a for a in answers if _s(a.get("№")) in only]
    base = Settings()
    rows, frag_rows = [], []
    frag_head = ["№ вопроса", "Словарь", "[n]", "Использован в ответе", "Название статьи / отчёта",
                 "Файл", "Год", "Страницы", "Раздел", "Текст фрагмента"]
    for a in answers:
        qid, question = _s(a.get("№")), _s(a.get("Вопрос"))
        dict_flag = _s(a.get("Словарь"))
        found = frags.get((qid, dict_flag)) or frags.get((qid, ""), [])
        sources = to_sources(found)
        row: dict[str, Any] = {"id": qid, "kind": _s(a.get("Вид")), "question": question}
        for prompt in prompts:
            settings = replace(base, prompt=prompt, dictionary=dict_flag != "нет",
                               temperature=args.temperature)
            if sources:
                row[prompt] = answer_on(question, sources, settings)
            else:
                row[prompt] = {"answer": "(в исходном прогоне фрагментов не было — не повторялся)",
                               "used": "", "seconds": 0, "rules": "", "window": "0"}
            print(f"{qid:>6} {LABELS[prompt]:<15} {row[prompt]['seconds']:>6} с", flush=True)
        rows.append(row)
        frag_rows += [[r.get(h) for h in frag_head] for r in found]
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    out = Path(args.out) if args.out else REPORTS / f"replay-{path.stem}-{stamp}.xlsx"
    for prompt in prompts:
        print(f"Промт «{LABELS[prompt]}»: отпечаток {fingerprint(system_prompt(prompt))}")
    print(f"Таблица: {write(rows, frag_rows, frag_head, prompts, out, path, args.temperature)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
