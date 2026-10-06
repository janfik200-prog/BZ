"""Как бот выглядит в Телеграме: меню, кнопки, тексты сообщений.

Здесь только оформление — без сети и без базы: на вход данные, на выход текст
(HTML Телеграма) и кнопки. Логика — в bot.py.

Два вида кнопок:

* **меню внизу экрана** (reply keyboard) — всегда на месте, вместо команд:
  Поиск, Статьи, Факты, Новый разговор, Помощь. Любой другой текст —
  вопрос чат-боту;
* **кнопки под сообщением** (inline keyboard) — действие с этим сообщением:
  открыть фрагмент [2], листать выдачу, посмотреть, как бот искал, остановить ответ.
  Нажатие приходит боту как callback с короткой строкой `действие:номер:…`
  (у Телеграма до 64 байт, поэтому в ней только номера, а сами данные — в
  разговоре у бота).
"""

from __future__ import annotations

import html
import re
from typing import Any

LIMIT = 3900  # у Телеграма 4096 знаков на сообщение, берём с запасом

# Меню внизу экрана: надпись на кнопке → что делать.
MENU_SEARCH = "Поиск"
MENU_ARTICLES = "Статьи"
MENU_DATASETS = "Факты"
MENU_NEW = "Новый разговор"
MENU_HELP = "Помощь"
MENU = {
    MENU_SEARCH: "search",
    MENU_ARTICLES: "articles",
    MENU_DATASETS: "datasets",
    MENU_NEW: "new",
    MENU_HELP: "help",
}

PAGE = {"search": 5, "articles": 8, "chunks": 6}  # сколько пунктов на странице
FRAGMENT_BUTTONS = 12  # кнопок фрагментов под ответом
PER_ROW = 6
DATASET_BUTTONS = 30

HELP = f"""<b>База знаний ГеоRAG</b>

<b>Вопрос</b> — просто напишите его, хоть сложный, из нескольких частей. Бот ищет вместе с моделью: Qwen3 разбирает вопрос, читает найденное, берёт только подходящее и, если чего-то не хватает, ищет ещё. Ответ — по статьям базы, с номерами фрагментов [1], [2].

Нужно подробно, кратко, списком — так и напишите в вопросе: «подробно расскажи…», «кратко: …». Вопрос вдогонку («а по ASTER?») бот понимает — помнит разговор; «{MENU_NEW}» начинает заново.

Под ответом: номер — фрагмент целиком; «Как искал» — какие были запросы и что модель отобрала. Пока бот пишет, его можно остановить. Если в базе такого нет, бот так и скажет и ответит из общих знаний — с пометкой.

Кнопки внизу:
<b>{MENU_SEARCH}</b> — найти фрагменты статей (обычный поиск, без модели)
<b>{MENU_ARTICLES}</b> — что лежит в базе; у статьи — ссылка и её фрагменты
<b>{MENU_DATASETS}</b> — картинка связей: сущность (место, разлом, метод) и то, с чем она связана по статьям; соседи — кнопками, «Цитаты» — факты с источниками, таблица для Excel

Команды: /search <i>запрос</i>, /articles, /facts <i>название</i>, /f 2, /new, /stats — что в базе и готова ли модель."""


# --------------------------------------------------------------------------- #
#  Кнопки
# --------------------------------------------------------------------------- #
def button(text: str, data: str) -> dict[str, Any]:
    return {"text": text, "callback_data": data}


def url_button(text: str, url: str) -> dict[str, Any]:
    return {"text": text, "url": url}


def inline(rows: list[list[dict[str, Any]]]) -> dict[str, Any] | None:
    rows = [r for r in rows if r]
    return {"inline_keyboard": rows} if rows else None


def menu_keyboard() -> dict[str, Any]:
    return {
        "keyboard": [
            [{"text": MENU_SEARCH}, {"text": MENU_ARTICLES}, {"text": MENU_DATASETS}],
            [{"text": MENU_NEW}, {"text": MENU_HELP}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "input_field_placeholder": "Вопрос по статьям базы…",
    }


def rows_of(buttons: list[dict[str, Any]], per_row: int = PER_ROW) -> list[list[dict[str, Any]]]:
    return [buttons[i : i + per_row] for i in range(0, len(buttons), per_row)]


def is_link(url: Any) -> bool:
    return isinstance(url, str) and url.startswith(("http://", "https://"))


# --------------------------------------------------------------------------- #
#  Текст
# --------------------------------------------------------------------------- #
def esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=False)


def answer_html(text: str) -> str:
    """Ответ модели → HTML Телеграма: жирное, заголовки, пункты списков."""
    out = []
    for line in esc(text).split("\n"):
        stripped = line.strip()
        heading = re.match(r"^#{1,4}\s+(.*)", stripped)
        if heading:
            line = f"<b>{heading.group(1)}</b>"
        elif re.match(r"^[-*•]\s+", stripped):
            line = "• " + re.sub(r"^[-*•]\s+", "", stripped)
        out.append(line)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", "\n".join(out))


def plain(text: str) -> str:
    """HTML Телеграма → простой текст (когда разметку не приняли)."""
    return esc(re.sub(r"<[^>]+>", "", html.unescape(text)))


def split(text: str, limit: int = LIMIT) -> list[str]:
    """Длинный текст — на сообщения по абзацам, не разрывая теги посреди строки."""
    parts, current = [], ""
    for block in text.split("\n"):
        while len(block) > limit:  # одна строка длиннее сообщения
            parts.append((current + "\n" + block[:limit]).strip())
            block, current = block[limit:], ""
        if len(current) + len(block) + 1 > limit:
            parts.append(current.strip())
            current = block
        else:
            current = f"{current}\n{block}" if current else block
    if current.strip():
        parts.append(current.strip())
    return parts or [""]


def where(s: dict[str, Any]) -> str:
    bits = []
    if s.get("year"):
        bits.append(str(s["year"]))
    if s.get("pages"):
        bits.append("с. " + ", ".join(str(p) for p in s["pages"][:3]))
    return ", ".join(bits)


def source_line(s: dict[str, Any]) -> str:
    link = s.get("url") or ""
    title = esc(str(s.get("title", ""))[:110])
    title = f'<a href="{esc(link)}">{title}</a>' if is_link(link) else title
    place = where(s)
    return f"[{s['n']}] {title}" + (f" — {place}" if place else "")


def search_notes(info: dict[str, Any]) -> str:
    """Как искалось: датасет, части вопроса, отбор моделью, новые круги — строками."""
    notes = []
    if info.get("dataset"):
        notes.append(f"собрано из фактов графа о «{info['dataset']}» — по всем статьям базы")
    if info.get("graph"):
        notes.append(f"из графа связей — фрагментов о предмете вопроса: {info['graph']}")
    if info.get("parts"):
        notes.append(
            "вопрос разобран на части: "
            + "; ".join(
                f"{i}) {p['question']} — фрагментов {p['found']}"
                for i, p in enumerate(info["parts"], start=1)
            )
        )
    if info.get("judged") is False:
        notes.append("модель при отборе не ответила — фрагменты взяты по близости")
    elif info.get("checked"):
        notes.append(
            f"модель прочла найденное (фрагментов: {info['checked']}) и взяла подходящие"
            + (f"; кругов поиска: {info['rounds']}" if (info.get("rounds") or 1) > 1 else "")
        )
    if info.get("extra"):
        missing = f", не хватало: {'; '.join(info['missing'])}" if info.get("missing") else ""
        notes.append(f"искал ещё{missing}; запросы: " + ", ".join(f"«{q}»" for q in info["extra"]))
    return "\n".join(notes)


def how_text(record: dict[str, Any]) -> str:
    """«Как искал» — по кнопке под ответом."""
    info = record.get("info") or {}
    lines = [f"<b>Как искал ответ на:</b> {esc(record['question'])}"]
    notes = search_notes(info)
    if notes:
        lines.append(esc(notes))
    first = [q for q in info.get("queries") or [] if q not in (info.get("extra") or [])]
    if first:
        lines.append("<b>Запросы к базе:</b>\n" + "\n".join(f"• {esc(q)}" for q in first))
    sources = record.get("sources") or []
    if sources:
        lines.append(
            f"<b>Фрагментов у модели:</b> {len(sources)}"
            + (
                f", процитировано {len(record.get('used') or [])}"
                if record.get("used") is not None
                else ""
            )
        )
    elif record.get("mode") == "без базы":
        lines.append("Близкого в базе не нашлось — ответ из общих знаний модели.")
    if record.get("seconds") is not None:
        lines.append(f"<i>{esc(record.get('model') or '')}, {record['seconds']} с</i>")
    return "\n\n".join(lines)


def answer_text(
    answer: str, general: str | None, sources: list[dict[str, Any]], done: dict[str, Any]
) -> str:
    """Готовый ответ: текст, источники, подвал."""
    parts = []
    if general:
        parts.append(f"<b>{esc(general)}</b>\n")
    if answer.strip():
        parts.append(answer_html(answer))
    elif not done.get("off_topic"):  # вопрос не по теме — отказ уже написан выше
        parts.append("<i>(пусто)</i>")
    if done.get("stopped"):
        parts.append("\n<i>— остановлено</i>")
    if sources:
        used = set(done.get("used") or [])
        lines = [
            source_line(s) + ("" if s["n"] in used else " <i>(не процитирован)</i>")
            for s in sources
        ]
        parts.append(
            "\n<b>Источники</b> — кнопки с номерами ниже открывают фрагмент целиком\n"
            + "\n".join(lines)
        )
    foot = []
    cited, claims = done.get("coverage") or (0, 0)
    if claims:
        foot.append(f"со ссылкой {cited} из {claims} утверждений")
    if done.get("unknown"):
        foot.append(f"ссылки на несуществующие фрагменты: {done['unknown']}")
    if done.get("removed"):
        foot.append(f"убрано фраз без ссылки на фрагмент: {done['removed']}")
    if done.get("mode") == "без базы" and not done.get("off_topic"):
        foot.append("ответ модели, не из базы знаний")
    if done.get("model"):
        foot.append(f"{done['model']}, {done.get('seconds')} с")
    if foot:
        parts.append("\n<i>" + esc(" · ".join(foot)) + "</i>")
    return "\n".join(parts)


def answer_keyboard(aid: int, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Под ответом: фрагменты, как искал, новый разговор."""
    frags = [button(f"[{s['n']}]", f"f:{aid}:{s['n']}") for s in sources[:FRAGMENT_BUTTONS]]
    return inline([*rows_of(frags), [button("Как искал", f"how:{aid}"), button(MENU_NEW, "new")]])


def stop_keyboard(aid: int) -> dict[str, Any] | None:
    return inline([[button("Остановить", f"stop:{aid}")]])


def fragment_text(src: dict[str, Any]) -> str:
    head = source_line(src) if "n" in src else esc(src.get("title", ""))
    path = " / ".join(src.get("headings") or [])
    note = ""
    if src.get("kind") == "датасет":
        note = "<i>Сводка фактов из графа: выписаны из всех статей базы.</i>\n"
    elif src.get("feature"):
        note = f"<i>Доказательство признака «{esc(src['feature'])}».</i>\n"
    return (
        f"{head}\n"
        + note
        + (f"<i>{esc(path)}</i>\n" if path else "")
        + f"\n{esc(src.get('text', ''))}"
    )


def fragment_keyboard(
    src: dict[str, Any], article_data: str | None = None
) -> dict[str, Any] | None:
    row = []
    if is_link(src.get("url")):
        row.append(url_button("Статья у издателя", src["url"]))
    if article_data:
        row.append(button("Вся статья", article_data))
    return inline([row])


# --------------------------------------------------------------------------- #
#  Списки с листанием: поиск, статьи, фрагменты статьи
# --------------------------------------------------------------------------- #
def pages(total: int, kind: str) -> int:
    size = PAGE[kind]
    return max(1, (total + size - 1) // size)


def list_text(lst: dict[str, Any], page: int) -> str:
    kind, items = lst["kind"], lst["items"]
    size = PAGE[kind]
    chunk = items[page * size : (page + 1) * size]
    head = lst["title"]
    if len(items) > size:
        head += f" · стр. {page + 1} из {pages(len(items), kind)}"
    lines = [head, ""]
    for i, item in enumerate(chunk, start=page * size + 1):
        if kind == "search":
            near = (
                f", похожесть {item['similarity']:.0%}"
                if item.get("similarity") is not None
                else ""
            )
            place = where(item)
            lines.append(
                f"<b>{i}.</b> {esc(item['title'][:100])}"
                + (f" — {place}" if place else "")
                + f" <i>(нашёл: {esc(item.get('found_by'))}{near})</i>\n"
                f"{esc(item['text'][:300])}…\n"
            )
        elif kind == "articles":
            lines.append(
                f"<b>{i}.</b> {esc((item['title'] or item['doc_id'])[:110])} — "
                f"{item.get('year') or '—'}, фрагментов {item['chunks']}"
            )
        else:  # фрагменты одной статьи
            path = " / ".join(item.get("headings") or [])
            pages_ = f"с. {item['pages'][0]}" if item.get("pages") else ""
            lines.append(
                f"<b>{i}.</b> "
                + (f"<i>{esc(path[:80])}</i> " if path else "")
                + (f"({pages_}) " if pages_ else "")
                + esc(" ".join(item["text"].split())[:160])
                + "…"
            )
    lines.append("\n<i>Номер — открыть</i>" if chunk else "<i>Пусто</i>")
    return "\n".join(lines)


def list_keyboard(
    pid: int, lst: dict[str, Any], page: int, extra: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    kind, items = lst["kind"], lst["items"]
    size = PAGE[kind]
    start = page * size
    numbers = [
        button(str(i + 1), f"it:{pid}:{i}") for i in range(start, min(start + size, len(items)))
    ]
    total = pages(len(items), kind)
    nav = []
    if total > 1:
        nav = [
            button("‹ Назад", f"pg:{pid}:{page - 1}") if page > 0 else button(" ", "noop"),
            button(f"{page + 1} / {total}", "noop"),
            button("Дальше ›", f"pg:{pid}:{page + 1}") if page < total - 1 else button(" ", "noop"),
        ]
    return inline([*rows_of(numbers, 8 if kind == "articles" else PER_ROW), nav, extra or []])


def article_text(doc: dict[str, Any]) -> str:
    lines = [f"<b>{esc(doc.get('title') or doc['doc_id'])}</b>"]
    if doc.get("authors"):
        lines.append(
            esc(", ".join(doc["authors"][:4]) + (" и др." if len(doc["authors"]) > 4 else ""))
        )
    bits = [str(doc["year"]) if doc.get("year") else "", doc.get("journal") or ""]
    bits = [b for b in bits if b]
    if bits:
        lines.append(esc(" · ".join(bits)))
    lines.append(
        f"Фрагментов в базе: {doc.get('chunks', '—')}"
        + (f" · страниц: {doc['pages']}" if doc.get("pages") else "")
    )
    if doc.get("source"):
        lines.append(f"<i>источник: {esc(doc['source'])}</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Датасеты
# --------------------------------------------------------------------------- #
def datasets_text(rows: list[dict[str, Any]], total: int | None = None) -> str:
    if not rows:
        return "Фактов пока нет: граф не построен. На компьютере: python georag.py graph"
    total = total or len(rows)
    more = (
        f"\nВсего сущностей с фактами: {total}; любую — командой /facts <i>название</i>"
        if total > DATASET_BUTTONS
        else ""
    )
    return (
        "<b>Факты из статей</b>\n"
        "Всё, что статьи базы говорят о чём-то — месте, разломе, методе, процессе, — "
        "с цитатами. Больше всего фактов у этих; нажмите — придёт картинка связей." + more
    )


def datasets_keyboard(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return inline(
        [
            [button(f"{r['name'][:40]} · {r['documents']} ст. · {r['facts']} факт.", f"ds:{i}")]
            for i, r in enumerate(rows[:DATASET_BUTTONS])
        ]
    )


def fact_line(row: dict[str, Any]) -> str:
    """Факт датасета одной строкой: «от — связь — к»."""
    if row["direction"] == "→":
        return f"{row['about']} — {row['relation']} — {row['other']}"
    return f"{row['other']} — {row['relation']} — {row['about']}"


def dataset_text(data: dict[str, Any], limit: int = 25) -> str:
    inside = f" (вместе с: {esc(', '.join(data['includes']))})" if data.get("includes") else ""
    lines = [
        f"<b>Факты: {esc(data['name'])}</b>{inside}",
        f"Фактов: {len(data['facts'])} · статей: {data['documents']}",
        "",
    ]
    for row in data["facts"][:limit]:
        lines.append(f"• {esc(fact_line(row))} <i>({row['documents']} ст.)</i>")
    if len(data["facts"]) > limit:
        lines.append(f"<i>… и ещё {len(data['facts']) - limit} — в таблице (с цитатами)</i>")
    return "\n".join(lines)


def dataset_keyboard(idx: int) -> dict[str, Any] | None:
    return inline(
        [
            [button("Картинка связей", f"ds:{idx}"), button("Таблица для Excel", f"csv:{idx}")],
            [button("Спросить бота", f"ask:{idx}"), button("‹ Все сущности", "dsl")],
        ]
    )


# --------------------------------------------------------------------------- #
#  Картинка связей (сама картинка — picture.py)
# --------------------------------------------------------------------------- #
NEIGHBOR_BUTTONS = 8


def graph_caption(data: dict[str, Any], shown: int) -> str:
    facts = len(data.get("facts") or [])
    inside = (
        f"\nВходит в неё: {esc(', '.join(data['includes'][:5]))}" if data.get("includes") else ""
    )
    more = f" На картинке — {shown} главных связей." if facts > shown else ""
    return (
        f"<b>{esc(data['name'])}</b> — фактов {facts} из {data.get('documents', 0)} "
        f"{'статьи' if data.get('documents') == 1 else 'статей'}.{more}{inside}\n"
        "Кнопки ниже — перейти к соседу; «Цитаты» — факты списком, с источниками."
    )[:1000]


def graph_keyboard(idx: int, neighbors: list[tuple[str, int]]) -> dict[str, Any] | None:
    """Соседи по два в ряд (переход — в том же сообщении), потом цитаты, таблица, вопрос."""
    near = [button(("→ " + name)[:40], f"dn:{n}") for name, n in neighbors]
    return inline(
        [
            *rows_of(near, 2),
            [button("Цитаты", f"dt:{idx}"), button("Таблица для Excel", f"csv:{idx}")],
            [button("Спросить бота", f"ask:{idx}"), button("‹ Все сущности", "dsl")],
        ]
    )


def dataset_question(name: str) -> str:
    return f"Что известно о «{name}»: с чем связано и что о нём написано в статьях?"


# --------------------------------------------------------------------------- #
#  Состояние: база и модель
# --------------------------------------------------------------------------- #
def status_text(status: dict[str, Any] | None, stats: dict[str, Any] | None) -> str:
    lines = []
    if stats is None:
        lines.append("<b>База:</b> не отвечает. На компьютере: python georag.py start")
    else:
        lo, hi = stats.get("years") or (None, None)
        lines.append(
            f"<b>База:</b> статей {stats['documents']}, фрагментов {stats['chunks']}"
            + (f", годы {lo}–{hi}" if lo else "")
        )
    if status is not None:
        lines.append(
            "<b>Модель:</b> "
            + esc(status.get("model") or "")
            + (" — готова" if status.get("ok") else f" — {esc(status.get('error'))}")
        )
        if not status.get("ok"):
            lines.append(
                "<i>Без модели бот ищет по близости, как обычный поиск, — сложные "
                "вопросы он так понимает хуже.</i>"
            )
    return "\n".join(lines)
