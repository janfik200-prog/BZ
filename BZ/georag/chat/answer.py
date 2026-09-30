"""Чат-бот: вопрос → поиск по базе вместе с моделью → ответ Qwen3 по найденному.

Обычный поиск (векторы + слова) на сложном вопросе ошибается в обе стороны:
не находит нужное, если оно сказано другими словами, и приносит похожее по
словам, но не о том. Поэтому поиск ведёт модель:

1. **Разбор.** Qwen3 делит составной вопрос на части (до трёх), на каждую
   составляет запросы — по-русски и по-английски, терминами предметной
   области; отмечает, о чём конкретно вопрос и просят ли «собрать всё».
2. **Широкий набор.** По запросам ищется с запасом — до десяти кандидатов на
   часть, с мягким порогом: лучше показать модели лишнее, чем не показать нужное.
3. **Модель отбирает.** Qwen3 читает кандидатов и называет те, где правда есть
   ответ (похожее по словам, но о другом — отбрасывает), говорит, чего не
   хватает, и составляет запросы, чтобы это найти.
4. **Ищет ещё.** По этим запросам — новый набор, модель отбирает снова. Всего
   до трёх кругов; останавливается, когда всего хватает или новое не находится.
5. **«Собрать всё»** («что известно об Анабарском щите») — из графа: все факты
   о предмете вопроса и о том, что в него входит, выписанные моделью из всех
   статей, с цитатами; поиск только добавляет.
6. **Ответ.** Модель отвечает ТОЛЬКО по отобранным фрагментам, после каждого
   утверждения — номер фрагмента [1]. Длина — по просьбе в самом вопросе:
   «подробно», «кратко», «перечисли» — так и отвечает; настроек нет.
7. **Не нашлось** — статьи не показываются: бот говорит, что в базе такого нет,
   и отвечает из общих знаний модели, с пометкой.

Модель не ответила при отборе (Ollama выключена или сбой) — отбор по порогу
близости, как у обычного поиска: бот работает, только хуже на сложных вопросах.

Ответ идёт потоком — событиями:

    {"type": "status", "text": "…"}                         что сейчас делается
    {"type": "sources", "queries": […], "sources": […],     на что опирается ответ
     "parts": […], "extra": […], "missing": […], "dataset": "…"|None,
     "rounds": 1–3, "checked": N, "judged": true|false}
    {"type": "general", "queries": […], "text": "…", …}     в базе нет — ответ без неё
                                                            (off_topic: true — вопрос не о
                                                            геологии: только отказ, без ответа)
    {"type": "token", "text": "…"}                          кусок ответа, много раз
    {"type": "done", "mode": "база"|"без базы", "used": [1, 3], "unknown": [],
     "coverage": [5, 6], "rounds": …, …}
    {"type": "error", "error": "…"}
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Iterator

from ..index.search import RRF_K, hybrid_search
from ..llm import DEFAULT_MODEL, OLLAMA_HOST, LLMError, Ollama, no_think

MODEL = DEFAULT_MODEL
OLLAMA = OLLAMA_HOST

# Сколько фрагментов давать модели для ответа: восемь, из одной статьи до
# трёх. Вместе с соседними кусками (см. NEIGHBOR_CHARS) это около 7 тысяч
# токенов — окно модели 12 тысяч, с запасом на ответ и историю разговора.
TOP_K = 8
PER_DOC = 3
QUERIES = 3                    # сколько поисковых запросов на часть вопроса
CANDIDATES = 12                # сколько брать из каждого запроса до слияния
PARTS = 3                      # на сколько частей самое большее делится вопрос
PART_MIN = 2                   # гарантированных мест у каждой части
# Поиск с моделью: широкий набор кандидатов, модель отбирает и ищет ещё.
WIDE = 16                      # кандидатов на простой вопрос (и на запросы нового круга)
WIDE_PART = 8                  # на каждую часть составного вопроса и на вопрос целиком
LOOSE_SIMILARITY = 0.35        # мягкий порог для кандидатов: отбирает уже модель
# Модель читает кандидатов пачками по JUDGE_MAX — все, а не первые сколько-то:
# раньше при трёх частях третьей доставалось 4 кандидата из 10, а найденное
# по вопросу целиком модель не видела вовсе. Текста — CANDIDATE_CHARS: фрагмент
# около 2000 знаков, и по первым 450 нужное отбрасывалось, если суть дальше.
JUDGE_MAX = 12
JUDGE_SLACK = 4                # короткий хвост пачки читается вместе с предыдущей
CANDIDATE_CHARS = 1000
MAX_ROUNDS = 3                 # кругов поиска самое большее
EXTRA_QUERIES = 3              # новых запросов за круг
# Новый запрос, почти совпадающий по основам слов с уже сделанным, не ищется:
# он приносит те же фрагменты, а круг стоит вызова модели.
REPEAT_OVERLAP = 0.6
# «Собрать всё» о предмете вопроса: все факты о нём из графа и фрагменты к главным.
DATASET_EVIDENCE = 5
# Вопрос о названном предмете (не «собрать всё»): фрагменты, из которых граф
# выписал факты о нём, идут модели в кандидаты вместе с найденным поиском.
GRAPH_CANDIDATES = 6
DATASET_ITEMS = 40             # сколько фактов (×2) попадает в сводку
DATASET_CHARS = 6000

# Порог близости — как «отсечь непохожее» во вкладке поиска. Совпадение по
# словам запроса проходит всегда: оно само по себе доказательство.
MIN_SIMILARITY = 0.45

FRAGMENT_CHARS = 2400
# Короткий фрагмент (заголовок, подпись, обрывок абзаца) без соседей почти
# ничего не говорит. К такому добавляется следующий кусок той же статьи,
# если он из того же раздела. Номер у них общий.
SHORT_FRAGMENT = 900
NEIGHBOR_CHARS = 1200

HISTORY_TURNS = 2              # сколько прошлых вопросов-ответов помнит модель
HISTORY_CHARS = 1500

# Окно модели считается грубо: знаков на токен у Qwen3 на русском — около трёх.
# Сообщение и ответ должны влезть в num_ctx, иначе Ollama молча отрезает начало —
# вместе с правилами. Если не влезает, сначала укорачивается история, потом
# отбрасываются последние фрагменты.
CHARS_PER_TOKEN = 3.0
ANSWER_RESERVE = 3000          # токенов под сам ответ

SYSTEM = """Ты — помощник геолога. Отвечаешь на вопросы по базе знаний о рудной геологии \
и прогнозе оруденения. Твой единственный источник — фрагменты статей из базы, которые \
даны в сообщении.

Правила:
1. Отвечай ТОЛЬКО по фрагментам. Не добавляй фактов, чисел, названий, методов и ссылок, \
которых во фрагментах нет, — даже если знаешь их сам.
2. После каждого утверждения ставь номер фрагмента в квадратных скобках: [1]. \
Если утверждение опирается на несколько — [1][3].
3. Если во фрагментах нет ответа на вопрос, напиши только одну фразу: \
«В базе знаний нет ответа на этот вопрос.» — и больше ничего.
4. Пиши по-русски, даже если фрагменты на английском. Термин можно дать в скобках \
по-английски.
5. Объём и вид ответа — как просит человек в вопросе. Просит подробно, развёрнуто, \
обзор, всё о чём-то — отвечай развёрнуто, по разделам с короткими заголовками жирным \
(например, **Методы**), с числами, условиями и различиями между источниками. Просит \
кратко, в двух словах, одним списком — так и отвечай. Не сказал — по существу: вывод \
в 1–2 предложения и главное списком, без воды и без повторов.
6. Если фрагменты отвечают лишь частично — ответь на то, что в них есть, и одной \
фразой назови, чего во фрагментах нет. Фраза из правила 3 — только когда нет совсем \
ничего.
7. Сведения во фрагментах относятся к конкретным районам, месторождениям и объектам. \
Называй, к чему относится утверждение («на Кокпатасском рудном поле [3]», «в \
Восточном Донбассе [1]»), и не выдавай частный случай из одной статьи за общее правило."""


PLAN_SYSTEM = ("Ты разбираешь вопрос геолога и составляешь поисковые запросы к базе научных "
               "статей по рудной геологии. Отвечай только JSON, без пояснений.")

PLAN_USER = """Вопрос пользователя: «{question}»
{context}
Разбери вопрос для поиска по базе статей.
1. Если вопрос составной — в нём несколько разных вещей (признаки и методы; две
   территории для сравнения; причина и следствие), — раздели его на части, не больше
   {parts}. Простой вопрос — одна часть, не дроби его.
2. На каждую часть — {n} поисковых запроса: 2–6 слов, терминами предметной области,
   без кавычек и логических операторов; хотя бы один по-русски и хотя бы один
   по-английски; запросы различаются по смыслу, а не перестановкой слов.
3. «предмет» — если вопрос про что-то конкретное (место, разлом, метод, процесс,
   признак), его название в именительном падеже, как в вопросе; иначе null.
4. «всё» — true, если просят собрать всё известное о нём: что известно, что описано,
   с чем связано, полный перечень, обзор. Иначе false.
5. «вопрос_целиком» — вопрос так, чтобы он был понятен без разговора. Если это
   продолжение предыдущего вопроса («а по космоснимкам?», «а какие там породы?»),
   допиши предмет из предыдущего: «Что известно о Персияновском разломе по
   космоснимкам?» — и части, запросы и предмет составляй по нему. Если вопрос
   новый и понятен сам по себе — повтори его как есть, предыдущий не приплетай.
6. «по_теме» — true, если вопрос о геологии, полезных ископаемых, геофизике,
   геохимии, дистанционном зондировании Земли или методах их изучения; false —
   если о другом (кулинария, спорт, программирование…).

Верни JSON: {{"вопрос_целиком": "...", "части": [{{"вопрос": "...", "запросы": ["...", "..."]}}], \
"предмет": null, "всё": false, "по_теме": true}}"""

JUDGE_SYSTEM = ("Ты отбираешь фрагменты научных статей, из которых можно ответить на вопрос "
                "геолога, и решаешь, что ещё поискать в базе. Отвечай только JSON.")

JUDGE_USER = """Вопрос: «{question}»
{parts}{taken}
Кандидаты из базы — нашёл поиск по словам и смыслу, среди них бывают случайные:
{candidates}

1. «подходят» — номера кандидатов, в которых есть сведения для ответа (хотя бы на одну
   часть вопроса), от самых полезных к менее полезным. Кандидат, где только похожие
   слова, а по сути речь о другом, — не бери.
2. «не_хватает» — каких сведений для полного ответа нет ни в одном подходящем
   фрагменте, коротко. Всего хватает — пустой список.
3. «запросы» — до {n} новых поисковых запросов, чтобы найти недостающее: 2–6 слов,
   терминами предметной области, по-русски или по-английски, не повторяя сделанные
   и не переставляя в них слова: {done}. Всего хватает или искать больше нечего —
   пустой список.

Верни JSON: {{"подходят": [1, 4], "не_хватает": ["..."], "запросы": ["..."]}}"""


# Вопрос «собрать всё» — на случай, когда модель вопрос не разобрала.
_COLLECT = re.compile(
    r"что\s+(?:известно|знаем|есть|описано|написано)|вс[её]\s+(?:о|об|про|что)\b|"
    r"перечисл|полн\w*\s+(?:список|перечень)|обзор|датасет|"
    r"какие\s+(?:признаки|методы|процессы)\s+(?:на|в|для|по)\b",
    re.IGNORECASE)

NO_ANSWER = "В базе знаний нет ответа"      # начало фразы-отказа из правила 3

NOT_IN_BASE = ("По вашему запросу в базе знаний такой информации нет. Ниже — ответ модели "
               "из её общих знаний: он не подтверждён статьями базы, проверяйте его.")

# Вопрос не о геологии и в базе ничего нет: бот не отвечает из общих знаний
# (раньше выдавал, например, рецепт хлеба). Вернуть старое — Settings.general_off_topic.
OFF_TOPIC = ("Вопрос не относится к геологии, а бот отвечает только по рудной геологии и "
             "смежным наукам о Земле. В базе знаний статей об этом нет.")

GENERAL_SYSTEM = """Ты — помощник геолога, специалист по рудной геологии и прогнозу \
оруденения. В базе знаний статей ответа на вопрос не нашлось, поэтому отвечаешь из \
общих знаний.

Правила:
1. Не ссылайся на статьи, авторов, DOI и номера фрагментов — у тебя их нет. Не ставь \
номера в квадратных скобках.
2. Не выдумывай точных чисел, названий месторождений и дат. Если не уверен — так и скажи.
3. Пиши по-русски. Объём — как просит человек: подробно — развёрнуто, кратко — коротко; \
не сказал — по существу: вывод и главное списком, в конце — что стоит проверить по \
литературе."""


_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_CITE_MARK = re.compile(r"[ \t]*\[\d+(?:\s*[,;]\s*\d+)*\]")
# Фразы о самих фрагментах, а не о геологии: «по данным статей базы», «во
# фрагментах нет», «в базе знаний об этом ничего».
_ABOUT_SOURCES = re.compile(r"фрагмент|в базе|статей базы|статьях базы|по данным статей|"
                            r"в найденных", re.IGNORECASE)


@dataclass
class Settings:
    model: str = MODEL
    host: str = OLLAMA
    top_k: int = TOP_K
    per_doc: int = PER_DOC
    queries: int = QUERIES
    temperature: float = 0.1
    num_ctx: int = 12288
    timeout: int = 300
    parts: int = PARTS             # на сколько частей самое большее делить вопрос
    # Вопрос не по теме и в базе пусто: False — короткий отказ, True — ответ из общих знаний.
    general_off_topic: bool = False


@dataclass
class Part:
    """Часть вопроса и её поисковые запросы."""
    question: str
    queries: list[str]
    found: int = 0                 # сколько фрагментов за этой частью в ответе


@dataclass
class Plan:
    """Разбор вопроса: части с запросами, предмет вопроса, «собрать всё»."""
    parts: list[Part]
    subject: str | None = None      # о чём вопрос: «Анабарский щит», «окварцевание»
    collect: bool = False
    main: str = ""                 # сам вопрос как запрос (с прошлым, если вдогонку)
    extra: list[str] = field(default_factory=list)      # запросы следующих кругов
    missing: list[str] = field(default_factory=list)    # чего не хватало, по словам модели
    standalone: str = ""           # вопрос, понятный без разговора (вдогонку — с предметом)
    on_topic: bool = True          # о геологии ли вопрос, по словам модели

    @property
    def queries(self) -> list[str]:
        """Все запросы первого круга, без повторов: сам вопрос, затем по частям."""
        out: list[str] = []
        for q in [self.main, *(q for p in self.parts for q in p.queries)]:
            if q and q.lower() not in {x.lower() for x in out}:
                out.append(q)
        return out

    def __iter__(self):
        """Разбор можно перебирать как список запросов — так с ним работал старый код."""
        return iter(self.queries)

    def describe(self) -> list[dict]:
        return [{"question": p.question, "queries": p.queries, "found": p.found}
                for p in self.parts]


# --------------------------------------------------------------------------- #
#  Запросы к базе
# --------------------------------------------------------------------------- #
def previous_question(history: list[dict]) -> str:
    past = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
    return past[-1].strip() if past else ""


# Вопрос вдогонку: начинается с «а», «и», «ещё» или ссылается на сказанное
# («его», «там», «этот»). Одной длины мало: «Как испечь ржаной хлеб?» тоже
# короткий, и раньше к нему приклеивался прошлый вопрос про космоснимки.
_FOLLOW_UP = re.compile(
    r"^(?:а|и|ещё|еще|также|тоже|тогда|то\s+есть)\b|"
    r"\b(?:его|её|ее|их|него|неё|нее|них|там|тут|здесь|этот|эта|это|эти|этого|этой|этих|"
    r"этим|этому|этом|том|той|тех|такой|такие|такого|он|она|оно|они)\b",
    re.IGNORECASE)
FOLLOW_UP_WORDS = 8


def search_query(question: str, history: list[dict]) -> str:
    """Вопрос как есть — первый из запросов. Короткий вопрос вдогонку («а по
    ASTER?») без прошлого вопроса ничего осмысленного не найдёт — к нему
    приклеивается предыдущий. Это запасной путь: обычно вопрос целиком
    переписывает модель при разборе (Plan.standalone)."""
    question = question.strip()
    previous = previous_question(history)
    if previous and len(question.split()) <= FOLLOW_UP_WORDS and _FOLLOW_UP.search(question):
        return f"{previous} {question}"
    return question


def is_follow_up(question: str, plan: Plan | None, history: list[dict]) -> bool:
    """Продолжение ли разговора. Модель при разборе переписывает такой вопрос
    (вопрос целиком отличается от заданного); молчит — судим по словам."""
    if not previous_question(history):
        return False
    if plan is not None and plan.standalone:
        return " ".join(plan.standalone.lower().split()) != " ".join(question.lower().split())
    return len(question.split()) <= FOLLOW_UP_WORDS and bool(_FOLLOW_UP.search(question))


def _queries(values, limit: int) -> list[str]:
    out: list[str] = []
    for q in values or []:
        q = " ".join(str(q).replace('"', " ").split())
        if 2 <= len(q) <= 120 and q.lower() not in {x.lower() for x in out}:
            out.append(q)
    return out[:limit]


def parse_plan(data, question: str, settings: Settings) -> Plan:
    """JSON модели → разбор вопроса. Непонятное — один вопрос без запросов."""
    if not isinstance(data, dict):
        return Plan([Part(question, [])])
    parts = []
    for item in data.get("части") or data.get("parts") or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("вопрос") or item.get("question") or "").split())
        queries = _queries(item.get("запросы") or item.get("queries"), settings.queries)
        if text or queries:
            parts.append(Part(text or question, queries))
    if not parts and data.get("queries"):              # старый вид ответа: только запросы
        parts = [Part(question, _queries(data["queries"], settings.queries))]
    parts = parts[: max(1, settings.parts)] or [Part(question, [])]
    subject = data.get("предмет") or data.get("территория") or data.get("subject")
    subject = " ".join(str(subject).split()) if isinstance(subject, str) else None
    if subject and subject.lower() in {"null", "none", "нет"}:
        subject = None
    collect = data.get("всё", data.get("все", data.get("collect")))
    collect = collect is True or str(collect).lower() in {"true", "да", "1"}
    standalone = " ".join(str(data.get("вопрос_целиком") or data.get("standalone") or "").split())
    topic = data.get("по_теме", data.get("on_topic", True))
    on_topic = not (topic is False or str(topic).lower() in {"false", "нет", "0"})
    return Plan(parts, subject or None, collect, standalone=standalone[:300], on_topic=on_topic)


def plan_question(question: str, history: list[dict], settings: Settings) -> Plan:
    """Разбор вопроса моделью: части, запросы, предмет вопроса, «собрать всё».

    Модель не ответила — вопрос одной частью: ищется сам вопрос, а «собрать
    всё» узнаётся по словам («что известно о…», «перечисли…»)."""
    previous = previous_question(history)
    context = f"Предыдущий вопрос в разговоре: «{previous}»\n" if previous else ""
    user = PLAN_USER.format(question=question.strip(), context=context, n=settings.queries,
                            parts=max(1, settings.parts))
    try:
        # Окно — то же, что у ответа (settings.num_ctx): при другом Ollama выгружает
        # модель и грузит заново, а это секунды на каждом шаге каждого вопроса.
        data = _client(settings).chat_json(PLAN_SYSTEM, user, temperature=0.2, timeout=(10, 90))
    except Exception:  # noqa: BLE001 — без разбора поиск всё равно пойдёт
        return Plan([Part(question, [])], collect=bool(_COLLECT.search(question)))
    return parse_plan(data, question, settings)


def plan_queries(question: str, history: list[dict], settings: Settings) -> list[str]:
    """Только запросы из разбора — для тех, кому части не нужны."""
    plan = plan_question(question, history, settings)
    return [q for p in plan.parts for q in p.queries]


def make_plan(question: str, history: list[dict], settings: Settings, planner) -> Plan:
    """Разбор от planner'а, приведённый к одному виду.

    planner может вернуть Plan или просто список запросов (так было раньше и
    так удобнее в проверках) — тогда вопрос одной частью."""
    raw = planner(question, history, settings)
    if isinstance(raw, Plan):
        plan = raw
    else:
        plan = Plan([Part(question, _queries(raw, 99))],
                    collect=bool(_COLLECT.search(question)))
    if not plan.parts:
        plan.parts = [Part(question, [])]
    if not plan.main:
        if plan.standalone:
            # Модель переписала вопрос так, чтобы он был понятен без разговора.
            plan.main = plan.standalone
        elif plan.subject:
            # Вопрос о названном предмете — самостоятельный, прошлый к нему не клеится.
            plan.main = question.strip()
        else:
            plan.main = search_query(question, history)
    return plan


# --------------------------------------------------------------------------- #
#  Поиск
# --------------------------------------------------------------------------- #
def source_dict(n: int, hit, part=None) -> dict:
    return {
        "n": n,
        "doc_id": hit.doc_id,
        "ord": hit.ord,
        "title": hit.title or hit.doc_id,
        "year": hit.year,
        "journal": hit.journal,
        "authors": hit.authors[:3],
        "pages": hit.pages,
        "headings": hit.headings[-2:],
        "url": hit.url,
        "text": " ".join(hit.text.split()),
        "similarity": hit.similarity,
        "found_by": hit.found_by,
        "part": part,
    }


def merged_search(conn, embedder, queries: list[str], cache: dict | None = None) -> list:
    """Все запросы → одна выдача. Сливается по местам в выдачах (RRF), как
    вектор и слова внутри одного поиска: фрагмент, найденный несколькими
    запросами, поднимается выше. Порог здесь не ставится — его ставит
    find_sources. cache — чтобы один запрос не искать дважды (общая выдача
    и выдачи частей вопроса строятся из одних и тех же запросов)."""
    hits, scores = {}, {}
    for query in queries:
        if cache is not None and query in cache:
            found = cache[query]
        else:
            found = hybrid_search(conn, embedder, query, limit=CANDIDATES, max_per_doc=0)
            if cache is not None:
                cache[query] = found
        for rank, hit in enumerate(found, start=1):
            key = hit.chunk_id
            if key not in hits:
                # Копия: попадания из кэша общие для разных слияний, их не правим.
                hits[key] = replace(hit)
            known = hits[key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if hit.similarity is not None and (known.similarity or 0) < hit.similarity:
                known.similarity = hit.similarity
            if hit.fts_rank is not None and known.fts_rank is None:
                known.fts_rank = hit.fts_rank
            known.fts_strict = known.fts_strict or hit.fts_strict
    ordered = sorted(hits.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in ordered:
        hit.score = scores[hit.chunk_id]
    return ordered


def is_close(hit) -> bool:
    """Близок ли фрагмент: все слова запроса в тексте или похожесть от порога."""
    strict = hit.fts_rank is not None and getattr(hit, "fts_strict", True)
    return strict or (hit.similarity or 0.0) >= MIN_SIMILARITY


def _key(item) -> tuple:
    if isinstance(item, dict):
        return (item.get("doc_id"), item.get("ord"))
    return (item.doc_id, item.ord)


def find_sources(conn, embedder, queries, settings: Settings, start: int = 1,
                 taken: list[dict] | None = None, limit: int | None = None) -> list[dict]:
    """Близкие фрагменты для ответа; дальние отбрасываются совсем.

    queries — список запросов или разбор вопроса (Plan). В разборе из
    нескольких частей у каждой части свои гарантированные места: сначала
    лучшие фрагменты каждой части, затем остаток — лучшим из общей выдачи.
    taken — фрагменты, что уже есть в ответе (из датасета): не повторяются
    и считаются в ограничение «не больше N из статьи»."""
    plan = queries if isinstance(queries, Plan) else Plan([Part("", list(queries))])
    top_k = settings.top_k if limit is None else limit
    taken = taken or []
    cache: dict = {}
    seen = {_key(s) for s in taken}
    per_doc: dict[str, int] = {}
    for s in taken:
        per_doc[s["doc_id"]] = per_doc.get(s["doc_id"], 0) + 1
    chosen: list[tuple] = []

    def take(hit, part, strict: bool = True) -> bool:
        if _key(hit) in seen:
            return False
        if strict and settings.per_doc and per_doc.get(hit.doc_id, 0) >= settings.per_doc:
            return False
        seen.add(_key(hit))
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        chosen.append((hit, part))
        return True

    parts = plan.parts if len(plan.parts) > 1 else []
    if parts:
        quota = max(PART_MIN, (top_k - 2) // len(parts))
        for i, part in enumerate(parts, start=1):
            got = 0
            for hit in merged_search(conn, embedder, part.queries or [part.question], cache):
                if got >= quota or len(chosen) >= top_k:
                    break
                if is_close(hit) and take(hit, i):
                    got += 1
            part.found = got
    ranked = [h for h in merged_search(conn, embedder, plan.queries, cache) if is_close(h)]
    for hit in ranked:
        if len(chosen) >= top_k:
            break
        take(hit, None)
    # Статей в базе мало — добирается лишними кусками тех же статей, но после
    # всех остальных (как в cap_per_doc). Во втором круге — нет: там нужно новое.
    if limit is None:
        for hit in ranked:
            if len(chosen) >= top_k:
                break
            take(hit, None, strict=False)
    if not parts and plan.parts:
        plan.parts[0].found = len(chosen)
    sources = [source_dict(n, h, part) for n, (h, part) in enumerate(chosen, start=start)]
    if conn is not None:
        add_neighbors(conn, sources)
    return sources


# --------------------------------------------------------------------------- #
#  Поиск с моделью: широкий набор → модель отбирает → ищет недостающее
# --------------------------------------------------------------------------- #
@dataclass
class Retrieval:
    """Что нашёл поиск с моделью."""
    sources: list[dict]
    rounds: int = 1                # сколько кругов поиска было
    checked: int = 0               # сколько кандидатов модель прочла
    judged: bool = True            # отбирала модель (False — отбор по порогу, модель молчит)

    def info(self) -> dict:
        return {"rounds": self.rounds, "checked": self.checked, "judged": self.judged}


def is_candidate(hit) -> bool:
    """Годится ли в кандидаты: все слова запроса в тексте или похожесть от мягкого порога."""
    strict = hit.fts_rank is not None and getattr(hit, "fts_strict", True)
    return strict or (hit.similarity or 0.0) >= LOOSE_SIMILARITY


def candidates(conn, embedder, queries: list[str], cache: dict, limit: int = WIDE) -> list:
    return [h for h in merged_search(conn, embedder, queries, cache) if is_candidate(h)][:limit]


def judge_prompt(question: str, plan: Plan, batch: list[dict], accepted: list[dict]) -> str:
    parts = ""
    if len(plan.parts) > 1:
        parts = "Части вопроса:\n" + "\n".join(
            f"{i}) {p.question}" for i, p in enumerate(plan.parts, start=1)) + "\n"
    taken = ""
    if accepted:
        taken = "\nУже отобрано раньше (снова не называй):\n" + "\n".join(
            f"- «{c['hit'].title[:90]}»: {' '.join(c['hit'].text.split())[:160]}"
            for c in accepted[:12]) + "\n"
    lines = []
    for c in batch:
        h = c["hit"]
        year = f", {h.year}" if h.year else ""
        lines.append(f"[{c['id']}] «{(h.title or h.doc_id)[:100]}»{year}: "
                     f"{' '.join(h.text.split())[:CANDIDATE_CHARS]}")
    shown = "\n\n".join(lines) or "— кандидатов нет: по запросам ничего близкого не нашлось."
    done = ", ".join(f"«{q}»" for q in plan.queries + plan.extra) or "—"
    return JUDGE_USER.format(question=question.strip(), parts=parts, taken=taken,
                             candidates=shown, n=EXTRA_QUERIES, done=done)


def judge_fragments(question: str, plan: Plan, batch: list[dict], accepted: list[dict],
                    settings: Settings) -> dict | None:
    """Модель читает кандидатов: {"подходят": [id…], "не_хватает": […], "запросы": […]}.
    Не ответила — None: тогда отбор по порогу, как у обычного поиска."""
    try:
        data = _client(settings).chat_json(
            JUDGE_SYSTEM, judge_prompt(question, plan, batch, accepted),
            temperature=0.1, timeout=(10, 180))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _query_words(query: str) -> set[str]:
    from ..text import name_key
    return set(name_key(query).split())


def is_repeat(query: str, done: list[str]) -> bool:
    """Тот же запрос, что уже был, — по основам слов, с перестановкой и падежами."""
    words = _query_words(query)
    if not words:
        return True
    for other in map(_query_words, done):
        if other and len(words & other) / len(words | other) >= REPEAT_OVERLAP:
            return True
    return False


def _ids(values) -> list[int]:
    out = []
    for v in values if isinstance(values, list) else []:
        m = re.search(r"\d+", str(v))
        if m and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def _select(accepted: list[dict], plan: Plan, settings: Settings, top_k: int,
            taken: list[dict]) -> list[dict]:
    """Отобранное моделью → фрагменты ответа: у каждой части свои места,
    не больше per_doc из статьи (пока есть другие), порядок — как решила модель."""
    per_doc: dict[str, int] = {}
    for s in taken:
        per_doc[s["doc_id"]] = per_doc.get(s["doc_id"], 0) + 1
    chosen: list[dict] = []

    def take(c, strict=True) -> bool:
        if c in chosen or len(chosen) >= top_k:
            return False
        doc = c["hit"].doc_id
        if strict and settings.per_doc and per_doc.get(doc, 0) >= settings.per_doc:
            return False
        per_doc[doc] = per_doc.get(doc, 0) + 1
        chosen.append(c)
        return True

    if len(plan.parts) > 1:
        quota = max(PART_MIN, (top_k - 2) // len(plan.parts))
        for i in range(1, len(plan.parts) + 1):
            got = 0
            for c in accepted:
                if got >= quota:
                    break
                if c["part"] == i and take(c):
                    got += 1
    for c in accepted:
        take(c)
    for c in accepted:              # статей мало — лишние куски тех же статей, в конце
        take(c, strict=False)
    if len(plan.parts) > 1:
        for i, part in enumerate(plan.parts, start=1):
            part.found = sum(c["part"] == i for c in chosen)
    elif plan.parts:
        plan.parts[0].found = len(chosen)
    return chosen


def retrieve(conn, embedder, question: str, plan: Plan, settings: Settings,
             judge=judge_fragments, taken: list[dict] | None = None, limit: int | None = None,
             max_rounds: int = MAX_ROUNDS, extra: list | None = None):
    """Поиск с моделью. Генератор: отдаёт события status, в конце возвращает Retrieval
    (`result = yield from retrieve(…)`).

    Модель читает всех кандидатов круга — пачками по JUDGE_MAX. Новый круг
    ищется, только если модель говорит, чего не хватает, и предлагает запросы,
    которых ещё не было; круг, не добавивший ни одного фрагмента, — последний.

    judge=None или молчащая модель — отбор по порогу близости (find_sources)."""
    taken = list(taken or [])
    if judge is None:                               # без модели — сразу отбор по порогу
        sources = find_sources(conn, embedder, plan, settings, start=len(taken) + 1,
                               taken=taken, limit=limit)
        return Retrieval(sources, rounds=1, checked=0, judged=False)
    top_k = limit or settings.top_k
    cache: dict = {}
    pool: list[dict] = []
    seen = {_key(s) for s in taken}

    def add(hits, part, rnd) -> list[dict]:
        new = []
        for h in hits:
            if _key(h) in seen:
                continue
            seen.add(_key(h))
            c = {"id": len(pool) + 1, "hit": h, "part": part, "round": rnd}
            pool.append(c)
            new.append(c)
        return new

    multi = len(plan.parts) > 1
    # Фрагменты о предмете вопроса из графа — первыми: порога близости у них нет,
    # они найдены не по похожести, а по фактам.
    add(extra or [], None if multi else 1, 1)
    if multi:
        for i, part in enumerate(plan.parts, start=1):
            add(candidates(conn, embedder, part.queries or [part.question], cache, WIDE_PART), i, 1)
        add(candidates(conn, embedder, plan.queries, cache, WIDE_PART), None, 1)
    else:
        add(candidates(conn, embedder, plan.queries, cache, WIDE), 1, 1)

    fresh, accepted = list(pool), []
    rounds, checked, judged = 1, 0, False
    while True:
        before = len(accepted)
        batches = [fresh[i:i + JUDGE_MAX] for i in range(0, len(fresh), JUDGE_MAX)] or [[]]
        if len(batches) > 1 and len(batches[-1]) <= JUDGE_SLACK:
            batches[-2:] = [batches[-2] + batches[-1]]   # 12 + 1 — лишний вызов модели
        missing, proposed, silent, answered = [], [], [], False
        for k, batch in enumerate(batches, start=1):
            part = f" (пачка {k} из {len(batches)})" if len(batches) > 1 else ""
            yield {"type": "status", "text": (f"модель читает найденное: {len(batch)} "
                                              f"фрагментов{part}…" if batch else
                                              "в базе ничего близкого — модель подбирает "
                                              "другие запросы…")}
            data = judge(question, plan, batch, accepted, settings)
            if data is None:                        # модель не ответила на эту пачку
                silent += batch
                continue
            answered = True
            checked += len(batch)
            by_id = {c["id"]: c for c in batch}
            for i in _ids(data.get("подходят") or data.get("relevant")):
                if i in by_id and by_id[i] not in accepted:
                    accepted.append(by_id[i])
            missing += [" ".join(str(m).split()) for m in data.get("не_хватает") or []
                        if str(m).strip()]
            proposed += _queries(data.get("запросы") or data.get("queries"), EXTRA_QUERIES)
        if not answered:
            break                                   # модель молчит — отбор уже сделанный
        judged = True
        if rounds == 1:
            # Модель ответила не на все пачки первого круга — из молчаливых
            # берётся близкое по порогу, как у обычного поиска.
            accepted += [c for c in silent if is_close(c["hit"])]
        if missing or rounds == 1:
            plan.missing = list(dict.fromkeys(missing))[:5]
        if rounds >= max_rounds:
            break
        if rounds > 1 and len(accepted) == before:
            break                                   # новый круг ничего не добавил
        if accepted and not missing:
            break                                   # всего хватает
        done = plan.queries + plan.extra
        queries = []
        for q in proposed:
            if q.lower() not in {d.lower() for d in done + queries} and not is_repeat(q, done):
                queries.append(q)
        queries = queries[:EXTRA_QUERIES]
        if not queries:
            break
        plan.extra += queries
        rounds += 1
        yield {"type": "status", "text": "ищу недостающее: " + " · ".join(queries)}
        fresh = add(candidates(conn, embedder, queries, cache, WIDE), None, rounds)
        if not fresh and not missing:
            break

    if not judged:                                  # модель не ответила — по порогу
        sources = find_sources(conn, embedder, plan, settings, start=len(taken) + 1,
                               taken=taken, limit=limit)
        return Retrieval(sources, rounds=1, checked=0, judged=False)
    chosen = _select(accepted, plan, settings, top_k, taken)
    sources = []
    for n, c in enumerate(chosen, start=len(taken) + 1):
        s = source_dict(n, c["hit"], c["part"] if multi else None)
        if c["round"] > 1:
            s["round"] = c["round"]
        sources.append(s)
    if conn is not None:
        add_neighbors(conn, sources)
    return Retrieval(sources, rounds=rounds, checked=checked, judged=True)


def run(gen):
    """Прогнать генератор-поиск без событий и взять результат (для оценки и проверок)."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


# --------------------------------------------------------------------------- #
#  «Собрать всё» — из датасета: все факты о сущности из графа
# --------------------------------------------------------------------------- #
def _synonyms():
    from ..graph import synonyms
    return synonyms.load()


def resolve_entity(g, hint: str | None, question: str) -> str | None:
    """О чём вопрос: по подсказке модели или по самому вопросу — среди сущностей графа."""
    from ..text import normalize, phrase_pattern

    if hint:
        name = g.resolve(hint)
        if name:
            return name
    # По тексту вопроса: самое длинное имя (или синоним), что в нём встречается.
    text = normalize(question)
    hits = []
    for name in g.entities:
        spellings = [name, *g.syn.groups.get(name, [])]
        for spelled in spellings:
            if len(spelled) < 4:
                continue
            try:
                if phrase_pattern(spelled).search(text):
                    hits.append(name)
                    break
            except re.error:
                continue
    return max(hits, key=len) if hits else None


def collect_dataset(conn, hint: str | None, question: str) -> dict | None:
    """Датасет для вопроса «собрать всё»: факты о сущности и фрагменты к главным.
    None — сущность не узнана, фактов нет (граф не строили) или про неё ничего нет."""
    if conn is None:
        return None
    from ..graph import api, store

    try:
        syn = _synonyms()
        g = api.graph(conn, syn)
        name = resolve_entity(g, hint, question)
        if not name:
            return None
        data = api.dataset(conn, syn, name, g)
        if not data["facts"]:
            return None
        chunk_ids = []
        for row in data["facts"]:
            for cid in row["chunk_ids"][:1]:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
            if len(chunk_ids) >= DATASET_EVIDENCE:
                break
        frags = store.fragments(conn, chunk_ids)
        data["evidence"] = [frags[c] for c in chunk_ids if c in frags]
        data["pass"] = store.summary(conn)
        return data
    except Exception:  # noqa: BLE001 — нет фактов или словаря: обычный поиск
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


def graph_candidates(conn, hint: str | None, question: str) -> list:
    """Фрагменты, где сказано о предмете вопроса, — по графу, а не по похожести.

    Редкое название («Персияновский разлом») векторный поиск находит плохо:
    для него это просто похожие слова. Граф знает, из каких фрагментов о нём
    выписаны факты. Эти фрагменты — кандидаты: брать ли их, решает модель, как
    и для найденного поиском. Графа нет или предмет не узнан — пусто."""
    if conn is None or not hint:
        return []
    from ..graph import api, store
    from ..index.search import Hit

    try:
        syn = _synonyms()
        g = api.graph(conn, syn)
        name = resolve_entity(g, hint, question)
        if not name:
            return []
        ids: list[int] = []
        for row in api.dataset(conn, syn, name, g)["facts"]:
            for cid in row["chunk_ids"]:
                if cid not in ids:
                    ids.append(cid)
            if len(ids) >= GRAPH_CANDIDATES:
                break
        ids = ids[:GRAPH_CANDIDATES]
        frags = store.fragments(conn, ids)
    except Exception:  # noqa: BLE001 — нет графа или словаря: обойдёмся поиском
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return []
    return [Hit(chunk_id=cid, doc_id=f["doc_id"], ord=f["ord"], text=f["text"],
                headings=f["headings"], pages=f["pages"], title=f["title"], year=f["year"],
                journal=f["journal"], url=f["url"], authors=f["authors"], found_by="граф")
            for cid in ids if (f := frags.get(cid))]


def _count(n: int) -> str:
    last2, last = n % 100, n % 10
    word = ("статья" if last == 1 and last2 != 11 else
            "статьи" if 2 <= last <= 4 and not 12 <= last2 <= 14 else "статей")
    return f"{n} {word}"


def dataset_summary(data: dict) -> str:
    """Факты о сущности одним текстом — «фрагмент» для модели."""
    includes = data.get("includes") or []
    # «Полный список» — только если модель прошла все фрагменты базы. Граф
    # строится ночью порциями, и новые статьи попадают в него не сразу.
    passed = data.get("pass") or {}
    done, total = passed.get("passed") or 0, passed.get("chunks") or 0
    scope = ("это полный список по графу" if not total or done >= total else
             f"это всё, что есть в графе, но модель прошла пока {done} из {total} фрагментов "
             "базы — в остальных может быть ещё")
    lines = [f"Факты о «{data['name']}»"
             + (f" (вместе с тем, что в неё входит: {', '.join(includes)})" if includes else "")
             + " — выписаны моделью из статей базы и проверены по цитатам, а не найдены "
             f"поиском; {scope}. Статей: {data['documents']}."]
    for row in data["facts"][:DATASET_ITEMS * 2]:
        fact = (f"{row['about']} — {row['relation']} — {row['other']}" if row["direction"] == "→"
                else f"{row['other']} — {row['relation']} — {row['about']}")
        lines.append(f"- {fact} ({_count(row['documents'])})")
    more = len(data["facts"]) - DATASET_ITEMS * 2
    if more > 0:
        lines.append(f"- и ещё {more}")
    return "\n".join(lines)


def dataset_sources(data: dict, start: int = 1) -> list[dict]:
    """Сводка фактов — первым фрагментом, за ней фрагменты статей к главным фактам."""
    out = [{
        "n": start, "doc_id": f"датасет:{data['name']}", "ord": -1,
        "title": f"Факты из графа: {data['name']}", "year": None, "journal": None,
        "authors": [], "pages": [], "headings": [], "url": "",
        "text": dataset_summary(data), "similarity": None, "found_by": "датасет",
        "part": None, "kind": "датасет",
    }]
    for e in data.get("evidence") or []:
        out.append({
            "n": start + len(out), "doc_id": e["doc_id"], "ord": e["ord"],
            "title": e.get("title") or e["doc_id"], "year": e.get("year"),
            "journal": e.get("journal"), "authors": e.get("authors") or [],
            "pages": e.get("pages") or [], "headings": e.get("headings") or [],
            "url": e.get("url") or "", "text": e["text"], "similarity": None,
            "found_by": "датасет", "part": None,
        })
    return out


def add_neighbors(conn, sources: list[dict]) -> None:
    """Короткому фрагменту — продолжение из той же статьи и того же раздела.

    Поиск находит самый похожий кусок, а он бывает обрывком: заголовок и две
    строки, дальше «см. ниже». Для развёрнутого ответа модели нужно то, что
    идёт следом. Добавленный текст идёт под тем же номером — ссылка [n]
    по-прежнему ведёт на ту же статью.
    """
    short = [s for s in sources if len(s["text"]) < SHORT_FRAGMENT]
    if not short:
        return
    wanted = {(s["doc_id"], s["ord"] + 1) for s in short}
    taken = {(s["doc_id"], s["ord"]) for s in sources}
    wanted -= taken
    if not wanted:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, ord, text, headings, pages FROM chunks
                WHERE (doc_id, ord) IN (SELECT * FROM unnest(%s::text[], %s::int[]))
                """,
                ([d for d, _ in wanted], [o for _, o in wanted]),
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — без соседей ответ всё равно будет
        return
    after = {(r[0], r[1]): r for r in rows}
    for s in short:
        row = after.get((s["doc_id"], s["ord"] + 1))
        if not row:
            continue
        same_section = list(row[3] or [])[-1:] == list(s.get("headings") or [])[-1:]
        if not same_section:
            continue
        extra = " ".join((row[2] or "").split())[:NEIGHBOR_CHARS]
        s["text"] = f"{s['text']} {extra}"
        s["pages"] = sorted(set(s.get("pages") or []) | set(row[4] or []))
        s["with_next"] = True


# --------------------------------------------------------------------------- #
#  Сообщения для модели
# --------------------------------------------------------------------------- #
def fragment_block(source: dict) -> str:
    where = []
    if source.get("year"):
        where.append(str(source["year"]))
    if source.get("pages"):
        where.append("с. " + ", ".join(str(p) for p in source["pages"][:3]))
    text = source["text"]
    limit = DATASET_CHARS if source.get("kind") == "датасет" else FRAGMENT_CHARS
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    head = f"[{source['n']}] «{source['title']}»" + (f" ({', '.join(where)})" if where else "")
    return f"{head}\n{text}"


GENERAL_MARK = ("(Этот ответ был не из базы знаний, а из общих знаний модели — "
                "как источник его не используй.)\n")


def past_turns(history: list[dict]) -> list[dict]:
    """Прошлые реплики — чтобы понимать вопросы вдогонку. Последние два обмена,
    каждая укорочена.

    Из прошлых ответов убираются номера [n]: они вели на прошлые фрагменты, а в
    новом сообщении те же номера — у других, и модель переносила их. Ответ «без
    базы» (mode) помечается: иначе сказанное из общих знаний попадало в ответ по базе."""
    past = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    out = []
    for m in past[-HISTORY_TURNS * 2:]:
        content = str(m["content"])
        if m["role"] == "assistant":
            content = _CITE_MARK.sub("", content)
            if m.get("mode") == "без базы":
                content = GENERAL_MARK + content
        out.append({"role": m["role"], "content": content[:HISTORY_CHARS]})
    return out


def build_messages(question: str, sources: list[dict], history: list[dict],
                   model: str = MODEL, plan: Plan | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}, *past_turns(history)]
    fragments = "\n\n".join(fragment_block(s) for s in sources)
    notes = []
    summary = next((s for s in sources if s.get("kind") == "датасет"), None)
    if summary is not None:
        notes.append(
            f"Фрагмент [{summary['n']}] — все факты о предмете вопроса из графа знаний "
            "(выписаны из статей базы). Перечисли из него всё, сгруппировав по смыслу, со "
            f"ссылкой [{summary['n']}]; подробности, примеры и числа бери из остальных "
            "фрагментов, с их номерами.")
    if plan is not None and len(plan.parts) > 1:
        lines = "\n".join(f"{i}) {p.question}" for i, p in enumerate(plan.parts, start=1))
        notes.append("Вопрос состоит из частей:\n" + lines + "\nОтветь на каждую часть "
                     "отдельным разделом. Если по какой-то части во фрагментах ничего нет — "
                     "так и напиши в её разделе.")
    articles = {s["doc_id"]: s["title"] for s in sources if s.get("kind") != "датасет"}
    if len(articles) == 1 and len(sources) > 1:
        notes.append(f"Все фрагменты — из одной статьи «{next(iter(articles.values()))}». "
                     "Скажи об этом в ответе и не выдавай сказанное в ней об одном районе или "
                     "объекте за общее правило.")
    asked = question.strip()
    if plan is not None and plan.standalone and plan.standalone.lower() != asked.lower():
        asked += f"\n(Это продолжение разговора; вопрос целиком: {plan.standalone})"
    note = ("\n\n" + "\n\n".join(notes)) if notes else ""
    user = (f"Фрагменты из базы знаний:\n\n{fragments}\n\n"
            f"Вопрос: {asked}{note}\n\nОтвечай только по этим фрагментам, со ссылками.")
    messages.append({"role": "user", "content": no_think(model, user)})
    return messages


def _tokens(messages: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + 1


def fit_budget(question: str, sources: list[dict], history: list[dict], settings: Settings,
               plan: Plan | None = None) -> tuple[list[dict], list[dict]]:
    """Фрагменты и история, которые влезают в окно модели вместе с ответом.

    Сначала уходят старые обмены разговора, потом — последние фрагменты (самые
    слабые: порядок — как отобрала модель), но не меньше трёх. Нумерация не
    сбивается: отбрасываются только с конца."""
    budget = settings.num_ctx - ANSWER_RESERVE
    history = [m for m in history if m.get("role") in ("user", "assistant")
               and m.get("content")][-HISTORY_TURNS * 2:]

    def size():
        return _tokens(build_messages(question, sources, history, settings.model, plan))

    while history and size() > budget:
        history = history[2:]
    sources = list(sources)
    while len(sources) > 3 and size() > budget:
        sources.pop()
    return sources, history


def general_messages(question: str, history: list[dict], model: str = MODEL) -> list[dict]:
    """Сообщения для ответа без базы: без фрагментов, с запретом на ссылки."""
    messages = [{"role": "system", "content": GENERAL_SYSTEM}, *past_turns(history)]
    user = question.strip()
    messages.append({"role": "user", "content": no_think(model, user)})
    return messages


def _norm(text: str) -> str:
    return re.sub(r"[«»\"'*_.\s]+", " ", text.lower().replace("ё", "е")).strip()


def is_refusal(start: str) -> bool:
    """Начинается ли ответ с фразы «в базе знаний нет ответа» (правило 3)."""
    return _norm(start).startswith(_norm(NO_ANSWER))


def citations(answer: str, count: int) -> tuple[list[int], list[int]]:
    """Какие номера фрагментов в ответе есть и какие из них выдуманы."""
    found: set[int] = set()
    for group in _CITE.findall(answer):
        for part in re.split(r"[,;]", group):
            if part.strip().isdigit():
                found.add(int(part))
    used = sorted(n for n in found if 1 <= n <= count)
    unknown = sorted(n for n in found if not 1 <= n <= count)
    return used, unknown


def coverage(answer: str) -> tuple[int, int]:
    """Сколько утверждений ответа подкреплены ссылкой: (со ссылкой, всего).

    Утверждение — предложение или пункт списка длиннее 40 знаков. Не считаются
    фразы о самих фрагментах («По данным статей базы…», «Во фрагментах нет…»):
    это не факт из статьи, ссылка к ним не нужна. Ссылка может стоять и сразу
    после точки — она относится к предыдущему предложению.
    """
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", answer)
    text = re.sub(r"([.!?…])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)", r" \2\1", text)
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    claims = [p for p in parts if len(_CITE.sub("", p).strip()) > 40
              and not (_ABOUT_SOURCES.search(p) and not _CITE.search(p))]
    cited = [p for p in claims if _CITE.search(p)]
    return len(cited), len(claims)


class ThinkFilter:
    """Срезает блок <think>…</think>, если модель его всё-таки прислала.

    Старые Ollama не знают поля think и отдают размышления прямо в тексте;
    Qwen3 с /no_think присылает пустой блок. Приходит это кусками, поэтому
    фильтр держит состояние между кусками.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.state = "start"          # start → (think →) text

    def feed(self, chunk: str) -> str:
        if self.state == "text":
            return chunk
        self.buffer += chunk
        if self.state == "start":
            head = self.buffer.lstrip()
            if not head:
                return ""
            if "<think>".startswith(head[:7]) and len(head) < 7:
                return ""             # ещё не ясно, начало ли это блока
            if head.startswith("<think>"):
                self.state = "think"
            else:
                self.state = "text"
                out, self.buffer = self.buffer.lstrip(), ""
                return out
        if self.state == "think" and "</think>" in self.buffer:
            rest = self.buffer.split("</think>", 1)[1].lstrip()
            self.state, self.buffer = "text", ""
            return rest
        return ""

    def flush(self) -> str:
        out = self.buffer if self.state != "think" else ""
        self.buffer = ""
        return out


# --------------------------------------------------------------------------- #
#  Модель
# --------------------------------------------------------------------------- #
def _client(settings: Settings) -> Ollama:
    return Ollama(model=settings.model, host=settings.host, timeout=settings.timeout,
                  temperature=settings.temperature, num_ctx=settings.num_ctx)


def ollama_status(host: str = OLLAMA, model: str = MODEL, timeout: int = 5) -> dict:
    """Жива ли Ollama и скачана ли модель — для подсказки на странице."""
    return Ollama(model=model, host=host).status(timeout=timeout)


def stream_ollama(messages: list[dict], settings: Settings) -> Iterator[dict]:
    """Ответ Ollama кусками: {"text": …} и в конце {"done": True, "tokens": …}."""
    return _client(settings).stream(messages)


# --------------------------------------------------------------------------- #
#  Всё вместе
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """Всё, что нужно для ответа: разбор вопроса, датасет, найденное."""
    plan: Plan
    sources: list[dict]
    retrieval: Retrieval
    dataset: dict | None = None
    graph: int = 0                 # сколько кандидатов дал граф (вопрос о названном предмете)

    def info(self) -> dict:
        plan = self.plan
        return {"queries": plan.queries + plan.extra, "standalone": plan.standalone,
                "graph": self.graph,
                "parts": plan.describe() if len(plan.parts) > 1 else [],
                "extra": plan.extra, "missing": plan.missing,
                "dataset": self.dataset["name"] if self.dataset else None,
                **self.retrieval.info()}


def prepare(conn, embedder, question: str, history: list[dict], settings: Settings,
            planner=plan_question, judge=judge_fragments, datasets=collect_dataset,
            graph_hits=graph_candidates):
    """Разбор вопроса и поиск с моделью. Генератор событий status; в конце — Prepared."""
    yield {"type": "status", "text": "разбираю вопрос…"}
    plan = make_plan(question, history, settings, planner)

    # «Собрать всё» о предмете вопроса — из датасета; поиск только добавляет.
    data = None
    if plan.collect and datasets is not None:
        yield {"type": "status", "text": "собираю факты из графа…"}
        data = datasets(conn, plan.subject, question)
    base = dataset_sources(data) if data else []

    # Вопрос о названном предмете — фрагменты о нём из графа в кандидаты.
    extra = []
    if not data and plan.subject and graph_hits is not None and judge is not None:
        extra = graph_hits(conn, plan.subject, question) or []
        if extra:
            yield {"type": "status", "text": f"беру из графа фрагменты о «{plan.subject}»…"}

    yield {"type": "status", "text": (f"ищу в базе по {len(plan.parts)} частям вопроса…"
                                      if len(plan.parts) > 1 else "ищу в базе…")}
    limit = max(settings.top_k - (len(base) - 1), 3) if base else None
    # Вопрос не о геологии — один круг: проверить, что в базе правда пусто, и всё.
    rounds = MAX_ROUNDS if plan.on_topic else 1
    found = yield from retrieve(conn, embedder, question, plan, settings, judge, base, limit,
                                rounds, extra)
    return Prepared(plan, base + found.sources, found, data, len(extra))


def answer(conn, embedder, question: str, history: list[dict] | None = None,
           settings: Settings | None = None, llm=stream_ollama,
           planner=plan_question, judge=judge_fragments, datasets=collect_dataset,
           prepared: Prepared | None = None) -> Iterator[dict]:
    """Ответ на вопрос событиями (см. начало файла).

    llm, planner, judge и datasets подменяются в проверках: planner может
    вернуть и просто список запросов, judge — None (отбор по порогу), datasets
    (conn, предмет, вопрос) — датасет или None. prepared — уже найденное
    (оценка ищет один раз и для проверки поиска, и для ответа)."""
    settings = settings or Settings()
    history = history or []
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "error": "пустой вопрос"}
        return

    started = time.monotonic()
    if prepared is None:
        prepared = yield from prepare(conn, embedder, question, history, settings,
                                      planner, judge, datasets)
    sources, plan, info = prepared.sources, prepared.plan, prepared.info()

    # Разговор модели нужен только для вопроса вдогонку. В самостоятельном
    # вопросе прошлый ответ — помеха: на живом прогоне в ответ о Персияновском
    # разломе попала фраза из прошлого ответа об Ишимбинском, со ссылкой [1].
    if not is_follow_up(question, plan, history):
        history = []

    text, tokens = "", None
    if sources:
        sources, turns = fit_budget(question, sources, history, settings, plan)
        yield {"type": "status", "text": "пишу ответ по найденному…"}
        stream = _clean(llm(build_messages(question, sources, turns, settings.model, plan),
                            settings))
        try:
            # Первые слова держатся, пока не ясно, не отказ ли это: если модель
            # прочла фрагменты и ответа в них нет, статьи не показываются вовсе.
            head, finished = "", False
            for part in stream:
                if "done" in part:
                    tokens, finished = part["done"], True
                    break
                head += part["text"]
                if len(head.strip()) >= len(NO_ANSWER) + 10:
                    break
            if head.strip() and not is_refusal(head):
                yield {"type": "sources", "sources": sources, **info}
                text = head
                if head:
                    yield {"type": "token", "text": head}
                if not finished:
                    for part in stream:
                        if "done" in part:
                            tokens = part["done"]
                            break
                        text += part["text"]
                        yield {"type": "token", "text": part["text"]}
                used, unknown = citations(text, len(sources))
                cited, claims = coverage(text)
                yield {"type": "done", "mode": "база", "used": used, "unknown": unknown,
                       "coverage": [cited, claims], "model": settings.model,
                       "seconds": round(time.monotonic() - started, 1), "tokens": tokens,
                       **info}
                return
        except LLMError as exc:
            yield {"type": "error", "error": str(exc)}
            return
        finally:
            stream.close()

    if not plan.on_topic and not settings.general_off_topic:
        # Не о геологии и в базе пусто — короткий отказ, без ответа из общих знаний.
        yield {"type": "general", "text": OFF_TOPIC, "off_topic": True, **info}
        yield {"type": "done", "mode": "без базы", "used": [], "unknown": [], "coverage": [0, 0],
               "model": settings.model, "seconds": round(time.monotonic() - started, 1),
               "tokens": None, "off_topic": True, **info}
        return

    # В базе нет: ни статей, ни номеров — ответ модели с пометкой.
    yield {"type": "general", "text": NOT_IN_BASE, **info}
    yield {"type": "status", "text": "отвечаю без базы…"}
    try:
        for part in _clean(llm(general_messages(question, history, settings.model), settings)):
            if "done" in part:
                tokens = part["done"]
                break
            text += part["text"]
            yield {"type": "token", "text": part["text"]}
    except LLMError as exc:
        yield {"type": "error", "error": str(exc)}
        return
    yield {"type": "done", "mode": "без базы", "used": [], "unknown": [], "coverage": [0, 0],
           "model": settings.model, "seconds": round(time.monotonic() - started, 1),
           "tokens": tokens, **info}


def _clean(parts) -> Iterator[dict]:
    """Поток модели без блока размышлений: {"text": …} и в конце {"done": токены}."""
    think = ThinkFilter()
    try:
        for part in parts:
            if part.get("done"):
                tail = think.flush()
                if tail:
                    yield {"text": tail}
                yield {"done": part.get("tokens")}
                return
            piece = think.feed(part.get("text", ""))
            if piece:
                yield {"text": piece}
        tail = think.flush()
        if tail:
            yield {"text": tail}
        yield {"done": None}
    finally:
        close = getattr(parts, "close", None)
        if close:
            close()
