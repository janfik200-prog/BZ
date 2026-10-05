"""Чистка папки добычи от статей не по теме.

Ранние прогоны шли со старым отбором, и в data/acquired осели работы про
маркетинг, палеогеографию и звёздные скопления. Этот скрипт прогоняет по уже
добытому тот же доменный фильтр, что теперь стоит на входе: статья, где ни в
названии, ни в тексте нет ни одного слова рудной тематики, к базе отношения
не имеет.

Ничего не удаляется. Отбракованное переезжает в data/acquired/_отсев,
оттуда его можно вернуть или посмотреть глазами.

    python georag.py clean            # только показать, что уйдёт
    python georag.py clean --apply    # действительно перенести
    python georag.py clean "тема"     # ещё и спросить Qwen3: по теме ли каждая статья

С темой статьи, прошедшие доменный фильтр, читает Qwen3 — по тем же правилам,
что и при добыче (название и начало текста вместо аннотации). Так убираются
статьи, которые попали в базу по старым правилам отбора.

После --apply база знаний забывает перенесённые статьи при следующем
`python georag.py ingest`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ..common import add_llm_args
from .heuristic import has_domain_term

REJECTED_DIR = "_отсев"
# Смотрим название и начало текста: если рудная терминология есть, она встретится
# в первых страницах — в аннотации или во введении.
TEXT_HEAD = 4000


def judge(acquired: Path, doc_id: str) -> tuple[bool, str]:
    meta_path = acquired / f"{doc_id}.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return True, f"не читается ({exc}) — оставляем"

    title = meta.get("title") or doc_id
    text = ""
    md_path = acquired / f"{doc_id}.md"
    if md_path.exists():
        try:
            text = md_path.read_text(encoding="utf-8")[:TEXT_HEAD]
        except OSError:
            text = ""

    return has_domain_term(f"{title} {text}"), title


def recheck(
    acquired: Path, keep: list[tuple[str, str]], topic: str, model: str, host: str
) -> tuple[list[Any], list[Any]]:
    """Спросить Qwen3 про каждую оставшуюся статью. Не отвечает — ничего не трогаем."""
    from .llm import OllamaLLM
    from .models import Candidate

    llm = OllamaLLM(model=model, host=host)
    status = llm.status()
    if not status["ok"]:
        print(f"Qwen3 недоступна ({status['error']}) — проверяю только доменным фильтром.\n")
        return keep, []
    print(f"Qwen3 читает {len(keep)} статей по теме «{topic}»…")
    cands = []
    for doc_id, title in keep:
        head = ""
        md = acquired / f"{doc_id}.md"
        if md.exists():
            head = " ".join(md.read_text(encoding="utf-8", errors="replace")[:TEXT_HEAD].split())
        cands.append(Candidate(source="-", external_id=doc_id, title=title, abstract=head[:1500]))
    for start in range(0, len(cands), 6):
        try:
            llm.filter_batch(topic, cands[start : start + 6])
        except Exception as exc:  # noqa: BLE001 — партия без решения остаётся
            print(f"  модель не ответила на партию: {type(exc).__name__}")
    kept, dropped = [], []
    for (doc_id, title), cand in zip(keep, cands, strict=True):  # по кандидату на статью
        if cand.relevant is False:
            dropped.append((doc_id, f"{title} — Qwen3: {cand.reason}"))
        else:
            kept.append((doc_id, title))
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="Убрать из добытого статьи не по теме")
    parser.add_argument("--acquired", type=Path, default=Path("data/acquired"))
    parser.add_argument("--apply", action="store_true", help="перенести, а не только показать")
    parser.add_argument("--topic", default="", help="тема: проверить статьи ещё и моделью")
    add_llm_args(parser)
    args = parser.parse_args()

    if not args.acquired.exists():
        print(f"Папки {args.acquired} нет", file=sys.stderr)
        return 1

    # Рядом лежат <doc_id>.chunks.json и <doc_id>.docling.json — это не карточки
    # статей, и принимать их за отдельные документы нельзя.
    doc_ids = sorted(
        p.name[: -len(".json")]
        for p in args.acquired.glob("*.json")
        if not p.name.startswith("acquire-report-")
        and not p.name.endswith((".chunks.json", ".docling.json"))
    )
    if not doc_ids:
        print("В папке нет добытых статей.")
        return 0

    out = args.acquired / REJECTED_DIR
    keep: list[tuple[str, str]] = []
    drop: list[tuple[str, str]] = []

    for doc_id in doc_ids:
        ok, title = judge(args.acquired, doc_id)
        (keep if ok else drop).append((doc_id, title))

    if args.topic and keep:
        keep, by_model = recheck(args.acquired, keep, args.topic, args.model, args.ollama_host)
        drop += by_model

    print(f"Просмотрено статей: {len(doc_ids)}")
    print(f"По теме: {len(keep)}")
    print(f"Не по теме: {len(drop)}\n")

    for _doc_id, title in drop:
        print(f"  — {title[:70]}")

    if not drop:
        print("Чистить нечего.")
        return 0

    if not args.apply:
        print(f"\nЭто был просмотр. Чтобы перенести их в {out}, запустите с --apply")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    moved = 0
    for doc_id, _ in drop:
        for path in args.acquired.glob(f"{doc_id}.*"):
            target = out / path.name
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
            moved += 1

    print(f"\nПеренесено файлов: {moved} → {out}")
    print("Записи в index.jsonl оставлены: так эти статьи не будут скачиваться заново.")
    print("Из базы знаний они уйдут при следующем: python georag.py ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
