"""Чистка папки добычи от статей не по теме.

Ранние прогоны шли со старым отбором, и в data/acquired осели работы про
маркетинг, палеогеографию и звёздные скопления. Этот скрипт прогоняет по уже
добытому тот же доменный фильтр, что теперь стоит на входе: статья, где ни в
названии, ни в тексте нет ни одного слова рудной тематики, к базе отношения
не имеет.

Ничего не удаляется. Отбракованное переезжает в data/acquired/_отсев,
оттуда его можно вернуть или посмотреть глазами.

    python scripts/clean_acquired.py            # только показать, что уйдёт
    python scripts/clean_acquired.py --apply    # действительно перенести
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.acquire.heuristic import has_domain_term  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Убрать из добытого статьи не по теме")
    parser.add_argument("--acquired", type=Path, default=Path("data/acquired"))
    parser.add_argument("--apply", action="store_true", help="перенести, а не только показать")
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
    keep, drop = [], []

    for doc_id in doc_ids:
        ok, title = judge(args.acquired, doc_id)
        (keep if ok else drop).append((doc_id, title))

    print(f"Просмотрено статей: {len(doc_ids)}")
    print(f"По теме: {len(keep)}")
    print(f"Не по теме: {len(drop)}\n")

    for doc_id, title in drop:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
