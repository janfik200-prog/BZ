"""Спросить чат-бота из терминала.

    python -m georag.chat.cli "как выделяют рудные узлы по линеаментам?"

То же, что вкладка «Чат-бот» в браузере, только один вопрос. Модель
эмбеддингов грузится на каждый запуск, поэтому для разговора удобнее
браузер: python georag.py web.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..common import add_db_args, add_embedder_args, add_llm_args
from ..index import db
from ..index.embed import build_embedder
from .answer import Settings, answer, ollama_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Вопрос к чат-боту базы знаний")
    parser.add_argument("question")
    add_llm_args(parser)
    add_db_args(parser)
    add_embedder_args(parser)
    args = parser.parse_args(argv)

    settings = Settings(model=args.model, host=args.ollama_host)
    status = ollama_status(settings.host, settings.model)
    if not status["ok"]:
        print(status["error"], file=sys.stderr)
        return 1

    print("Загружаю модель эмбеддингов…", file=sys.stderr)
    embedder = build_embedder(args.embedder, device=args.device)
    sources: list[dict[str, Any]] = []
    code = 0
    with db.connect(args.dsn) as conn:
        for event in answer(conn, embedder, args.question, settings=settings):
            if event["type"] == "status":
                print(event["text"], file=sys.stderr)
            elif event["type"] == "sources":
                sources = event["sources"]
                if event.get("dataset"):
                    print(
                        f"Собрано из фактов графа о «{event['dataset']}» — по всем статьям базы",
                        file=sys.stderr,
                    )
                for i, part in enumerate(event.get("parts") or [], start=1):
                    print(
                        f"Часть {i}: {part['question']} — фрагментов {part['found']}",
                        file=sys.stderr,
                    )
                print("Искал: " + " · ".join(f"«{q}»" for q in event["queries"]), file=sys.stderr)
                if event.get("judged") is False:
                    print(
                        "Модель при отборе не ответила — фрагменты взяты по близости",
                        file=sys.stderr,
                    )
                elif event.get("checked"):
                    print(
                        f"Модель прочла найденное (фрагментов: {event['checked']}), взяла "
                        f"{len(sources)}; кругов поиска: {event.get('rounds', 1)}",
                        file=sys.stderr,
                    )
                if event.get("extra"):
                    missing = event.get("missing") or []
                    print(
                        "Искал ещё"
                        + (f" (не хватало: {'; '.join(missing)})" if missing else "")
                        + ": "
                        + " · ".join(f"«{q}»" for q in event["extra"]),
                        file=sys.stderr,
                    )
                print(file=sys.stderr)
            elif event["type"] == "general":
                print("Искал: " + " · ".join(f"«{q}»" for q in event["queries"]), file=sys.stderr)
                print(f"\n!!! {event['text']}\n")
            elif event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "revised":
                # В терминале напечатанное не стереть — показываем, что убрано.
                print("\n\nУбрано как фразы без ссылки на фрагмент (слова модели, не статей):")
                for phrase in event["removed"]:
                    print(f"  − {phrase}")
            elif event["type"] == "error":
                print(f"\nОшибка: {event['error']}", file=sys.stderr)
                code = 1
            elif event["type"] == "done":
                print("\n")
                used = set(event["used"])
                for s in sources:
                    mark = "→" if s["n"] in used else " "
                    pages = f", с. {s['pages'][0]}" if s.get("pages") else ""
                    year = f", {s['year']}" if s.get("year") else ""
                    print(f"{mark} [{s['n']}] {s['title'][:80]}{year}{pages}")
                    if s.get("url"):
                        print(f"       {s['url']}")
                if event["unknown"]:
                    print(
                        f"\nВнимание: модель сослалась на несуществующие фрагменты "
                        f"{event['unknown']} — эти утверждения не подтверждены."
                    )
                cited, claims = event.get("coverage") or (0, 0)
                if claims:
                    print(f"\nСсылка на фрагмент — у {cited} из {claims} утверждений.")
                if event.get("mode") == "без базы":
                    print("[Ответ модели из общих знаний — не из базы знаний]")
                if event.get("model"):
                    print(f"{event['model']}, {event['seconds']} с")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
