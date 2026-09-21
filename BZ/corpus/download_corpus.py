"""Скачать тестовый корпус в data/pdf — для проверки качества разбора PDF.

    python corpus/download_corpus.py

Список статей берётся из corpus.json. Это единственное место в проекте, где PDF
сохраняются на диск, и сделано это намеренно: чтобы мерить разбор по эталонам
из golden/, нужны одни и те же файлы от прогона к прогону. Обычная добыча
статей PDF не хранит.

Уже скачанные файлы пропускаются. Между запросами пауза, чтобы КиберЛенинка
не приняла скачивание за атаку.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "pdf"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def main() -> int:
    import requests

    articles = json.loads((ROOT / "corpus" / "corpus.json").read_text(encoding="utf-8"))
    TARGET.mkdir(parents=True, exist_ok=True)
    failed = []
    for art in articles:
        path = TARGET / f"{art['id']}.pdf"
        if path.exists() and path.stat().st_size > 0:
            print(f"  есть     {path.name}")
            continue
        try:
            response = requests.get(art["pdf"], headers={"User-Agent": UA}, timeout=120)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("пришёл не PDF — возможно, страница-заглушка")
            path.write_bytes(response.content)
            print(f"  скачано  {path.name} ({len(response.content) // 1024} КБ)")
        except Exception as exc:  # noqa: BLE001 — одна статья не должна останавливать остальные
            failed.append(art["id"])
            print(f"  не вышло {path.name}: {exc}")
        time.sleep(2)
    print(f"\nГотово. Файлы в {TARGET}")
    if failed:
        print("Не скачались: " + ", ".join(failed) + ". Их можно скачать руками по ссылкам "
              "из corpus/README.md и положить в data/pdf.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
