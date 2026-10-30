"""Сопоставление названия места из вопроса (по-русски) с написанием в статьях базы.

Три способа, на вход — только русское написание (как спрашивает геолог):
  current    — concepts.resolve: ключ транслитерации и близость векторов BGE-M3;
  symphonym  — фонетические векторы Symphonym v8 (.local/external/symphonym/v8, CC-BY-4.0);
  geonames   — другие написания из справочника GeoNames (allCountries.zip, CC-BY-4.0),
               оставлены те, что встречаются среди названий базы.
Верно — если среди трёх предложенных есть ожидаемое название (по началу слова).
Ожидаемые написания записаны до прогона, по названиям статей базы.

    place_names_eval.py [--scan]   (--scan — заново пройти GeoNames, около 5 мин)
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from georag.index import concepts as C  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.text import normalize  # noqa: E402

DSN = "postgresql://georag:georag@127.0.0.1:5433/georag_v14"
SYM = ROOT / ".local" / "external" / "symphonym" / "v8"
GEO = ROOT / ".local" / "external" / "geonames" / "allCountries.zip"
RES = HERE / "results"
CACHE = RES / "geonames_hits.json"
TOP = 3

# Русское написание → ожидаемое в базе (латиница).
CASES = {
    "Лудзин": "Lujing",
    "Минас-ду-Камакуан": "Camaquã",
    "Колдар": "Koldar",
    "Букантау": "Bukantau",
    "Нахичевань": "Nakhchivan",
    "Селигдар": "Seligdar",
    "Северный Вьетнам": "Northern Vietnam",
    "Восточный Донбасс": "Eastern Donbass",
    "Манкаян": "Mankayan",
    "Джалалабад": "Jalalabad",
    "Кадуна": "Kaduna",
    "Насарава": "Nasarawa",
    "Жакобина": "Jacobina",
    "Оканаган": "Okanagan",
    "Джеральдтон": "Geraldton",
    "Саньшаньдао": "Sanshandao",
    "Леймэньгоу": "Leimengou",
    "Бангка-Белитунг": "Bangka",
    "Камерун": "Cameroon",
    "Пиндвара": "Pindwara",
}


def _ok(expected: str, got: list[str]) -> bool:
    words = C._words(expected)
    return any(C.has(expected, C._words(g)) or C.has(g, words) for g in got)


def latin_names(conn, emb) -> list[str]:
    base = C._base(conn, emb)
    return [n for n in base.names if not any("а" <= ch.lower() <= "я" for ch in n)]


def by_current(conn, emb, ru: str) -> list[str]:
    return C.resolve(conn, emb, [ru])[:TOP]


def by_symphonym(model, names: list[str], vecs, ru: str) -> list[str]:
    import numpy as np

    q = model.embed(C._distinct(ru), "ru")
    sims = vecs @ q
    return [names[i] for i in np.argsort(-sims)[:TOP]]


def scan_geonames(keys: set[str]) -> dict[str, list[str]]:
    """Один проход по allCountries: записи, где среди названий есть русское из вопроса."""
    hits: dict[str, list[str]] = {}
    with zipfile.ZipFile(GEO) as z, z.open("allCountries.txt") as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            parts = line.split("\t")
            alts = parts[3].split(",") if len(parts) > 3 and parts[3] else []
            names = [parts[1], parts[2], *alts]
            for n in names:
                k = normalize(n)
                if k in keys:
                    hits.setdefault(k, []).extend(x for x in names if x)
    return {k: list(dict.fromkeys(v)) for k, v in hits.items()}


def by_geonames(hits: dict[str, list[str]], names: list[str], ru: str) -> list[str]:
    alts = hits.get(normalize(C._distinct(ru)), []) + hits.get(normalize(ru), [])
    known = {normalize(n): n for n in names}
    out = [known[normalize(a)] for a in alts if normalize(a) in known]
    if not out:  # частью названия базы («Nakhchivan» в «Nakhchivan AR»)
        out = [n for n in names for a in alts if len(a) > 3 and normalize(a) in normalize(n).split()]
    return list(dict.fromkeys(out))[:TOP]


def main() -> None:
    sys.path.insert(0, str(SYM))
    from inference import SymphonymModel  # noqa: PLC0415 — модель из скачанной папки

    t0 = time.monotonic()
    emb = Embedder()
    model = SymphonymModel(model_dir=SYM)
    keys = {normalize(k) for k in CASES} | {normalize(C._distinct(k)) for k in CASES}
    if "--scan" in sys.argv or not CACHE.exists():
        t1 = time.monotonic()
        hits = scan_geonames(keys)
        CACHE.write_text(json.dumps(hits, ensure_ascii=False, indent=0), "utf-8")
        print(f"GeoNames пройден за {time.monotonic() - t1:.0f} с, найдено {len(hits)} из {len(keys)}")
    hits = json.loads(CACHE.read_text("utf-8"))
    res: dict = {"cases": {}}
    with connect(DSN) as conn:
        names = latin_names(conn, emb)
        vecs = model.batch_embed([(C._distinct(n), "en") for n in names])
        for ru, want in CASES.items():
            got = {"current": by_current(conn, emb, ru),
                   "symphonym": by_symphonym(model, names, vecs, ru),
                   "geonames": by_geonames(hits, names, ru)}
            res["cases"][ru] = {"expected": want, **{m: {"top": g, "ok": _ok(want, g)} for m, g in got.items()}}
            print(ru, "→", want, "|", {m: (v["ok"], v["top"][:2]) for m, v in res["cases"][ru].items() if m != "expected"})
    for m in ("current", "symphonym", "geonames"):
        res[m] = sum(c[m]["ok"] for c in res["cases"].values())
    res["any"] = sum(any(c[m]["ok"] for m in ("current", "symphonym", "geonames")) for c in res["cases"].values())
    res["n"] = len(CASES)
    res["minutes"] = round((time.monotonic() - t0) / 60, 1)
    (RES / "place_names_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print({k: v for k, v in res.items() if k != "cases"})


if __name__ == "__main__":
    main()
