"""WP10. Выборка пар (факт, вопрос) для эксперимента V и лист разметки — до запуска методов.

Источник фактов — граф боевой базы (копия georag_v14, только чтение): факты, которые
выписала Qwen3:14b по всем фрагментам (E0, 09–10.10.2026), с голосами confirm.py.
Пары отбираются так, как их получил бы граф без контроля контекста (V0): все факты о
золоте подходят к любому вопросу о золоте. Стратификация — по области исследования
статьи, заданной вручную по названию и аннотации (STUDY ниже; это метаданные разметки,
а не вывод проверяемого кода):

  in   — статья о территории вопроса или её части;
  near — статья о соседней территории той же страны/щита (сиблинг, «север/юг»);
  out  — статья о другой территории;
  gen  — обзор или методика без конкретной территории.

DEV/TEST — по семействам источников (fusion.source_families): препринт и журнальная
версия, части одной работы — в одном разбиении. Семейство → DEV, если
sha256("v14-split|" + family_id) mod 100 < 35. Результат — data/pairs.jsonl,
annotation/annotation_sheet.csv (без предсказаний методов), split_manifest.json,
source_manifest.csv, source_families.csv.

    .venv/Scripts/python.exe research/v14/experiments/build_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.evidence import fusion  # noqa: E402
from georag.evidence.claims import features  # noqa: E402
from georag.evidence.georegistry import CONTAINS, DISJOINT, EXACT, OVERLAP, PART_OF, load  # noqa: E402
from georag.evidence.provenance import normalize  # noqa: E402
from georag.text import split_sentences  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
V14 = ROOT / "research" / "v14"
DATA = V14 / "data"
ANN = V14 / "annotation"
SEED = 20261029
DEV_SHARE = 35

# Область исследования статьи — вручную по названию/аннотации; kind: case / regional / review / methods.
STUDY: dict[str, tuple[tuple[str, ...], str]] = {
    "10.47765_0869-5997-2024-10017": (("R:RU:URAL:MID",), "case"),
    "openalex-w7213480904": (("R:RU:URAL:MURZINKA",), "case"),
    "10.47765_0869-7175-2024-10029": (("R:RU:CHULKOVSKY",), "case"),
    "10.18799_24131830_2024_12_4577": (("R:RU:BAYAG",), "case"),
    "10.47765_0869-5997-2025-10005": (("R:RU:FE:ARCTICBELT",), "regional"),
    "10.56195_20793332-2025-25-5-49-54": (("R:RU:ARYLAKH",), "case"),
    "10.18799_24131830_2022_8_3708": (("R:RU:SVETLOE",), "case"),
    "10.34078_1814-0998-2026-1-12-18": (("R:RU:MOGOT",), "case"),
    "10.55959_msu0579-9406-4-2025-64-4-71-83": (("R:RU:CHUKOTKA",), "case"),
    "10.25283_2223-4594-2017-2-62-75": (("R:RU:ARCTIC",), "regional"),
    "10.17308_geology.2018.1_1447": (("R:RU:YENISEI",), "case"),
    "10.18799_24131830_2025_7_5053": (("R:RU:SUKHOILOG",), "case"),
    "10.18799_24131830_2025_10_5304": (("R:RU:SUKHOILOG",), "case"),
    "10.17223_25421379_31_5": (("R:RU:TRANSBAIKAL:E",), "regional"),
    "10.52349_0869-7892_2024_100_147-164": (("R:BALTIC",), "regional"),
    "10.24411_0869-7175-2019-10008": (("R:RU:ZYGYRKOL",), "case"),
    "10.47765_0869-7175-2025-10017": (("R:ALTAI",), "regional"),
    "10.31857_s0016777024050011": ((), "review"),
    "10.1134_s1075701525600574": ((), "review"),
    "10.1038_s41598-026-60088-7": (("R:EG:ATALLASID",), "case"),
    "10.33271_mining20.03.010": (("R:EG:UMMEISH",), "case"),
    "10.1038_s41598-026-52061-1": (("R:EG:DAWI",), "case"),
    "10.1038_s41598-026-45708-6": (("R:EG:KHARIT",), "case"),
    "10.1007_s43994-025-00234-5": (("R:EG:UMARA",), "case"),
    "10.1038_s41598-023-34531-y": (("R:EG:ED",), "case"),
    "10.31248_gjees2024.162": (("R:SD:ARTOLI",), "case"),
    "10.31248_gjees2024.170": (("R:SD:TOHAMIYAM",), "case"),
    "10.30958_ajs.13-1-3": (("R:SD:RSH",), "case"),
    "10.30967_ijcrset_bikila-argeta-ayele_210": (("R:ET:MENGE",), "case"),
    "10.1007_s42461-026-01551-x": (("R:NG:MARU",), "case"),
    "10.62292_10.62292_njp.v34i3.2025.400": (("R:NG:ILESA",), "case"),
    "10.51584_ijrias.2026.11010045": (("R:NG:EKITI",), "case"),
    "10.21203_rs.3.rs-10216943_v1": (("R:NG:WAMBA",), "case"),
    "openalex-w7218290360": (("R:NG:BORGU",), "case"),
    "10.64290_bima.v10i2.1611": (("R:NG",), "case"),
    "10.70882_josrar.2026.v3i5.246": (("R:NG:PATEGI",), "case"),
    "10.1007_s44288-026-00567-6": (("R:NG:NASARAWA",), "case"),
    "10.12789_geocanj.2023.50.202": (("R:CA:OKANAGAN",), "case"),
    "10.21203_rs.3.rs-5384377_v1": (("R:CA:GERALDTON",), "case"),
    "10.6084_m9.figshare.c.8387100.v1": (("R:UK:DUNS",), "case"),
    "10.1007_s00126-023-01220-9": (("R:BR:JACOBINA",), "case"),
    "10.1007_s00126-025-01395-3": (("R:GUIANA:RORAIMA",), "case"),
    "10.22564_brjg.v43i1.2335": (("R:BR:CASTRO",), "case"),
    "10.33677_ggianas20250100142": (("R:UZ:ALMALYK",), "case"),
    "10.52349_0869-7892_2026_106_43-52": (("R:UZ:BUKANTAU",), "case"),
    "10.47390_ydif-y2026v2i12_n06": (("R:UZ:TUKMAN",), "case"),
    "10.33677_ggianas20250100147": (("R:AZ:NAKHCHIVAN",), "case"),
    "10.52349_0869-7892_2024_97_109-117": (("R:VN:N",), "regional"),
    "10.31857_s0205-96142019559-71": (("R:MN:OLONOVOOT",), "case"),
    "10.3389_feart.2026.1770354": (("R:CN:SANSHANDAO",), "case"),
    "10.31643_2028_6445.27": (("R:KZ:E",), "regional"),
    "10.29150_jhrs.v7.4.p189-211": ((), "review"),
    "10.4314_cps.v13i1.4": ((), "review"),
    "10.30574_gjeta.2025.25.2.0334": ((), "methods"),
    "10.5194_egusphere-2026-1293": ((), "methods"),
    "10.5194_gmd-19-8755-2026": ((), "methods"),
    "10.46698_vnc.2021.23.67.003": (("R:DONBASS",), "case"),
    "10.46698_vnc.2023.84.41.008": (("R:DONBASS:E",), "regional"),
    "10.24028_gj.v47i2.322500": (("R:UA:KIROVOHRAD",), "case"),
}


def q(i, text, region, *, quant="EXISTS", mode="FACTUAL", pol="AFFIRM", analog=False, deposit="gold", n_in=3, n_near=1, n_out=3, n_gen=1):
    return dict(query_id=i, raw_text=text, target_region_id=region, target_deposit_type=deposit,
                requested_quantifier=quant, requested_mode=mode, requested_polarity=pol, allow_analogies=analog,
                quota=dict(inn=n_in, near=n_near, out=n_out, gen=n_gen))


QUERIES = [
    q("Q01", "Какие факты о золоторудной минерализации известны для Среднего Урала?", "R:RU:URAL:MID"),
    q("Q02", "Что известно о коренных источниках и составе россыпного золота Мурзинской группы россыпей?", "R:RU:URAL:MURZINKA", n_near=2),
    q("Q03", "Какие признаки и связи золотого оруденения описаны для Якутии?", "R:RU:YAKUTIA"),
    q("Q04", "Какие признаки золотого оруденения установлены на Чулковском участке Селигдарского рудного поля?", "R:RU:CHULKOVSKY", n_near=2),
    q("Q05", "Характерна ли для золоторудных объектов Якутии в целом связь золота с определёнными минералами и структурами?", "R:RU:YAKUTIA", quant="TYPICAL", n_in=4),
    q("Q06", "Какие факты о золотом оруденении известны для Дальнего Востока России?", "R:RU:FE"),
    q("Q07", "Что известно о золотом оруденении Чукотки?", "R:RU:CHUKOTKA", n_near=2),
    q("Q08", "Какие особенности золоторудного месторождения Арылах (Магаданская область) описаны?", "R:RU:MAGADAN", n_near=2),
    q("Q09", "Какие признаки золотого оруденения описаны для Енисейского кряжа?", "R:RU:YENISEI"),
    q("Q10", "Что известно о строении и золотом оруденении месторождения Сухой Лог?", "R:RU:SUKHOILOG"),
    q("Q11", "Какие этапы и связи золотого оруденения установлены в Восточном Забайкалье?", "R:RU:TRANSBAIKAL:E"),
    q("Q12", "Что известно о золото-урановом оруденении Балтийского щита?", "R:BALTIC", deposit="gold;uranium"),
    q("Q13", "Какие факты о золотом оруденении известны для территории России?", "R:RU", n_in=4, n_near=0),
    q("Q14", "Какие черты золотого оруденения типичны для России в целом?", "R:RU", quant="TYPICAL", n_in=4, n_near=0),
    q("Q15", "Какие структурные признаки золотого оруденения известны для Восточной пустыни Египта?", "R:EG:ED"),
    q("Q16", "Какие признаки золотого оруденения описаны для Центральной Восточной пустыни Египта?", "R:EG:ED:C", n_near=2),
    q("Q17", "Типичен ли для Египта в целом сдвиговый контроль золотого оруденения?", "R:EG", quant="TYPICAL", n_in=4),
    q("Q18", "Какие факты о золотом оруденении известны для Холмов Красного моря в Судане?", "R:SD:RSH", n_near=2),
    q("Q19", "Какие аналоги по другим районам Нубийского щита известны для золотого оруденения Восточной пустыни Египта?",
      "R:EG:ED", mode="COMPARATIVE", analog=True, n_in=2, n_near=3, n_out=2),
    q("Q20", "Какие факты о золоторудной минерализации известны для Нигерии?", "R:NG"),
    q("Q21", "Что известно о золотом оруденении Юго-Западной Нигерии?", "R:NG:SW", n_near=2),
    q("Q22", "Что известно о составе россыпного золота Британской Колумбии?", "R:CA:BC"),
    q("Q23", "Что известно о детритовом золоте района Данс в Шотландии?", "R:UK:SCOT"),
    q("Q24", "Какие признаки золотого оруденения описаны для Бразилии?", "R:BR"),
    q("Q25", "Какие аналоги метаконгломератового золота Якобины известны в других районах?", "R:BR:JACOBINA",
      mode="COMPARATIVE", analog=True, n_in=2, n_near=0, n_out=4),
    q("Q26", "Какие факты о золотом оруденении известны для Узбекистана?", "R:UZ"),
    q("Q27", "Что известно о золотом оруденении Нахчыванской Автономной Республики?", "R:AZ:NAKHCHIVAN"),
    q("Q28", "Какие признаки золотого оруденения описаны для Северного Вьетнама?", "R:VN:N"),
    q("Q29", "Что известно о золотом оруденении района Джералдтон (Онтарио)?", "R:CA:ON"),
    q("Q30", "Какие признаки золотого оруденения описаны для зеленокаменного пояса Менге (Эфиопия)?", "R:ET"),
    q("Q31", "Что известно о золотом оруденении Южной Монголии?", "R:MN"),
    q("Q32", "Какие гипотезы образования золотого оруденения опубликованы для месторождения Сухой Лог?", "R:RU:SUKHOILOG",
      mode="RESEARCH_REVIEW", n_in=4, n_near=0, n_out=2),
    q("Q33", "Какие предположения о происхождении золота опубликованы для Среднего Урала?", "R:RU:URAL:MID",
      mode="RESEARCH_REVIEW", n_in=4, n_near=0, n_out=2),
    q("Q34", "Какие факты о золотом оруденении известны для Гайаны (супергруппа Рорайма)?", "R:GUIANA:RORAIMA"),
    q("Q35", "Какие факты о золотом оруденении известны для провинции Цзяодун (Китай)?", "R:CN"),
]

GOLD_SQL = r"""
SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.quote, f.votes, f.model, f.rel_type,
       c.text, c.headings, c.pages, c.ord, d.title, d.doi, d.language,
       (coalesce(f.votes,0) >= 3 OR EXISTS (SELECT 1 FROM facts o WHERE o.doc_id <> f.doc_id AND
         ((o.src_key = f.src_key AND o.dst_key = f.dst_key) OR (o.src_key = f.dst_key AND o.dst_key = f.src_key)))) AS accepted
FROM facts f JOIN chunks c ON c.id = f.chunk_id JOIN documents d ON d.doc_id = f.doc_id
WHERE (f.src || ' ' || f.dst || ' ' || f.quote) ~* '(gold|золот|auriferous|\mau\M)'
  AND f.doc_id = ANY(%s)
ORDER BY f.id
"""


def split_of(family: str) -> str:
    h = int(hashlib.sha256(f"v14-split|{family}".encode()).hexdigest(), 16) % 100
    return "DEV" if h < DEV_SHARE else "TEST"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stratum(reg, doc_regions: tuple[str, ...], kind: str, target: str) -> str:
    if not doc_regions or kind in ("review", "methods"):
        return "gen"
    rels = {reg.relation(r, target) for r in doc_regions}
    if rels & {EXACT, PART_OF}:
        return "inn"
    if rels & {CONTAINS, OVERLAP}:
        return "near"
    # Сиблинги и соседи: общий предок в двух уровнях — «near», иначе «out».
    for r in doc_regions:
        anc_r = {r, *reg.ancestors(r)}
        anc_t = {target, *reg.ancestors(target)}
        common = (anc_r & anc_t) - {"R:RU"} if target != "R:RU" else anc_r & anc_t
        if common and any(reg.areas[c].level not in ("country", "macroregion") for c in common):
            return "near"
        if reg.areas[r].country and reg.areas[r].country == reg.areas[target].country and target not in ("R:RU",):
            return "near"
    return "out"


def main() -> None:
    reg = load(DATA / "regional_registry.csv", DATA / "registry_aliases.csv")
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, title, authors, doi, language, sha256, parser, year FROM documents ORDER BY doc_id")
        docs = [dict(zip(("doc_id", "title", "authors", "doi", "language", "sha256", "parser", "year"), r)) for r in cur.fetchall()]
        cur.execute("SELECT doc_id, text FROM chunks ORDER BY doc_id, ord")
        chunks: dict[str, list[str]] = defaultdict(list)
        for d, t in cur.fetchall():
            chunks[d].append(t or "")
        cur.execute(GOLD_SQL, (list(STUDY),))
        cols = [c.name for c in cur.description]
        facts = [dict(zip(cols, r)) for r in cur.fetchall()]

    families = fusion.source_families(docs, chunks)
    fam_of = fusion.family_of(families)
    with (DATA / "source_families.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family_id", "kind", "doc_ids", "evidence", "split"])
        for fam in families:
            w.writerow([fam.family_id, fam.kind, ";".join(fam.doc_ids), fam.evidence, split_of(fam.family_id)])

    with (DATA / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "title", "doi", "lang", "license", "permission_to_publish", "doc_family", "region_name",
                    "region_ids", "doc_kind", "deposit_type", "pdf_sha256", "parsed_sha256", "parser_version", "split",
                    "in_study_set", "source_file"])
        for d in docs:
            regions, kind = STUDY.get(d["doc_id"], ((), ""))
            parsed = hashlib.sha256("\n".join(normalize(t) for t in chunks[d["doc_id"]]).encode()).hexdigest()
            w.writerow([d["doc_id"], d["title"], d["doi"] or "", d["language"] or "", "UNVERIFIED",
                        "NOT_CONFIRMED: цитаты до 1 предложения для исследования; перед публикацией проверить лицензию",
                        fam_of[d["doc_id"]], "; ".join(reg.areas[r].name for r in regions), ";".join(regions), kind,
                        "gold" if d["doc_id"] in STUDY else "", d["sha256"], parsed, f"{d['parser']}@unrecorded",
                        split_of(fam_of[d["doc_id"]]), "yes" if d["doc_id"] in STUDY else "no", "DB:georag_v14.documents"])

    # --- кандидаты по вопросам ----------------------------------------------------------------
    rng = random.Random(SEED)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for fct in facts:
        by_doc[fct["doc_id"]].append(fct)
    pairs = []
    for query in QUERIES:
        target = query["target_region_id"]
        pools: dict[str, list[dict]] = defaultdict(list)
        for doc_id, rows in by_doc.items():
            regions, kind = STUDY[doc_id]
            pools[stratum(reg, regions, kind, target)].extend(rows)
        chosen: list[tuple[str, dict]] = []
        used_chunks: set[int] = set()
        for name, n in (("inn", query["quota"]["inn"]), ("near", query["quota"]["near"]),
                        ("out", query["quota"]["out"]), ("gen", query["quota"]["gen"])):
            pool = sorted(pools.get(name, []), key=lambda r: r["id"])
            rng.shuffle(pool)
            # Разные статьи прежде всего: по одному факту на статью, пока хватает.
            picked_docs: set[str] = set()
            for row in pool:
                if len([c for c in chosen if c[0] == name]) >= n:
                    break
                if row["chunk_id"] in used_chunks or row["doc_id"] in picked_docs:
                    continue
                chosen.append((name, row))
                used_chunks.add(row["chunk_id"])
                picked_docs.add(row["doc_id"])
        # Один дополнительный — с предположением или отрицанием, если такие есть среди кандидатов.
        special = [r for name in ("inn", "near", "out") for r in pools.get(name, [])
                   if r["chunk_id"] not in used_chunks and features(r["quote"], r["relation"]).modality == "HYPOTHESIS"
                   or (r["chunk_id"] not in used_chunks and features(r["quote"], r["relation"]).polarity == "NEGATE")]
        special.sort(key=lambda r: r["id"])
        if special:
            row = rng.choice(special)
            chosen.append(("special", row))
            used_chunks.add(row["chunk_id"])
        for name, row in chosen:
            fam = fam_of[row["doc_id"]]
            sents = split_sentences(" ".join((row["text"] or "")[:4000].split()))
            quote_norm = normalize(row["quote"])
            idx = next((i for i, s in enumerate(sents) if normalize(s) and normalize(s)[:50] in quote_norm), None)
            before = " ".join(sents[max(0, idx - 2) : idx]) if idx is not None else ""
            pairs.append({
                "sample_id": f"{query['query_id']}-F{row['id']}",
                "query_id": query["query_id"],
                "stratum": name,
                "fact_id": row["id"],
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "source_family": fam,
                "split": split_of(fam),
                "src": row["src"], "relation": row["relation"], "dst": row["dst"],
                "quote": row["quote"], "votes": row["votes"], "accepted": bool(row["accepted"]),
                "model": row["model"], "rel_type": row["rel_type"],
                "headings": list(row["headings"] or []), "pages": list(row["pages"] or []),
                "title": row["title"], "doi": row["doi"], "language": row["language"],
                "context_before": before, "chunk_text": " ".join((row["text"] or "").split()),
                "doc_study_regions": list(STUDY[row["doc_id"]][0]), "doc_kind": STUDY[row["doc_id"]][1],
            })

    DATA.mkdir(parents=True, exist_ok=True)
    ANN.mkdir(parents=True, exist_ok=True)
    with (DATA / "queries.json").open("w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in qq.items() if k != "quota"} | {"target_region_name": reg.areas[qq["target_region_id"]].name}
                   for qq in QUERIES], f, ensure_ascii=False, indent=1)
    with (DATA / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    qtext = {qq["query_id"]: qq for qq in QUERIES}
    with (ANN / "annotation_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query_id", "query_text", "target_region", "quantifier", "mode", "analogies",
                    "title", "headings", "pages", "context_before", "quote", "src", "relation", "dst", "split",
                    "gold_support", "gold_region_of_fact", "gold_quantifier", "gold_modality", "gold_polarity",
                    "gold_applicability", "reason_notes", "annotator_id", "adjudication_status"])
        for p in pairs:
            qq = qtext[p["query_id"]]
            w.writerow([p["sample_id"], p["query_id"], qq["raw_text"], reg.areas[qq["target_region_id"]].name,
                        qq["requested_quantifier"], qq["requested_mode"], qq["allow_analogies"], p["title"],
                        " / ".join(p["headings"]), ",".join(map(str, p["pages"])), p["context_before"], p["quote"],
                        p["src"], p["relation"], p["dst"], p["split"], "", "", "", "", "", "", "", "", ""])
    manifest = {
        "created": "2026-10-29",
        "seed": SEED,
        "dev_share_percent": DEV_SHARE,
        "rule": "split = DEV if sha256('v14-split|'+family_id) mod 100 < 35 else TEST",
        "n_families": len(families),
        "families": {f.family_id: {"split": split_of(f.family_id), "docs": list(f.doc_ids), "kind": f.kind} for f in families},
        "n_pairs": len(pairs),
        "pairs_by_split": {s: sum(p["split"] == s for p in pairs) for s in ("DEV", "TEST")},
        "pairs_by_stratum": {s: sum(p["stratum"] == s for p in pairs) for s in ("inn", "near", "out", "gen", "special")},
        "registry_version": reg.version,
        "sha256": {"pairs.jsonl": sha_file(DATA / "pairs.jsonl"), "queries.json": sha_file(DATA / "queries.json"),
                   "regional_registry.csv": sha_file(DATA / "regional_registry.csv"),
                   "registry_aliases.csv": sha_file(DATA / "registry_aliases.csv")},
    }
    (DATA / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_families", "n_pairs", "pairs_by_split", "pairs_by_stratum", "registry_version")}, ensure_ascii=False))
    leaks = defaultdict(set)
    for p in pairs:
        leaks[p["source_family"]].add(p["split"])
    assert all(len(v) == 1 for v in leaks.values())
    multi = [f for f in families if len(f.doc_ids) > 1]
    for f_ in multi:
        print("family", f_.kind, f_.doc_ids, f_.evidence)


if __name__ == "__main__":
    main()
