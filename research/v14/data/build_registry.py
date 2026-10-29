"""Справочник территорий v1 (WP05) → regional_registry.csv и registry_aliases.csv.

Собран ДО формирования выборки пар (факт, вопрос) из кандидатов-топонимов корпуса
(mine_toponyms.py → toponym_candidates.csv, _toponyms_by_doc.txt) и названий статей.
После сборки замораживается: версия = отпечаток двух CSV (georegistry.load).

Основание каждой связи включения (basis):
  SOURCE   — так сказано в статье корпуса (basis_doc_id, basis_quote — название статьи);
  ADMIN    — административное деление (страна, область, штат);
  GEOLOGY  — общепринятое региональное геологическое деление, внесено при курировании.
review_status у всех строк CURATED_UNREVIEWED: справочник собран кодирующим агентом
(Claude), а не геологом; до публикации его должен просмотреть специалист.
official_id не выдумывается — пусто. Перекрытия (overlaps) — для пар «геологическая
единица ↔ административная», где нет вложенности: по ним DIRECT не выдаётся.

settings — теги геологической обстановки: аналогия (ANALOG) обоснована, только если
у двух территорий общий тег и совпадает тип оруденения (applicability.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# id, имя, уровень, родители, страна, типы оруденения, обстановка, перекрытия, основание,
# doc основания, цитата основания, синонимы
A = []


def area(i, name, level, parents="", country="", deposits="", settings="", overlaps="",
         basis="ADMIN", doc="", quote="", aliases=()):
    A.append(dict(area_id=i, canonical_name=name, level=level, parent_ids=parents,
                  country=country, deposit_types=deposits, settings=settings,
                  overlaps=overlaps, basis=basis if parents else "", basis_doc_id=doc,
                  basis_quote=quote, review_status="CURATED_UNREVIEWED",
                  _aliases=list(aliases)))


T = "SOURCE"
G = "GEOLOGY"

# ---------------------------------------------------------------- Россия
area("R:RU", "Россия", "country", aliases=["Russia", "Russian Federation", "Российская Федерация", "РФ"])
area("R:RU:URAL", "Урал", "region", "R:RU", "RU", basis=G, aliases=["Ural", "Urals", "Уральский регион"])
area("R:RU:URAL:MID", "Средний Урал", "region", "R:RU:URAL", "RU", "gold", "ural_orogen", basis=T,
     doc="10.47765_0869-5997-2024-10017", quote="gold mineralization of the Middle Urals",
     aliases=["Middle Urals", "Middle Ural", "Central Urals"])
area("R:RU:URAL:MID:AUERBAKH", "Ауэрбаховский рудный район", "ore_district", "R:RU:URAL:MID", "RU", "gold;copper",
     "ural_orogen", basis=T, doc="10.47765_0869-5997-2024-10017", quote="Ауэрбаховского рудного района (Средний Урал)",
     aliases=["Auerbakh ore district", "Ауэрбаховский", "Турьинско-Ауэрбаховский рудный район"])
area("R:RU:URAL:VORONTSOVSKOE", "Воронцовское месторождение", "deposit", "R:RU:URAL:MID:AUERBAKH", "RU", "gold",
     basis=T, doc="10.47765_0869-5997-2024-10017", quote="Воронцовского месторождения",
     aliases=["Воронцовское", "Vorontsovskoe", "Vorontsovskoye"])
area("R:RU:URAL:PESHCHERNOE", "Пещерное месторождение", "deposit", "R:RU:URAL:MID:AUERBAKH", "RU", "gold",
     basis=T, doc="10.47765_0869-5997-2024-10017", quote="месторождение Пещерное",
     aliases=["Пещерное", "Peshchernoe", "Peschernoe", "Peshchernoye"])
area("R:RU:URAL:MURZINKA", "Мурзинская группа россыпей", "placer_group", "R:RU:URAL:MID", "RU", "gold",
     basis=T, doc="openalex-w7213480904", quote="Мурзинской группы россыпей (Средний Урал)",
     aliases=["Murzinka group", "Murzinka", "Мурзинка", "Мурзинские россыпи", "Murzinka River"])
area("R:RU:SIBPLAT", "Сибирская платформа", "craton", "R:RU", "RU", basis=G,
     aliases=["Siberian platform", "Siberian craton", "Сибирский кратон"])
area("R:RU:YAKUTIA", "Республика Саха (Якутия)", "admin_region", "R:RU", "RU",
     aliases=["Якутия", "Yakutia", "Republic of Sakha", "Sakha", "Республика Саха"])
area("R:RU:ALDAN", "Алданский щит", "shield", "R:RU:SIBPLAT", "RU", "gold", "aldan_mesozoic_alkaline_au",
     overlaps="R:RU:YAKUTIA", basis=G, aliases=["Aldan shield", "Алдано-Становой щит", "Aldan-Stanovoy shield", "Центральный Алдан", "Central Aldan"])
area("R:RU:ELKON", "Томмот-Эльконская зона", "zone", "R:RU:ALDAN", "RU", "gold;uranium", basis=T,
     doc="10.47765_0869-7175-2024-10029", quote="Томмот-Эльконской зоны",
     aliases=["Эльконская зона", "Elkon zone", "Tommot-Elkon zone", "Эльконский горст"])
area("R:RU:SELIGDAR", "Селигдарское рудное поле", "ore_field", "R:RU:ALDAN;R:RU:YAKUTIA", "RU", "gold",
     basis=T, doc="10.47765_0869-7175-2024-10029", quote="Seligdar Ore Field, Republic of Sakha (Yakutia)",
     aliases=["Seligdar ore field", "Seligdar", "Селигдар", "Селигдарское"])
area("R:RU:CHULKOVSKY", "Чулковский участок", "site", "R:RU:SELIGDAR", "RU", "gold", basis=T,
     doc="10.47765_0869-7175-2024-10029", quote="Chulkovsky Site of the Seligdar Ore Field",
     aliases=["Chulkovsky site", "Chulkovsky", "Чулковское рудопроявление", "Чулковская площадь", "Чулковское"])
area("R:RU:LEBEDINY", "Лебединский рудный узел", "ore_cluster", "R:RU:ALDAN", "RU", "gold", basis=G,
     aliases=["Lebediny ore cluster", "Лебединский"])
area("R:RU:FE", "Дальний Восток России", "macroregion", "R:RU", "RU",
     aliases=["Дальний Восток", "Russian Far East", "Far East of Russia", "Дальневосточный регион"])
area("R:RU:ARCTIC", "Арктическая зона России", "macroregion", "R:RU", "RU", overlaps="R:RU:FE;R:RU:YAKUTIA;R:RU:SIBPLAT",
     aliases=["Arctic zone of Russia", "Russian Arctic", "Российская Арктика", "Арктическая зона РФ", "Russian Arctic zone"])
area("R:RU:FE:ARCTICBELT", "Арктический пояс Дальнего Востока", "belt", "R:RU:FE", "RU", "gold", "ru_ne_mesozoic_orogenic_au",
     overlaps="R:RU:ARCTIC;R:RU:YAKUTIA", basis=T, doc="10.47765_0869-5997-2025-10005",
     quote="Gold mineralization of the Arctic Belt of the Russian Far East",
     aliases=["Arctic Belt of the Russian Far East", "Arctic belt", "Арктический пояс"])
area("R:RU:CHUKOTKA", "Чукотка", "admin_region", "R:RU:FE", "RU", "gold", "ru_ne_mesozoic_orogenic_au",
     overlaps="R:RU:FE:ARCTICBELT;R:RU:ARCTIC", aliases=["Chukotka", "Чукотский автономный округ", "Чукотский АО", "Чукотский сектор"])
area("R:RU:MAYSKOE_CHUK", "Майское месторождение (Чукотка)", "deposit", "R:RU:CHUKOTKA", "RU", "gold", basis=G,
     aliases=["Майское", "Mayskoe", "Maiskoe", "Mayskoye"])
area("R:RU:KEKURA", "Кекура", "deposit", "R:RU:CHUKOTKA", "RU", "gold", basis=G, aliases=["Kekura", "месторождение Кекура"])
area("R:RU:KUPOL", "Купол", "deposit", "R:RU:CHUKOTKA", "RU", "gold;silver", basis=G, aliases=["Kupol", "месторождение Купол"])
area("R:RU:KULAR", "Куларо-Полоусненский район", "ore_district", "R:RU:YAKUTIA", "RU", "gold", overlaps="R:RU:FE:ARCTICBELT",
     basis=G, aliases=["Куларо-Полоусненский", "Kular", "Кулар", "Kular-Polousny"])
area("R:RU:MAGADAN", "Магаданская область", "admin_region", "R:RU:FE", "RU", "gold", "ru_ne_mesozoic_orogenic_au",
     aliases=["Magadan region", "Магадан", "Magadan", "Магаданская"])
area("R:RU:ARYLAKH", "Арылах", "deposit", "R:RU:MAGADAN", "RU", "gold", basis=T, doc="10.56195_20793332-2025-25-5-49-54",
     quote="Arylyakh deposit (Magadan region)", aliases=["Arylyakh", "Арылахское", "месторождение Арылах"])
area("R:RU:YANAKOLYMA", "Яно-Колымский пояс", "belt", "R:RU:FE", "RU", "gold", "ru_ne_mesozoic_orogenic_au",
     overlaps="R:RU:YAKUTIA;R:RU:MAGADAN", basis=G, aliases=["Yana-Kolyma belt", "Яно-Колымский", "Yana-Kolyma", "Колыма", "Kolyma"])
area("R:RU:VERKHOYANSK", "Верхояно-Колымская складчатая система", "fold_system", "R:RU:FE", "RU", "gold",
     "ru_ne_mesozoic_orogenic_au", overlaps="R:RU:YAKUTIA;R:RU:YANAKOLYMA;R:RU:MAGADAN", basis=G,
     aliases=["Verkhoyansk-Kolyma fold system", "Verkhoyansk-Kolyma", "Верхояно-Колымская", "Верхоянье", "Verkhoyansk"])
area("R:RU:BAYAG", "Баягское рудное поле", "ore_field", "R:RU:VERKHOYANSK;R:RU:YAKUTIA", "RU", "gold", basis=T,
     doc="10.18799_24131830_2024_12_4577", quote="Verkhoyansk-Kolyma fold system (Bayag ore field, Yakutia)",
     aliases=["Bayag ore field", "Bayag", "Баягское"])
area("R:RU:NEZHDANINSKOE", "Нежданинское месторождение", "deposit", "R:RU:YAKUTIA", "RU", "gold", basis=G,
     aliases=["Nezhdaninskoe", "Nezhdaninskoye", "Нежданинское"])
area("R:RU:PRIAMURYE", "Приамурская провинция", "province", "R:RU:FE", "RU", "gold", basis=G,
     aliases=["Приамурье", "Приамурская", "Amur province", "Priamurye", "Amur region"])
area("R:RU:MOGOT", "Моготский рудно-россыпной узел", "ore_cluster", "R:RU:PRIAMURYE", "RU", "gold", basis=T,
     doc="10.34078_1814-0998-2026-1-12-18", quote="Моготского рудно-россыпного узла Приамурской провинции",
     aliases=["Моготский", "Mogot", "Джалон", "Джалонский массив"])
area("R:RU:KHABAROVSK", "Хабаровский край", "admin_region", "R:RU:FE", "RU", aliases=["Khabarovsk krai", "Khabarovsk", "Хабаровский"])
area("R:RU:ULYA", "Ульинский прогиб", "trough", "R:RU:KHABAROVSK", "RU", "gold", basis=G,
     aliases=["Ulya trough", "Ульинский", "Среднеульинский рудный район"])
area("R:RU:SVETLOE", "Рудное поле Светлое", "ore_field", "R:RU:ULYA", "RU", "gold", basis=T,
     doc="10.18799_24131830_2022_8_3708", quote="рудного поля Светлое (Хабаровский край)",
     aliases=["Svetloe ore field", "Svetloe", "Светлое", "Уенминский рудный узел"])
area("R:RU:TRANSBAIKAL", "Забайкалье", "region", "R:RU", "RU", aliases=["Transbaikalia", "Transbaikal", "Забайкальский край"])
area("R:RU:TRANSBAIKAL:E", "Восточное Забайкалье", "region", "R:RU:TRANSBAIKAL", "RU", "gold;rare_metal",
     "mongol_okhotsk_au", basis=T, doc="10.17223_25421379_31_5", quote="оруденения Восточного Забайкалья",
     aliases=["Eastern Transbaikalia", "East Transbaikalia", "Восточно-Забайкальский"])
for i, (n, al) in enumerate([("Балейское", ["Baley", "Балей", "Балейское месторождение"]),
                              ("Карийское", ["Kara", "Карийское месторождение", "Александровское"]),
                              ("Шахтаминское", ["Shakhtama", "Шахтаминское месторождение"]),
                              ("Быстринское", ["Bystrinskoe", "Быстринское месторождение"]),
                              ("Лугиинское", ["Лугиинское месторождение", "Lugiya"]),
                              ("Серебряное", ["Серебряное месторождение"]),
                              ("Букукинское", ["Bukuka", "Букукинское месторождение"]),
                              ("Белухинское", ["Белухинское месторождение", "Belukha"])]):
    area(f"R:RU:TB:{i}", n, "deposit", "R:RU:TRANSBAIKAL:E", "RU", "gold;rare_metal", basis=T,
         doc="10.17223_25421379_31_5", quote="месторождений Восточного Забайкалья", aliases=al)
area("R:RU:MONGOLOKHOTSK", "Монголо-Охотский пояс", "belt", "", "", "gold", "mongol_okhotsk_au",
     overlaps="R:RU:TRANSBAIKAL;R:MN;R:RU:FE", aliases=["Mongol-Okhotsk belt", "Монголо-Охотский", "Mongol-Okhotsk"])
area("R:RU:YENISEI", "Енисейский кряж", "ridge", "R:RU", "RU", "gold", "yenisei_neoproterozoic_au", basis=G,
     aliases=["Yenisei Ridge", "Yenisey Ridge", "Енисейский", "Yenisei ridge"])
area("R:RU:OLIMPIADA", "Олимпиаднинское месторождение", "deposit", "R:RU:YENISEI", "RU", "gold", basis=T,
     doc="10.17308_geology.2018.1_1447", quote="Олимпиаднинское месторождение (Енисейский кряж)",
     aliases=["Olimpiada", "Олимпиада", "Олимпиаднинское", "Olimpiadninskoe"])
area("R:RU:BUREMINSKOE", "Буреминское рудопроявление", "occurrence", "R:RU:YENISEI", "RU", "gold", basis=T,
     doc="10.17308_geology.2018.1_1447", quote="рудопроявления Буреминское", aliases=["Буреминское"])
area("R:RU:ISHIMBA", "Ишимбинская зона", "zone", "R:RU:YENISEI", "RU", "gold", basis=T, doc="10.17308_geology.2018.1_1447",
     quote="зоны Ишимбинского разлома", aliases=["Ишимбинский разлом", "Ishimba fault", "Ишимбинского"])
area("R:RU:PATOM", "Байкало-Патомская складчатая область", "fold_belt", "R:RU", "RU", "gold", "patom_black_shale_au",
     basis=G, aliases=["Патомское нагорье", "Patom highland", "Patom Highlands", "Патомская складчатая область",
                       "Baikal-Patom", "Байкало-Патомская", "Ленский золоторудный район", "Lena gold district"])
area("R:RU:BODAIBO", "Мамско-Бодайбинская зона", "zone", "R:RU:PATOM", "RU", "gold", basis=T,
     doc="10.18799_24131830_2025_7_5053", quote="зоны Мамско-Бодайбинского", aliases=["Mama-Bodaibo block", "Bodaibo", "Бодайбо", "Бодайбинский"])
area("R:RU:SUKHOILOG", "Сухой Лог", "deposit", "R:RU:BODAIBO", "RU", "gold", "patom_black_shale_au", basis=T,
     doc="10.18799_24131830_2025_7_5053", quote="Золоторудное месторождение Сухой Лог (Патомское нагорье)",
     aliases=["Sukhoy Log", "Sukhoi Log", "Сухоложское", "Сухоложский"])
area("R:RU:KROPOTKIN", "Кропоткинский рудный узел", "ore_cluster", "R:RU:BODAIBO", "RU", "gold", basis=T,
     doc="10.18799_24131830_2025_7_5053", quote="Кропоткинского рудного узла", aliases=["Кропоткинский", "Kropotkin ore cluster"])
area("R:RU:KARELIA", "Карелия", "admin_region", "R:RU", "RU", overlaps="R:BALTIC", aliases=["Karelia", "Республика Карелия", "Северная Карелия", "North Karelia"])
area("R:RU:KOLA", "Кольский полуостров", "region", "R:RU", "RU", overlaps="R:BALTIC", aliases=["Kola Peninsula", "Кольский", "Мурманская область"])
area("R:BALTIC", "Балтийский щит", "shield", "", "", "gold;uranium", "svecofennian_paleoproterozoic_au_u",
     aliases=["Baltic Shield", "Fennoscandian Shield", "Фенноскандинавский щит", "Фенноскандия", "Fennoscandia"])
area("R:FI", "Финляндия", "country", overlaps="R:BALTIC", aliases=["Finland", "Финляндии"])
area("R:BALTIC:KUUSAMO", "Куусамо-Панакуолаярвинский прогиб", "trough", "R:BALTIC", "", "gold;uranium",
     "svecofennian_paleoproterozoic_au_u", basis=T, doc="10.52349_0869-7892_2024_100_147-164",
     quote="Куусамо-Панакуолаярвинского прогиба (Балтийский щит)",
     aliases=["Kuusamo", "Куусамо", "Панакуолаярвинский прогиб", "Panajärvi", "Kuusamo belt"])
area("R:BALTIC:KUOLAJARVI", "Куолаярвинский рудный узел", "ore_cluster", "R:BALTIC:KUUSAMO", "", "gold;uranium",
     basis=T, doc="10.52349_0869-7892_2024_100_147-164", quote="Куолаярвинского рудного узла",
     aliases=["Куолаярви", "Kuolajarvi", "Kuolajärvi", "Куолаярвинский"])
area("R:BALTIC:MAYSKOE_KAR", "Майское рудопроявление (Северная Карелия)", "occurrence", "R:BALTIC:KUOLAJARVI", "",
     "gold;uranium", basis=T, doc="10.52349_0869-7892_2024_100_147-164", quote="Майское (Куолаярвинский рудный узел)",
     aliases=["Майское"])
area("R:BALTIC:OZERNOE", "Озерное рудопроявление", "occurrence", "R:BALTIC:KUOLAJARVI", "", "gold;uranium", basis=T,
     doc="10.52349_0869-7892_2024_100_147-164", quote="рудопроявления Озерное", aliases=["Озерное", "Озёрное", "Ozernoe"])
area("R:BALTIC:LAGERNOE", "Лагерное рудопроявление", "occurrence", "R:BALTIC:KUOLAJARVI", "", "gold;uranium", basis=T,
     doc="10.52349_0869-7892_2024_100_147-164", quote="Лагерное", aliases=["Лагерное", "Lagernoe"])
area("R:FI:JUOMASUO", "Юомасуо", "deposit", "R:FI;R:BALTIC:KUUSAMO", "FI", "gold;uranium", basis=G,
     aliases=["Juomasuo", "месторождение Юомасуо"])
area("R:FI:ROMPAS", "Ромпас", "occurrence", "R:FI;R:BALTIC", "FI", "gold;uranium", basis=G, aliases=["Rompas"])
area("R:RU:CAUCASUS", "Северный Кавказ", "region", "R:RU", "RU", aliases=["North Caucasus", "Северного Кавказа", "Кабардино-Балкария", "Кабардино-Балкарская республика"])
area("R:RU:TYRNYAUZ", "Тырныаузский рудный узел", "ore_cluster", "R:RU:CAUCASUS", "RU", "tungsten;molybdenum;gold", basis=T,
     doc="10.24411_0869-7175-2019-10008", quote="Зыгыркольской зоны (северный фланг Тырныаузского рудного узла)",
     aliases=["Тырныауз", "Tyrnyauz", "Тырныаузское рудное поле", "Тырныаузский"])
area("R:RU:ZYGYRKOL", "Зыгыркольская зона", "zone", "R:RU:TYRNYAUZ", "RU", "tungsten;molybdenum;gold", basis=T,
     doc="10.24411_0869-7175-2019-10008", quote="Зыгыркольской зоны (северный фланг Тырныаузского рудного узла)",
     aliases=["Зыгыркольская", "Zygyrkol"])
area("R:DONBASS", "Донецкий бассейн", "basin", "", "", "base_metals;gold", aliases=["Donbass", "Донбасс", "Donetsk basin", "Донецкого бассейна"])
area("R:DONBASS:E", "Восточный Донбасс", "region", "R:DONBASS", "", "base_metals;gold", basis=T, doc="10.46698_vnc.2023.84.41.008",
     quote="Ore-magmatic systems of Eastern Donbass", aliases=["Eastern Donbass", "East Donbass"])
area("R:ALTAI", "Большой Алтай", "region", "", "", "base_metals;gold", "altai_vms", overlaps="R:KZ;R:RU;R:CN;R:MN",
     aliases=["Greater Altai", "Большого Алтая", "Рудный Алтай", "Rudny Altai", "Altai"])
area("R:RU:WSIB", "Западная Сибирь", "region", "R:RU", "RU", aliases=["West Siberia", "Западно-Сибирский", "Western Siberia"])
# ---------------------------------------------------------------- Центральная Азия
area("R:KZ", "Казахстан", "country", aliases=["Kazakhstan", "Республика Казахстан", "Қазақстан"])
area("R:KZ:E", "Восточный Казахстан", "region", "R:KZ", "KZ", aliases=["Eastern Kazakhstan", "East Kazakhstan"])
area("R:KZ:KOLDAR", "Кольдарский массив", "massif", "R:KZ", "KZ", "copper", basis=T, doc="10.20944_preprints202506.1483.v1",
     quote="Koldar Massif, Kazakhstan", aliases=["Koldar massif", "Koldar", "Кольдар"])
area("R:UZ", "Узбекистан", "country", aliases=["Uzbekistan", "Республика Узбекистан"])
area("R:UZ:ALMALYK", "Алмалыкский рудный район", "ore_district", "R:UZ", "UZ", "copper;gold", basis=T,
     doc="10.33677_ggianas20250100142", quote="Almalyk Ore Region", aliases=["Almalyk", "Алмалык", "Алмалыкский", "Almalyk ore region"])
area("R:UZ:KYZYLKUM", "Центральные Кызылкумы", "region", "R:UZ", "UZ", "gold", "tien_shan_black_shale_au", basis=G,
     aliases=["Central Kyzylkum", "Kyzylkum", "Кызылкум", "Кызылкумы"])
area("R:UZ:BUKANTAU", "Горы Букантау", "mountains", "R:UZ:KYZYLKUM", "UZ", "gold", "tien_shan_black_shale_au", basis=T,
     doc="10.52349_0869-7892_2026_106_43-52", quote="The Bukantau Mountains ring structures",
     aliases=["Bukantau", "Букантау", "Bukantau Mountains", "Central Bukantau"])
area("R:UZ:MURUNTAU", "Мурунтау", "deposit", "R:UZ:KYZYLKUM", "UZ", "gold", "tien_shan_black_shale_au", basis=G,
     aliases=["Muruntau", "месторождение Мурунтау"])
area("R:UZ:CHARMITAN", "Чармитанское рудное поле", "ore_field", "R:UZ", "UZ", "gold", basis=T,
     doc="10.47390_ydif-y2026v2i12_n06", quote="РУДОПРОЯВЛЕНИИ ТУКМАН (ЧАРМИТАНСКОЕ РУДНОЕ ПОЛЕ)",
     aliases=["Чармитан", "Charmitan", "Чармитанское"])
area("R:UZ:TUKMAN", "Тукман", "occurrence", "R:UZ:CHARMITAN", "UZ", "gold", basis=T, doc="10.47390_ydif-y2026v2i12_n06",
     quote="РУДОПРОЯВЛЕНИИ ТУКМАН (ЧАРМИТАНСКОЕ РУДНОЕ ПОЛЕ)", aliases=["Tukman", "рудопроявление Тукман"])
area("R:AZ", "Азербайджан", "country", aliases=["Azerbaijan", "Азербайджана"])
area("R:AZ:NAKHCHIVAN", "Нахчыванская Автономная Республика", "admin_region", "R:AZ", "AZ", "gold;base_metals",
     aliases=["Nakhchivan", "Нахчыван", "Нахчыванская", "Nakhchivan AR", "Nakhchivan Autonomous Republic"])
area("R:MN", "Монголия", "country", aliases=["Mongolia", "Монголии"])
area("R:MN:S", "Южная Монголия", "region", "R:MN", "MN", aliases=["South Mongolia", "Southern Mongolia"])
area("R:MN:OLONOVOOT", "Олон-Овоотский рудный узел", "ore_cluster", "R:MN:S", "MN", "gold", basis=T,
     doc="10.31857_s0205-96142019559-71", quote="Olon-Ovoot gold-ore cluster (South Mongolia)",
     aliases=["Olon-Ovoot", "Олон-Овоот", "Овоотский рудный узел"])
area("R:VN", "Вьетнам", "country", aliases=["Vietnam", "Вьетнама"])
area("R:VN:N", "Северный Вьетнам", "region", "R:VN", "VN", "gold;tin;tungsten", basis=T, doc="10.52349_0869-7892_2024_97_109-117",
     quote="metallogenic zoning of Northern Vietnam", aliases=["Northern Vietnam", "North Vietnam", "Северо-Западный Вьетнам", "Northwest Vietnam"])
for i, (n, al) in enumerate([("Чием Хоа", ["Chiem Hoa"]), ("Там Зао", ["Tam Dao"]), ("Бан Фук", ["Ban Phuc"]), ("Шонгма", ["Song Ma", "Sông Mã"])]):
    area(f"R:VN:N:{i}", f"Рудный район {n}", "ore_district", "R:VN:N", "VN", "", basis=T,
         doc="10.52349_0869-7892_2024_97_109-117", quote=f"рудный район {n}", aliases=[n, *al])
area("R:CN", "Китай", "country", aliases=["China", "КНР", "Китайская Народная Республика", "Eastern China"])
area("R:CN:JIAODONG", "Полуостров Цзяодун", "province", "R:CN", "CN", "gold", basis=G,
     aliases=["Jiaodong", "Jiaodong Peninsula", "Цзяодун", "провинция Цзяодун"])
area("R:CN:SANSHANDAO", "Саншандао", "deposit", "R:CN:JIAODONG", "CN", "gold", basis=T, doc="10.3389_feart.2026.1770354",
     quote="Sanshandao", aliases=["Sanshandao", "Sanshandao fault"])
area("R:CN:XILING", "Силин", "deposit", "R:CN:SANSHANDAO", "CN", "gold", basis=T, doc="10.3389_feart.2026.1770354", quote="Xiling deposit", aliases=["Xiling", "Xiling deposit"])
area("R:CN:HENAN", "Западный Хэнань", "region", "R:CN", "CN", aliases=["Western Henan", "Henan"])
area("R:CN:LEIMENGOU", "Леймэнгоу", "deposit", "R:CN:HENAN", "CN", "molybdenum", basis=T, doc="10.25236_ajets.2026.090201",
     quote="Leimengou Giant Mo Deposit, Western Henan, China", aliases=["Leimengou"])
area("R:CN:S", "Южный Китай", "region", "R:CN", "CN", aliases=["South China"])
area("R:CN:LUJING", "Рудное поле Луцзин", "ore_field", "R:CN:S", "CN", "uranium", basis=T, doc="10.1038_s41598-025-11307-0",
     quote="Lujing uranium ore field, South China", aliases=["Lujing", "Lujing ore field"])
# ---------------------------------------------------------------- Северо-Восточная Африка: Аравийско-Нубийский щит
area("R:ANS", "Аравийско-Нубийский щит", "shield", "", "", "gold", "ans_neoproterozoic_shear_au",
     overlaps="R:EG;R:SD;R:ET;R:SA", aliases=["Arabian-Nubian Shield", "Arabian Nubian Shield", "Nubian Shield", "ANS",
                                               "Аравийско-Нубийский"])
area("R:EG:ENS", "Египетская часть Нубийского щита", "shield_part", "R:EG;R:ANS", "EG", "gold;rare_metal", "ans_neoproterozoic_shear_au",
     overlaps="R:EG:ED", basis=T, doc="10.1038_s41598-026-60088-7", quote="gold deposits in the Egyptian Nubian Shield",
     aliases=["Egyptian Nubian Shield", "Nubian Shield of Egypt", "Egyptian part of the Arabian-Nubian Shield"])
area("R:EG", "Египет", "country", aliases=["Egypt", "Египта"])
area("R:EG:ED", "Восточная пустыня Египта", "region", "R:EG", "EG", "gold;rare_metal;fluorite", "ans_neoproterozoic_shear_au",
     overlaps="R:ANS", basis=G, aliases=["Eastern Desert", "Eastern Desert of Egypt", "Egyptian Eastern Desert", "Восточная пустыня"])
area("R:EG:ED:C", "Центральная Восточная пустыня", "region", "R:EG:ED", "EG", "gold", "ans_neoproterozoic_shear_au", basis=G,
     aliases=["Central Eastern Desert", "CED"])
area("R:EG:ED:S", "Южная Восточная пустыня", "region", "R:EG:ED", "EG", "gold;rare_metal", "ans_neoproterozoic_shear_au", basis=G,
     aliases=["Southern Eastern Desert", "South Eastern Desert", "Southeastern Desert", "SED"])
area("R:EG:ED:N", "Северная Восточная пустыня", "region", "R:EG:ED", "EG", "", basis=G, aliases=["Northern Eastern Desert", "North Eastern Desert", "NED"])
area("R:EG:ATALLASID", "Сдвиговый пояс Аталла-Сид", "shear_belt", "R:EG:ED:C", "EG", "gold", "ans_neoproterozoic_shear_au", basis=T,
     doc="10.1038_s41598-026-60088-7", quote="the Atalla-Sid Shear Belt, Central Eastern Desert",
     aliases=["Atalla-Sid Shear Belt", "Atalla-Sid", "Atalla area", "Sid area", "Atalla"])
area("R:EG:UMMEISH", "Вади Умм-Эйш Эль-Зарка", "area", "R:EG:ED:C", "EG", "gold", "ans_neoproterozoic_shear_au", basis=T,
     doc="10.33271_mining20.03.010", quote="Wadi Umm Eish El-Zarqa, Central Eastern Desert",
     aliases=["Wadi Umm Eish El-Zarqa", "Umm Eish El-Zarqa", "Wadi Umm Eish", "Wadi Kab El-Absi"])
area("R:EG:DAWI", "Сдвиговый пояс Дави", "shear_belt", "R:EG:ED:C", "EG", "gold;fluorite;barite", "ans_neoproterozoic_shear_au", basis=T,
     doc="10.1038_s41598-026-52061-1", quote="Dawi shear belt, Egypt",
     aliases=["Dawi shear belt", "Dawi", "Wadi Sodmein", "Sodmein", "Wadi Queih", "Queih shear zone", "Abu Ziran"])
area("R:EG:UMARA", "Умм-Ара — Вади Мурра", "area", "R:EG:ED:S", "EG", "rare_metal", "ans_neoproterozoic_granite_rm", basis=T,
     doc="10.1007_s43994-025-00234-5", quote="South Um Ara-Wadi Murrah area, Southeastern Desert, Egypt",
     aliases=["Um Ara-Wadi Murrah", "Um Ara", "Wadi Murrah", "Um Ara-Wadi Mura", "South Um Ara"])
area("R:EG:KHARIT", "Вади Харит", "area", "R:EG:ED:S", "EG", "gold", "ans_neoproterozoic_shear_au", basis=T,
     doc="10.1038_s41598-026-45708-6", quote="Wadi Kharit region, Southern Eastern Desert",
     aliases=["Wadi Kharit", "Kharit"])
area("R:EG:BARRAMIYA", "Барамия", "area", "R:EG:ED:C", "EG", "gold", "ans_neoproterozoic_shear_au", basis=G, aliases=["Barramiya", "Barramiya shear zone"])
area("R:EG:GHADIR", "Вади Гадир", "area", "R:EG:ED:S", "EG", "", basis=G, aliases=["Wadi Ghadir"])
area("R:SD", "Судан", "country", aliases=["Sudan", "Судана"])
area("R:SD:NE", "Северо-Восточный Судан", "region", "R:SD", "SD", aliases=["NE Sudan", "Northeastern Sudan", "North-eastern Sudan"])
area("R:SD:RSH", "Холмы Красного моря (Судан)", "region", "R:SD:NE", "SD", "gold", "ans_neoproterozoic_shear_au", overlaps="R:ANS",
     basis=G, aliases=["Red Sea Hills", "Red Sea Hills of Sudan"])
area("R:SD:TOHAMIYAM", "Тохамиям", "area", "R:SD:RSH", "SD", "gold", "ans_neoproterozoic_shear_au", basis=T,
     doc="10.31248_gjees2024.170", quote="NE Tohamiyam Area, Red Sea Hills", aliases=["Tohamiyam", "NE Tohamiyam"])
area("R:SD:BERBER", "Провинция Бербер", "admin_region", "R:SD:NE", "SD", aliases=["Berber Province", "Berber"])
area("R:SD:ARTOLI", "Артоли", "area", "R:SD:BERBER", "SD", "gold", "ans_neoproterozoic_shear_au", basis=T,
     doc="10.31248_gjees2024.162", quote="Artoli Area, Berber Province", aliases=["Artoli", "Artoli area"])
area("R:SA", "Саудовская Аравия", "country", aliases=["Saudi Arabia"])
area("R:ET", "Эфиопия", "country", aliases=["Ethiopia", "Эфиопии"])
area("R:ET:W", "Западная Эфиопия", "region", "R:ET", "ET", aliases=["Western Ethiopia", "Benishangul-Gumuz", "Benishangul-Gumuz Regional State"])
area("R:ET:MENGE", "Зеленокаменный пояс Менге", "greenstone_belt", "R:ET:W", "ET", "gold", "east_african_neoproterozoic_greenstone_au",
     basis=T, doc="10.30967_ijcrset_bikila-argeta-ayele_210", quote="Menge Greenstone Belt, Western Ethiopia",
     aliases=["Menge", "Menge greenstone belt", "Menge District", "Menge area"])
area("R:ET:LEGADEMBI", "Лега Дембе", "deposit", "R:ET", "ET", "gold", basis=G, aliases=["Lega Dembi"])
# ---------------------------------------------------------------- Западная Африка
area("R:NG", "Нигерия", "country", "", "", "", "nigerian_basement", aliases=["Nigeria", "Нигерии", "Nigerian"])
area("R:NG:NW", "Северо-Западная Нигерия", "region", "R:NG", "NG", aliases=["NW Nigeria", "Northwestern Nigeria", "North-western Nigeria", "Northern Nigeria"])
area("R:NG:NC", "Северо-Центральная Нигерия", "region", "R:NG", "NG", aliases=["North-Central Nigeria", "North Central Nigeria", "Central Nigeria"])
area("R:NG:SW", "Юго-Западная Нигерия", "region", "R:NG", "NG", aliases=["Southwestern Nigeria", "Southwest Nigeria", "SW Nigeria", "South-western Nigeria"])
area("R:NG:MARU", "Сланцевый пояс Мару", "schist_belt", "R:NG:NW", "NG", "gold", "nigerian_schist_belt_au", basis=T,
     doc="10.1007_s42461-026-01551-x", quote="Maru Schist Belt mineralization, NW Nigeria",
     aliases=["Maru Schist Belt", "Maru", "Maru schist belt", "Sabon Birni", "Yambuki"])
area("R:NG:KWARA", "Штат Квара", "state", "R:NG:NC", "NG", aliases=["Kwara State", "Kwara"])
area("R:NG:PATEGI", "Патеги", "area", "R:NG:KWARA", "NG", "", basis=T, doc="10.70882_josrar.2026.v3i5.246", quote="Pategi, North Central",
     aliases=["Pategi", "Isanlu"])
area("R:NG:KADUNA", "Штат Кадуна", "state", "R:NG:NW", "NG", aliases=["Kaduna State", "Kaduna", "Kaduna Basement Complex"])
area("R:NG:NASARAWA", "Штат Насарава", "state", "R:NG:NC", "NG", aliases=["Nasarawa", "Nasarawa State", "Akwanga"])
area("R:NG:WAMBA", "Переходная зона Вамба", "area", "R:NG:NASARAWA", "NG", "", basis=T, doc="10.21203_rs.3.rs-10216943_v1",
     quote="Wamba Transition Zone, North-Central Nigeria", aliases=["Wamba", "Wamba area", "Wamba Transition Zone"])
area("R:NG:BENUE", "Трог Бенуэ", "trough", "R:NG", "NG", aliases=["Benue Trough", "Middle Benue Trough", "the Benue Trough"])
area("R:NG:BORGU", "Боргу", "area", "R:NG:NC", "NG", "gold;rare_metal", "nigerian_schist_belt_au", basis=G,
     aliases=["Borgu", "Borgu area", "Borgu Basement Terrain", "Borgu Basement"])
area("R:NG:EKITI", "Штат Экити", "state", "R:NG:SW", "NG", aliases=["Ekiti", "Ekiti State", "Ado-Ekiti", "Effon-Alaaye", "Ilogbo", "Alawaye"])
area("R:NG:OSUN", "Штат Осун", "state", "R:NG:SW", "NG", aliases=["Osun", "Osun State", "Ilesa", "Ilesha", "Oshogbo", "Osogbo"])
area("R:NG:ILESA", "Сланцевый пояс Илеса", "schist_belt", "R:NG:OSUN", "NG", "gold", "nigerian_schist_belt_au", basis=T,
     doc="10.62292_10.62292_njp.v34i3.2025.400", quote="Ado-Ekiti-Ilesa Region of Southwest, Nigeria",
     aliases=["Ilesa schist belt", "Ilesha schist belt", "Ife-Ilesa", "Ifewara"])
area("R:NG:OKO", "Око", "area", "R:NG:SW", "NG", "iron", basis=T, doc="10.62292_njp.v35_s_.2026.639",
     quote="Iron Ore Deposit Geometry in Oko, Southwestern Nigeria", aliases=["Oko"])
area("R:GH", "Гана", "country", aliases=["Ghana"])
area("R:GH:TARKWA", "Тарква", "deposit", "R:GH", "GH", "gold", "paleoplacer_conglomerate_au", basis=G, aliases=["Tarkwa", "Tarkwaian"])
# ---------------------------------------------------------------- прочее
area("R:ZA", "ЮАР", "country", aliases=["South Africa", "Южная Африка"])
area("R:ZA:WITS", "Витватерсранд", "basin", "R:ZA", "ZA", "gold;uranium", "paleoplacer_conglomerate_au", basis=G,
     aliases=["Witwatersrand", "Witwatersrand Basin", "Витватерсранд"])
area("R:BR", "Бразилия", "country", aliases=["Brazil", "Бразилии"])
area("R:BR:JACOBINA", "Якобина", "basin", "R:BR", "BR", "gold", "paleoplacer_conglomerate_au", basis=T,
     doc="10.1007_s00126-023-01220-9", quote="metaconglomerate-hosted gold deposits at Jacobina, Brazil",
     aliases=["Jacobina", "Jacobina Basin", "Jacobina deposits"])
area("R:BR:CASTRO", "Бассейн Кастру", "basin", "R:BR", "BR", "gold", basis=T, doc="10.22564_brjg.v43i1.2335",
     quote="Epithermal Gold Deposit in the Castro Basin", aliases=["Castro Basin", "Castro"])
area("R:BR:CAMAQUA", "Минас-ду-Камакуан", "mine", "R:BR", "BR", "copper", basis=T, doc="10.24857_rgsa.v18n8-085",
     quote="Uruguay Mine, Minas do Camaquã", aliases=["Minas do Camaquã", "Camaquã", "Uruguay Mine"])
area("R:GUIANA", "Гвианский щит", "shield", "", "", "gold", overlaps="R:GY;R:BR;R:VE", aliases=["Guiana Shield", "Гвианский"])
area("R:GY", "Гайана", "country", aliases=["Guyana"])
area("R:VE", "Венесуэла", "country", aliases=["Venezuela"])
area("R:GUIANA:RORAIMA", "Супергруппа Рорайма", "basin", "R:GUIANA", "", "gold", "paleoplacer_conglomerate_au", overlaps="R:GY;R:VE;R:BR",
     basis=T, doc="10.1007_s00126-025-01395-3", quote="Gold potential of the Roraima Supergroup in Guyana",
     aliases=["Roraima Supergroup", "Roraima", "Roraima Basin", "Peaima", "Mahaica", "Eping"])
area("R:CA", "Канада", "country", aliases=["Canada", "Канады"])
area("R:CA:BC", "Британская Колумбия", "province", "R:CA", "CA", aliases=["British Columbia", "Британская Колумбия"])
area("R:CA:OKANAGAN", "Оканаган", "area", "R:CA:BC", "CA", "gold", basis=T, doc="10.12789_geocanj.2023.50.202",
     quote="Placer Gold Across the Okanagan Fault, Kelowna, British Columbia", aliases=["Okanagan", "Okanagan Fault", "Kelowna", "Kelowna area", "Okanagan Lake"])
area("R:CA:MISSION", "Ручей Мишн-Крик", "placer", "R:CA:OKANAGAN", "CA", "gold", basis=T, doc="10.12789_geocanj.2023.50.202",
     quote="Mission Creek", aliases=["Mission Creek"])
area("R:CA:LAMBLY", "Ручей Ламбли-Крик", "placer", "R:CA:OKANAGAN", "CA", "gold", basis=T, doc="10.12789_geocanj.2023.50.202",
     quote="Lambly Creek", aliases=["Lambly Creek"])
area("R:CA:ON", "Онтарио", "province", "R:CA", "CA", aliases=["Ontario"])
area("R:CA:GERALDTON", "Джералдтон", "area", "R:CA:ON", "CA", "gold", "superior_archean_greenstone_au", basis=T,
     doc="10.21203_rs.3.rs-5384377_v1", quote="Geraldton Area, Ontario", aliases=["Geraldton", "Geraldton area", "Beardmore-Geraldton", "Tashota", "Paint Lake Fault"])
area("R:CA:NU", "Нунавут", "territory", "R:CA", "CA", aliases=["Nunavut"])
area("R:CA:CAZD", "Центрально-Арктический Zn-Pb район", "district", "R:CA:NU", "CA", "base_metals", basis=T,
     doc="10.14430_arctic290", quote="Central Arctic Zn-Pb District, Nunavut", aliases=["Central Arctic Zn-Pb District", "Cornwallis Island", "Polaris"])
area("R:CA:NS", "Новая Шотландия", "province", "R:CA", "CA", aliases=["Nova Scotia"])
area("R:UK", "Великобритания", "country", aliases=["United Kingdom", "UK", "Britain"])
area("R:UK:SCOT", "Шотландия", "country_part", "R:UK", "UK", aliases=["Scotland", "Scottish"])
area("R:UK:SUPLANDS", "Южное нагорье Шотландии", "terrane", "R:UK:SCOT", "UK", "gold", basis=G, aliases=["Southern Uplands", "Southern Uplands Terrane"])
area("R:UK:DUNS", "Район Данс", "area", "R:UK:SCOT", "UK", "gold", basis=T, doc="10.6084_m9.figshare.c.8387100.v1",
     quote="Duns area, Berwickshire, Scotland", aliases=["Duns", "Duns area", "Berwickshire", "Preston Bridge", "Stoneshiel Hill", "Priestlaw"])
area("R:UK:LEADHILLS", "Лидхиллс", "district", "R:UK:SUPLANDS", "UK", "gold;base_metals", basis=G, aliases=["Leadhills"])
area("R:US", "США", "country", aliases=["USA", "United States", "U.S.", "the United States"])
area("R:US:NV", "Невада", "state", "R:US", "US", "gold", "carlin_type_au", aliases=["Nevada", "Невада"])
area("R:US:CARLIN", "Тренд Карлин", "trend", "R:US:NV", "US", "gold", "carlin_type_au", basis=G, aliases=["Carlin", "Carlin trend", "Карлин", "Карлинский"])
area("R:US:CUPRITE", "Куприт", "area", "R:US:NV", "US", "", basis=T, doc="10.29150_jhrs.v10.2.p87-94", quote="Cuprite, Nevada, USA", aliases=["Cuprite"])
area("R:US:CA", "Калифорния", "state", "R:US", "US", aliases=["California"])
area("R:US:MTPASS", "Маунтин-Пасс", "district", "R:US:CA", "US", "ree", basis=G, aliases=["Mountain Pass", "Mountain Pass District"])
area("R:US:AK", "Аляска", "state", "R:US", "US", "gold", aliases=["Alaska"])
area("R:US:AZ", "Аризона", "state", "R:US", "US", aliases=["Arizona"])
area("R:US:UT", "Юта", "state", "R:US", "US", aliases=["Utah"])
area("R:UA", "Украина", "country", aliases=["Ukraine"])
area("R:UA:SHIELD", "Украинский щит", "shield", "R:UA", "UA", basis=G, aliases=["Ukrainian Shield", "Украинский щит"])
area("R:UA:KIROVOHRAD", "Кировоградская зона разломов", "fault_zone", "R:UA:SHIELD", "UA", basis=T, doc="10.24028_gj.v47i2.322500",
     quote="Kirovohrad and Zvenyhorod-Bratsk fault zones of the Ukrainian Shield", aliases=["Kirovohrad fault zone", "Kirovohrad", "Zvenyhorod-Bratsk"])
area("R:MA", "Марокко", "country", aliases=["Morocco"])
area("R:MA:HIGHATLAS", "Восточный Высокий Атлас", "region", "R:MA", "MA", basis=T, doc="10.1038_s41598-025-99402-0",
     quote="mougueur inlier, Eastern high atlas, M[orocco]", aliases=["Eastern High Atlas", "High Atlas"])
area("R:MA:MOUGUEUR", "Выступ Мугер", "inlier", "R:MA:HIGHATLAS", "MA", "base_metals", basis=T, doc="10.1038_s41598-025-99402-0",
     quote="mougueur inlier, Eastern high atlas", aliases=["Mougueur", "Mougueur inlier"])
area("R:MA:ANTIATLAS", "Антиатлас", "region", "R:MA", "MA", aliases=["Anti-Atlas", "Anti Atlas", "central Anti-Atlas"])
area("R:MA:SIROUA", "Массив Сироуа", "massif", "R:MA:ANTIATLAS", "MA", "silver", basis=T, doc="10.12912_27197050_218151",
     quote="northern Siroua Massif, central Anti-Atlas, Morocco", aliases=["Siroua", "Siroua Massif", "Zgounder", "Imourkhssen"])
area("R:IR", "Иран", "country", aliases=["Iran"])
area("R:IR:BIDKHAN", "Бидхан", "area", "R:IR", "IR", basis=T, doc="10.4236_ojg.2017.79093", quote="Bidkhan Area", aliases=["Bidkhan"])
area("R:KE", "Кения", "country", aliases=["Kenya"])
area("R:KE:KITUI", "Китуи", "area", "R:KE", "KE", "copper", basis=T, doc="10.37284_eajenr.9.3.5322",
     quote="Kwa Mutonga–Vonza Area, Kitui West, Kenya", aliases=["Kitui", "Kitui West", "Kwa Mutonga", "Vonza", "Vonza area"])
area("R:CD", "ДР Конго", "country", aliases=["Democratic Republic of Congo", "DRC", "Congo", "Katanga"])
area("R:PH", "Филиппины", "country", aliases=["Philippines"])
area("R:PH:MANKAYAN", "Манкаян", "district", "R:PH", "PH", "copper;gold", basis=T, doc="10.5194_isprs-annals-x-5-w4-2025-269-2026",
     quote="Case of Mankaya[n]", aliases=["Mankayan"])
area("R:ID", "Индонезия", "country", aliases=["Indonesia"])
area("R:ID:BANGKA", "Банка-Белитунг", "belt", "R:ID", "ID", "tin;ree", basis=T, doc="10.29017_lpmgb.60.2.2169",
     quote="Bangka Belitung Tin-G[ranite Belt]", aliases=["Bangka Belitung", "Bangka", "Belitung"])
area("R:PK", "Пакистан", "country", aliases=["Pakistan"])
area("R:IN", "Индия", "country", aliases=["India"])
area("R:AU", "Австралия", "country", aliases=["Australia"])
area("R:NA", "Намибия", "country", aliases=["Namibia"])


def main() -> None:
    cols = ["area_id", "canonical_name", "level", "parent_ids", "country", "deposit_types", "settings",
            "overlaps", "basis", "basis_doc_id", "basis_quote", "review_status"]
    ids = [a["area_id"] for a in A]
    assert len(ids) == len(set(ids)), "повтор id"
    with (HERE / "regional_registry.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(A)
    with (HERE / "registry_aliases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alias", "area_id", "lang", "source"])
        for a in A:
            for al in a["_aliases"]:
                lang = "ru" if any("а" <= ch.lower() <= "я" for ch in al) else "en"
                w.writerow([al, a["area_id"], lang, a["basis_doc_id"] or "curation"])
    print(len(A), "территорий")


if __name__ == "__main__":
    main()
