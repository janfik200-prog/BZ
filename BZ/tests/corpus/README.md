# Тестовый корпус: выделение рудных узлов

10 статей по теме: 8 русских из КиберЛенинки и 2 английские из MDPI. Подобраны
под разные типы вёрстки — скан, тяжёлая графика, плотные таблицы, две колонки.
Нужны, чтобы мерить точность парсинга: эталоны к ним лежат в `tests/golden/`.

```
python tests/corpus/download_corpus.py
python -m georag.parse.cli --input data/pdf --device cuda
```

Данные корпуса — в `corpus.json`. Добавили статью туда — запустите
`python tests/corpus/build_corpus.py`, и эталоны с описанием обновятся; скачивание (`download_corpus.py`) берёт список прямо из corpus.json.

| Файл | Статья | Год | Стр. | Табл. | Что проверяет |
| --- | --- | --- | --- | --- | --- |
| `ru01-kaspinskiy-rudnyy-uzel.pdf` | [Разработка методики и оптимального комплекса геофизических исследовани…](https://cyberleninka.ru/article/n/razrabotka-metodiki-i-optimalnogo-kompleksa-geofizicheskih-issledovaniy-pri-vydelenii-perspektivnyh-oblastey-na-nalichie-rudnogo) | 2025 | ? | ? | самая свежая статья, прямо про выделение перспективных областей в рудном узле; много графики |
| `ru02-gur-mednokolchedannye-uzly.pdf` | [Закономерности размещения и прогноз медноколчеданных рудных узлов в ко…](https://cyberleninka.ru/article/n/zakonomernosti-razmescheniya-i-prognoz-mednokolchedannyh-rudnyh-uzlov-v-kollizionnoy-zone-glavnogo-uralskogo-razloma-gur) | 2013 | 11 | 3 | скан без нормального текстового слоя: has_text должен упасть, документ уйти на полностраничный OCR с русским языком |
| `ru03-toupugol-hanmeyshorskiy.pdf` | [Прогнозно-поисковая модель золоторудных объектов Тоупугол-Ханмейшорско…](https://cyberleninka.ru/article/n/prognozno-poiskovaya-model-zolotorudnyh-obektov-toupugol-hanmeyshorskogo-rudnogo-uzla-kak-osnova-dlya-vydeleniya-perspektivnyh) | 2021 | ? | ? | ядро темы: прогнозно-поисковая модель как основа выделения площадей. Составные названия через дефис |
| `ru04-zabaykalye-rudno-rossypnye.pdf` | [Опыт прогнозирования перспективных на золотое оруденение площадей на о…](https://cyberleninka.ru/article/n/opyt-prognozirovaniya-perspektivnyh-na-zolotoe-orudenenie-ploschadey-na-osnove-provedeniya-kompleksnogo-analiza-rudnoy-i-rossypnoy) | 2020 | 23 | 3 | самый крупный файл корпуса (10 МБ, 23 страницы) |
| `ru05-gis-verkhoyansk.pdf` | [ГИС как средство оценки рудообразующего потенциала интрузивных образов…](https://cyberleninka.ru/article/n/gis-kak-sredstvo-otsenki-rudoobrazuyuschego-potentsiala-intruzivnyh-obrazovaniy-verhoyanskogo-skladchatogo-poyasa-vostochnaya) | 2008 | 9 | 2 | единственная статья, где число таблиц и рисунков названо самими авторами |
| `ru06-yaregskiy-rudnyy-uzel.pdf` | [Новое о титаноносности Ярегского рудного узла (Южный Тиман)](https://cyberleninka.ru/article/n/novoe-o-titanonosnosti-yaregskogo-rudnogo-uzla-yuzhnyy-timan) | 2016 | 11 | ? | минералогические термины |
| `ru07-priamurye-rudnoe-zoloto.pdf` | [Перспективы Приамурья на рудное золото](https://cyberleninka.ru/article/n/perspektivy-priamurya-na-rudnoe-zoloto) | 2019 | 12 | ? | ранжирование узлов по добыче |
| `ru08-priamurye-zolotonosnye-uzly.pdf` | [Разновидности высокопродуктивных золотоносных узлов Приамурской провин…](https://cyberleninka.ru/article/n/raznovidnosti-vysokoproduktivnyh-zolotonosnyh-uzlov-priamurskoy-provintsii) | 2019 | 7 | 3 | самая короткая статья |
| `en01-tungsten-prospectivity-jiangxi.pdf` | [Prospectivity Mapping of Tungsten Mineralization in Southern Jiangxi P…](https://www.mdpi.com/2075-163X/13/5/669) | 2023 | ? | 1 | английская статья с чёткой структурой разделов |
| `en02-hadamengou-orefield-multifractal.pdf` | [Multivariate Statistical Analysis and S-A Multifractal Modeling of Lit…](https://www.mdpi.com/2076-3263/15/12/473) | 2025 | ? | 13 | главный тест TableFormer: 13 таблиц геохимии с числовыми колонками |

## Про эталоны

Разделы («Введение», «Методы») есть только в одной русской статье из восьми и в
обеих английских: русская геологическая периодика обычно идёт сплошным текстом
с УДК и ключевыми словами. Поэтому структурная проверка для неё почти всегда
пустая, а вес переносится на сущности — они взяты из авторских ключевых слов.

Число таблиц стоит там, где его можно обосновать; где нельзя — `null`, и проверка
пропускается: неверный эталон хуже отсутствующего. Числа страниц взяты из самих
PDF, кроме двух файлов со сжатыми object stream.
