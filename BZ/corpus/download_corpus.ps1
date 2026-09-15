# Скачивание тестового корпуса. Запуск из корня проекта:
#   powershell -ExecutionPolicy Bypass -File corpus\download_corpus.ps1
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path 'data\pdf' | Out-Null
$ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

Write-Host '→ ru01-kaspinskiy-rudnyy-uzel.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/razrabotka-metodiki-i-optimalnogo-kompleksa-geofizicheskih-issledovaniy-pri-vydelenii-perspektivnyh-oblastey-na-nalichie-rudnogo/pdf' -OutFile 'data\pdf\ru01-kaspinskiy-rudnyy-uzel.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru02-gur-mednokolchedannye-uzly.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/zakonomernosti-razmescheniya-i-prognoz-mednokolchedannyh-rudnyh-uzlov-v-kollizionnoy-zone-glavnogo-uralskogo-razloma-gur/pdf' -OutFile 'data\pdf\ru02-gur-mednokolchedannye-uzly.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru03-toupugol-hanmeyshorskiy.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/prognozno-poiskovaya-model-zolotorudnyh-obektov-toupugol-hanmeyshorskogo-rudnogo-uzla-kak-osnova-dlya-vydeleniya-perspektivnyh/pdf' -OutFile 'data\pdf\ru03-toupugol-hanmeyshorskiy.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru04-zabaykalye-rudno-rossypnye.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/opyt-prognozirovaniya-perspektivnyh-na-zolotoe-orudenenie-ploschadey-na-osnove-provedeniya-kompleksnogo-analiza-rudnoy-i-rossypnoy/pdf' -OutFile 'data\pdf\ru04-zabaykalye-rudno-rossypnye.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru05-gis-verkhoyansk.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/gis-kak-sredstvo-otsenki-rudoobrazuyuschego-potentsiala-intruzivnyh-obrazovaniy-verhoyanskogo-skladchatogo-poyasa-vostochnaya/pdf' -OutFile 'data\pdf\ru05-gis-verkhoyansk.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru06-yaregskiy-rudnyy-uzel.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/novoe-o-titanonosnosti-yaregskogo-rudnogo-uzla-yuzhnyy-timan/pdf' -OutFile 'data\pdf\ru06-yaregskiy-rudnyy-uzel.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru07-priamurye-rudnoe-zoloto.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/perspektivy-priamurya-na-rudnoe-zoloto/pdf' -OutFile 'data\pdf\ru07-priamurye-rudnoe-zoloto.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ ru08-priamurye-zolotonosnye-uzly.pdf'
Invoke-WebRequest -Uri 'https://cyberleninka.ru/article/n/raznovidnosti-vysokoproduktivnyh-zolotonosnyh-uzlov-priamurskoy-provintsii/pdf' -OutFile 'data\pdf\ru08-priamurye-zolotonosnye-uzly.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ en01-tungsten-prospectivity-jiangxi.pdf'
Invoke-WebRequest -Uri 'https://www.mdpi.com/2075-163X/13/5/669/pdf' -OutFile 'data\pdf\en01-tungsten-prospectivity-jiangxi.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host '→ en02-hadamengou-orefield-multifractal.pdf'
Invoke-WebRequest -Uri 'https://www.mdpi.com/2076-3263/15/12/473/pdf' -OutFile 'data\pdf\en02-hadamengou-orefield-multifractal.pdf' -UserAgent $ua -MaximumRedirection 5
Start-Sleep -Seconds 2
Write-Host 'Готово. Файлы в data\pdf'
