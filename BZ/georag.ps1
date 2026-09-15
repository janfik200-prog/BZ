<#
    ГеоRAG — один вход для всех действий.

    Вместо длинных команд с десятком ключей:

        .\georag.ps1 start                      поднять базу
        .\georag.ps1 add "выделение рудных узлов"   добыть статьи по теме
        .\georag.ps1 all                        добыть по всем темам из topics.yaml
        .\georag.ps1 clean                      показать статьи не по теме
        .\georag.ps1 clean -Apply               убрать их в _отсев
        .\georag.ps1 ingest                     загрузить добытое в базу
        .\georag.ps1 ingest -Ollama             то же, но векторы считать через Ollama
        .\georag.ps1 search "рудные узлы"       найти в базе
        .\georag.ps1 graph                      построить связи между статьями
        .\georag.ps1 graph -Llm                 то же, но с разбором моделью (дольше, точнее)
        .\georag.ps1 web                        открыть поиск в браузере
        .\georag.ps1 stats                      что в базе
        .\georag.ps1 stop                       остановить базу
        .\georag.ps1 test                       проверить, что код цел
        .\georag.ps1 help                       эта справка

    Если PowerShell откажется запускать скрипт («выполнение сценариев отключено»),
    один раз разрешите их для своей учётной записи:

        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1)]
    [string]$Argument = "",

    # Сколько статей разбирать за тему
    [int]$MaxDocs = 5,
    # Сколько результатов показывать в поиске
    [int]$Limit = 10,
    # Не спрашивать модель, решать эвристикой
    [switch]$NoLlm,
    # Заново обработать даже то, что уже было
    [switch]$Force,
    # Для clean: действительно перенести, а не только показать
    [switch]$Apply,
    # Считать векторы через Ollama, а не на видеокарте
    [switch]$Ollama,
    # Для graph: дополнить разбор языковой моделью
    [switch]$Llm
)

# --- настройки проекта ------------------------------------------------------ #
$Mailto = "tracybowmanpowxzfwl@outlook.com"   # для polite pool OpenAlex и Crossref
$Model  = "qwen3:14b"                          # модель в Ollama для отбора статей

# --------------------------------------------------------------------------- #
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Не найден $Python" -ForegroundColor Red
    Write-Host "Похоже, виртуальное окружение не создано. Создать:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv"
    Write-Host "    .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

function Show-Help {
    Get-Help $PSCommandPath -Detailed | Out-String | Write-Host
}

function Test-Database {
    $state = docker compose ps --format "{{.State}}" 2>$null
    if (-not $state) {
        Write-Host "База не запущена. Поднимаю..." -ForegroundColor Yellow
        docker compose up -d | Out-Null
        Start-Sleep -Seconds 5
    }
}

function Invoke-Acquire {
    param([string[]]$Topic)

    $cmdArgs = @("-m", "georag.acquire.cli") + $Topic + @(
        "--max-docs", $MaxDocs,
        "--no-ocr",
        "--device", "cuda",
        "--mailto", $Mailto,
        "--model", $Model
    )
    if ($NoLlm) { $cmdArgs += "--no-llm" }
    if ($Force) { $cmdArgs += "--force" }

    & $Python @cmdArgs
}

switch ($Command.ToLower()) {

    "start" {
        docker compose up -d
        Write-Host "`nЖду, пока база будет готова..." -ForegroundColor Gray
        Start-Sleep -Seconds 5
        docker compose ps
    }

    "stop" {
        docker compose stop
        Write-Host "База остановлена. Данные на месте." -ForegroundColor Green
    }

    "status" {
        docker compose ps
    }

    "add" {
        if (-not $Argument) {
            Write-Host 'Нужна тема: .\georag.ps1 add "выделение рудных узлов"' -ForegroundColor Red
            exit 1
        }
        Test-Database
        Invoke-Acquire -Topic @($Argument)
    }

    "all" {
        Test-Database
        Invoke-Acquire -Topic @("--topics", "topics.yaml")
    }

    "clean" {
        $cmdArgs = @("scripts\clean_acquired.py")
        if ($Apply) { $cmdArgs += "--apply" }
        & $Python @cmdArgs
    }

    "ingest" {
        Test-Database
        $cmdArgs = @("-m", "georag.index.cli", "ingest")
        if ($Force) { $cmdArgs += "--force" }
        if ($Ollama) { $cmdArgs += @("--embedder", "ollama") }
        & $Python @cmdArgs
    }

    "search" {
        if (-not $Argument) {
            Write-Host 'Нужен запрос: .\georag.ps1 search "рудные узлы"' -ForegroundColor Red
            exit 1
        }
        Test-Database
        $cmdArgs = @("-m", "georag.index.cli", "search", $Argument, "--limit", $Limit)
        if ($Ollama) { $cmdArgs += @("--embedder", "ollama") }
        & $Python @cmdArgs
    }

    "graph" {
        Test-Database
        $cmdArgs = @("-m", "georag.graph.cli", "build")
        if ($Llm) { $cmdArgs += @("--llm", "--model", $Model) }
        & $Python @cmdArgs
    }

    "web" {
        Test-Database
        $cmdArgs = @("-m", "georag.web.server")
        if ($Ollama) { $cmdArgs += @("--embedder", "ollama") }
        & $Python @cmdArgs
    }

    "stats" {
        Test-Database
        & $Python -m georag.index.cli stats
    }

    "init" {
        Test-Database
        & $Python -m georag.index.cli init
    }

    "test" {
        & $Python tests\smoke_test.py
        & $Python tests\acquire_smoke_test.py
        & $Python tests\index_smoke_test.py
        & $Python tests\graph_smoke_test.py
    }

    default { Show-Help }
}
