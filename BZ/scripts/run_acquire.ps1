# Один автоматический прогон по всем темам из topics.yaml.
# Это то, что запускает планировщик; вручную запускать не нужно.
#
# Проверить руками:  powershell -ExecutionPolicy Bypass -File scripts\run_acquire.ps1

$ErrorActionPreference = 'Stop'

# Корень проекта — на уровень выше папки scripts, чтобы скрипт работал из любого места.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "run-$stamp.txt"

"=== запуск $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $log

# Окружение проекта. Если его нет — понятная ошибка вместо падения на python.
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    "Не найдено окружение .venv. Создайте его: python -m venv .venv" | Tee-Object -FilePath $log -Append
    exit 1
}

# Ollama нужна для формулировки запросов и отбора по аннотациям. Если её нет,
# прогон не останавливается: отбор делает встроенная эвристика.
$ollamaUp = $false
try {
    Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 -UseBasicParsing | Out-Null
    $ollamaUp = $true
} catch {
    "Ollama не отвечает — отбор пойдёт эвристикой" | Tee-Object -FilePath $log -Append
}

$args = @('-m', 'georag.acquire.cli', '--topics', 'topics.yaml', '--device', 'cuda', '--max-docs', '10')
if (-not $ollamaUp) { $args += '--no-llm' }

& $python @args 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE

"=== конец, код возврата $code ===" | Tee-Object -FilePath $log -Append

# Чистим логи старше месяца, чтобы папка не росла бесконечно.
Get-ChildItem $logDir -Filter 'run-*.txt' |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $code
