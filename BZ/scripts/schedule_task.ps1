# Постановка ночного прогона в планировщик Windows. Запускается один раз.
#
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_task.ps1
#
# После этого база пополняется сама каждую ночь: планировщик запускает
# run_acquire.ps1, тот проходит по всем темам из topics.yaml.
# Это windows-аналог cron из части 11 плана.
#
# Посмотреть задачу:   Get-ScheduledTask -TaskName 'GeoRAG-acquire'
# Запустить сейчас:    Start-ScheduledTask -TaskName 'GeoRAG-acquire'
# Когда работала:      Get-ScheduledTaskInfo -TaskName 'GeoRAG-acquire'
# Убрать:              Unregister-ScheduledTask -TaskName 'GeoRAG-acquire' -Confirm:$false

param(
    [string]$At = '03:00',                  # время ночного запуска
    [string]$TaskName = 'GeoRAG-acquire'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script = Join-Path $root 'scripts\run_acquire.ps1'
if (-not (Test-Path $script)) { throw "Не найден $script" }

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -Daily -At $At

# Не будить машину ради задачи, но догнать пропущенный запуск, если она спала.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description 'ГеоRAG: ночное пополнение базы знаний по темам из topics.yaml' `
    -Force | Out-Null

Write-Host "Задача «$TaskName» поставлена на $At каждый день."
Write-Host "Проверить сразу, не дожидаясь ночи:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Логи прогонов появятся в $((Join-Path $root 'logs'))"
