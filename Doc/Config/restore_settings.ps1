# Восстановление настроек на новой машине
# Запуск: powershell -ExecutionPolicy Bypass -File restore_settings.ps1

$Green = @{ ForegroundColor = 'Green' }
$Red = @{ ForegroundColor = 'Red' }
$Yellow = @{ ForegroundColor = 'Yellow' }

Write-Host "`n🔄 Восстановление DrvGPU настроек..." @Yellow

# Найти архив
$BackupFile = Get-ChildItem "DrvGPU-Settings-*.zip" -ErrorAction SilentlyContinue | 
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1

if (-not $BackupFile) {
    Write-Host "❌ Backup файл DrvGPU-Settings-*.zip не найден!" @Red
    Write-Host "Поместите архив в текущую папку проекта и повторите." @Yellow
    exit 1
}

Write-Host "`n📦 Найден архив: $($BackupFile.Name)" @Green
Write-Host "⏳ Распаковываю..." @Yellow

try {
    Expand-Archive -Path $BackupFile.FullName -DestinationPath "." -Force
    Write-Host "`n✅ Успешно восстановлено!" @Green
    Write-Host "   Файлы:" @Green
    Write-Host "   ✓ .vscode/ (настройки VSCode)" @Green
    Write-Host "   ✓ cmake/ (модули CMake)" @Green
    Write-Host "   ✓ CMakeLists.txt (все уровни)" @Green
    Write-Host "`n🚀 Готово! Открой проект в VSCode:`n   code .`n" @Green
} catch {
    Write-Host "❌ Ошибка при распаковке: $_" @Red
    exit 1
}
