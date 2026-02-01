# Создание portable backup всех настроек DrvGPU
# Запуск: powershell -ExecutionPolicy Bypass -File backup_settings.ps1

$Green = @{ ForegroundColor = 'Green' }
$Red = @{ ForegroundColor = 'Red' }
$Yellow = @{ ForegroundColor = 'Yellow' }

Write-Host "`n🔄 Создание portable backup DrvGPU..." @Yellow

$ProjectRoot = Get-Location
$BackupName = "DrvGPU-Settings-$(Get-Date -Format 'yyyy-MM-dd-HHmm').zip"
$BackupPath = Join-Path $ProjectRoot $BackupName

# Что архивируем
$FoldersToBackup = @('.vscode', 'cmake')
$FilesToBackup = @(
    'CMakeLists.txt',
    'include/DrvGPU/CMakeLists.txt',
    'include/DrvGPU/backends/opencl/CMakeLists.txt',
    'include/DrvGPU/backends/rocm/CMakeLists.txt',
    'include/DrvGPU/memory/CMakeLists.txt',
    'include/DrvGPU/common/CMakeLists.txt',
    'tests/CMakeLists.txt'
)

Write-Host "`n✓ Проверка файлов:" @Green
$ValidItems = @()

foreach ($Folder in $FoldersToBackup) {
    $FolderPath = Join-Path $ProjectRoot $Folder
    if (Test-Path $FolderPath) {
        Write-Host "  ✓ $Folder" @Green
        $ValidItems += $FolderPath
    } else {
        Write-Host "  ✗ $Folder (не найдена)" @Red
    }
}

foreach ($File in $FilesToBackup) {
    $FilePath = Join-Path $ProjectRoot $File
    if (Test-Path $FilePath) {
        Write-Host "  ✓ $File" @Green
        $ValidItems += $FilePath
    } else {
        Write-Host "  ✗ $File (не найден)" @Red
    }
}

if ($ValidItems.Count -eq 0) {
    Write-Host "`n❌ Нечего архивировать!" @Red
    exit 1
}

Write-Host "`n📦 Создание архива: $BackupName" @Yellow

try {
    Compress-Archive -Path $ValidItems -DestinationPath $BackupPath -Force
    $Size = ([System.IO.FileInfo]$BackupPath).Length / 1MB
    Write-Host "✓ Архив создан успешно!" @Green
    Write-Host "  Размер: $([math]::Round($Size, 2)) MB"
    Write-Host "  Файл: $BackupName"
    Write-Host "`n✅ Готово! Скопируй архив на другую машину и распакуй:" @Green
    Write-Host "  Expand-Archive -Path `"$BackupName`" -DestinationPath `".`" -Force`n" @Yellow
} catch {
    Write-Host "❌ Ошибка: $_" @Red
    exit 1
}
