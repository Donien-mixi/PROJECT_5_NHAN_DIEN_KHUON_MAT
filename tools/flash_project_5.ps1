# =============================================================================
# Build & Nap Firmware Project 5 bang ESP-IDF 5.3 (Thu muc goc o dia D)
# Cach dung:
#   powershell -ExecutionPolicy Bypass -File tools\flash_project_5.ps1 -Port COM3
#   -BuildOnly : Chi build, khong nap
#   -NoBuild   : Nap ngay firmware da build (khong build lai)
# =============================================================================
param(
    [string]$Port = "COM3",
    [switch]$NoBuild,
    [switch]$BuildOnly
)

$IdfPath = "C:\Espressif\frameworks\esp-idf-v5.3.5"
$IdfPythonEnv = "C:\Espressif\python_env\idf5.3_py3.11_env"
$FwDir = Join-Path $PSScriptRoot "..\firmware_esp32" | Resolve-Path
$PythonExe = Join-Path $IdfPythonEnv "Scripts\python.exe"

$env:PYTHONIOENCODING = "utf-8"
$env:PATH = (Join-Path $IdfPythonEnv "Scripts") + ";" + $env:PATH
$env:IDF_PYTHON_ENV_PATH = $IdfPythonEnv
. (Join-Path $IdfPath "export.ps1") *> $null

Push-Location $FwDir
try {
    if (-not $NoBuild -and -not (Test-Path "build\face_attendance_esp32s3.bin")) {
        Write-Host "==> [Project 5] Dang bien dich Firmware tren ESP-IDF 5.3 (SIMD esp-nn)..." -ForegroundColor Cyan
        if (-not (Test-Path "sdkconfig")) {
            idf.py set-target esp32s3
        }
        idf.py build
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build that bai! Vui long kiem tra log." -ForegroundColor Red
            exit 1
        }
        Write-Host "Build thanh cong!" -ForegroundColor Green
    }

    if ($BuildOnly) {
        Write-Host "==> Da hoan tat che do BuildOnly." -ForegroundColor Green
        exit 0
    }

    Write-Host "==> Dang nap Firmware len cong $Port qua esptool..." -ForegroundColor Cyan
    & $PythonExe -m esptool --chip esp32s3 -p $Port -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0x10000 build\face_attendance_esp32s3.bin
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Nap that bai! Luu y: Neu Arduino IDE hoac VS Code dang mo Serial Monitor o cong $Port, vui long dong lai roi chay lai lenh." -ForegroundColor Red
        exit 1
    }

    Write-Host "Nap thanh cong! Dang mo Serial Monitor ($Port). Nhan Ctrl + ] de thoat..." -ForegroundColor Green
    idf.py -p $Port monitor
} finally {
    Pop-Location
}
