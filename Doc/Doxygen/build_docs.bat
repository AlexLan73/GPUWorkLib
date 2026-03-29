@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  GPUWorkLib - Doxygen Build Script
REM  Order: clean -> DrvGPU (base) -> modules -> main (TAGFILES)
REM ============================================================

set DOXYGEN=C:\Program Files\doxygen\bin\doxygen.exe

REM --- Add Graphviz to PATH if installed ---
if exist "C:\Program Files\Graphviz\bin\dot.exe" (
    set "PATH=%PATH%;C:\Program Files\Graphviz\bin"
)

if not exist "%DOXYGEN%" (
    echo [ERROR] Doxygen not found at: %DOXYGEN%
    echo         Install from https://www.doxygen.nl/download.html
    exit /b 1
)

echo ============================================================
echo  GPUWorkLib Doxygen Build
echo ============================================================

REM ============================================================
REM  Step 0: Clean generated files (html/, .tag)
REM ============================================================
echo.
echo === Step 0/4: Clean ===

if exist html (
    rd /s /q html
    echo     Removed html\
)

if exist DrvGPU\html (
    rd /s /q DrvGPU\html
    echo     Removed DrvGPU\html\
)
if exist DrvGPU\drvgpu.tag (
    del /q DrvGPU\drvgpu.tag
    echo     Removed DrvGPU\drvgpu.tag
)

for %%m in (signal_generators fft_func filters statistics heterodyne vector_algebra capon range_angle fm_correlator strategies lch_farrow) do (
    if exist modules\%%m\html (
        rd /s /q modules\%%m\html
    )
    if exist modules\%%m\%%m.tag (
        del /q modules\%%m\%%m.tag
    )
)
echo     Removed modules\*\html\ and *.tag

echo     Clean OK

REM ============================================================
REM  Step 1: DrvGPU (base, no dependencies)
REM ============================================================
echo.
echo === Step 1/4: DrvGPU ===
pushd DrvGPU
"%DOXYGEN%" Doxyfile
if !ERRORLEVEL! neq 0 (
    echo [ERROR] DrvGPU build failed!
    popd
    exit /b 1
)
popd
echo     DrvGPU OK

REM ============================================================
REM  Step 2: Modules (depend on DrvGPU .tag)
REM ============================================================
echo.
echo === Step 2/4: Modules ===

call :build_module signal_generators
call :build_module fft_func
call :build_module filters
call :build_module statistics
call :build_module heterodyne
call :build_module vector_algebra
call :build_module capon
call :build_module range_angle
call :build_module fm_correlator
call :build_module strategies
call :build_module lch_farrow

REM ============================================================
REM  Step 3: Main (connects ALL .tag files)
REM ============================================================
echo.
echo === Step 3/4: Main Doxyfile (TAGFILES) ===
"%DOXYGEN%" Doxyfile
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Main Doxyfile build failed!
    exit /b 1
)
echo     Main OK

REM ============================================================
REM  Step 4: Done
REM ============================================================
echo.
echo ============================================================
echo  Done! Opening documentation...
echo ============================================================
if exist html\index.html (
    start html\index.html
) else (
    echo [WARN] html\index.html not found
)
exit /b 0

REM ============================================================
REM  Subroutine: build one module
REM ============================================================
:build_module
echo --- %1 ---
pushd modules\%1
"%DOXYGEN%" Doxyfile
if !ERRORLEVEL! neq 0 (
    echo [WARN] %1 build had warnings
)
popd
echo     %1 OK
exit /b 0
