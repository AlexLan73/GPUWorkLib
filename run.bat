@echo off
cls
if "%1"=="" (
    echo.
    echo ERROR: Missing parameter!
    echo Usage: run.bat de  or  run.bat re
    echo.
    exit /b 1
)

if /i "%1"=="de" (
    echo.
    echo ========================
    echo DEBUG MODE
    echo ========================
    echo.
    if exist build rmdir /s /q build
    
    echo Running CMake with Debug configuration...
    cmake --preset windows-debug
    if %errorlevel% neq 0 (
        echo CMAKE ERROR!
        exit /b 1
    )
    
    echo Building project in Debug mode...
    cmake --build build --config Debug
    if %errorlevel% neq 0 (
        echo BUILD ERROR!
        exit /b 1
    )
    
    echo.
    echo SUCCESS - Debug build complete
    echo Output: ./build/Debug/GPUWorkLib.exe
    echo.
    exit /b 0
)

if /i "%1"=="re" (
    echo.
    echo ========================
    echo RELEASE MODE
    echo ========================
    echo.
    if exist build rmdir /s /q build
    
    echo Running CMake with Release configuration...
    cmake --preset windows-release
    if %errorlevel% neq 0 (
        echo CMAKE ERROR!
        exit /b 1
    )
    
    echo Building project in Release mode...
    cmake --build build --config Release
    if %errorlevel% neq 0 (
        echo BUILD ERROR!
        exit /b 1
    )
    
    echo.
    echo SUCCESS - Release build complete
    echo Output: ./build/Release/GPUWorkLib.exe
    echo.
    exit /b 0
)

echo ERROR: Use de or re
exit /b 1