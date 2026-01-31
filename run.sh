#!/bin/bash

################################################################################
# Build Script - Debug/Release Mode with CMakePresets
################################################################################

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Проверить что параметр передан
################################################################################

if [ -z "$1" ]; then
    echo ""
    echo "ERROR: Missing parameter!"
    echo "Usage: ./start.sh de  or  ./start.sh re"
    echo ""
    exit 1
fi

################################################################################
# DEBUG РЕЖИМ
################################################################################

if [ "$1" = "de" ] || [ "$1" = "debug" ]; then
    echo ""
    echo "========================"
    echo "DEBUG MODE"
    echo "========================"
    echo ""
    
    # Удалить старую директорию build
    echo "Removing old build directory..."
    if [ -d "build" ]; then
        rm -rf build
    fi
    
    # Запустить CMake с Debug конфигурацией используя preset
    echo "Running CMake with Debug configuration..."
    cmake --preset linux-main -DCMAKE_BUILD_TYPE=Debug
    if [ $? -ne 0 ]; then
        echo "CMAKE ERROR!"
        exit 1
    fi
    
    # Собрать проект в Debug режиме
    echo "Building project in Debug mode..."
    ninja -C build
    if [ $? -ne 0 ]; then
        echo "BUILD ERROR!"
        exit 1
    fi
    
    echo ""
    echo "SUCCESS - Debug build complete"
    echo "Output: ./build/LCH-Farrow"
    echo ""
    exit 0
fi

################################################################################
# RELEASE РЕЖИМ
################################################################################

if [ "$1" = "re" ] || [ "$1" = "release" ]; then
    echo ""
    echo "========================"
    echo "RELEASE MODE"
    echo "========================"
    echo ""
    
    # Удалить старую директорию build
    echo "Removing old build directory..."
    if [ -d "build" ]; then
        rm -rf build
    fi
    
    # Запустить CMake с Release конфигурацией используя preset
    echo "Running CMake with Release configuration..."
    cmake --preset linux-main -DCMAKE_BUILD_TYPE=Release
    if [ $? -ne 0 ]; then
        echo "CMAKE ERROR!"
        exit 1
    fi
    
    # Собрать проект в Release режиме
    echo "Building project in Release mode..."
    ninja -C build
    if [ $? -ne 0 ]; then
        echo "BUILD ERROR!"
        exit 1
    fi
    
    echo ""
    echo "SUCCESS - Release build complete"
    echo "Output: ./build/LCH-Farrow"
    echo ""
    exit 0
fi

################################################################################
# Если параметр неправильный
################################################################################

echo "ERROR: Use de or re"
exit 1