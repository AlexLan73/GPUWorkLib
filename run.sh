#!/bin/bash
# GPUWorkLib - Build and Run (Ubuntu/Linux)
# Usage: ./run.sh [de|re]  or  ./run.sh build  (build only)
#   de/debug  - Debug build
#   re/release - Release build
#   build    - build only, no run

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ARG="${1:-re}"
case "$ARG" in
  de|debug) BUILD_TYPE=Debug ;;
  re|release) BUILD_TYPE=Release ;;
  build) BUILD_TYPE=Release; SKIP_RUN=1 ;;
  *) echo "Usage: $0 [de|re|build]"; exit 1 ;;
esac

echo ""
echo "GPUWorkLib Build - ${BUILD_TYPE}"
echo ""

# Simple build without presets (works everywhere)
if [[ -d "build" ]]; then
    rm -rf build
fi

cmake -B build -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DENABLE_OPENCL=ON
cmake --build build

echo ""
echo -e "${GREEN}SUCCESS${NC} - Output: ./build/GPUWorkLib"
echo ""

if [[ -z "$SKIP_RUN" ]]; then
    echo "Running..."
    ./build/GPUWorkLib
fi
