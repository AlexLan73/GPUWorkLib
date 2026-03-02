#!/bin/bash
# GPUWorkLib Agent Test Runner — C++ and Python
# Usage: ./run_agent_tests.sh [all | <module> | --file <path>]

set -e
cd "$(dirname "$0")/.."

MODE="${1:-all}"
BUILD_DIR="${BUILD_DIR:-build}"
EXE="$BUILD_DIR/GPUWorkLib"

echo "=============================================="
echo "  GPUWorkLib Agent Tests"
echo "  Mode: $MODE"
echo "=============================================="

# Build C++
echo ""
echo "[1/3] Building..."
cmake --build "$BUILD_DIR" --target GPUWorkLib -j$(nproc 2>/dev/null || echo 4)

# C++ tests
echo ""
echo "[2/3] C++ tests"
echo "----------------------------------------------"
if [[ "$MODE" == "--file" ]]; then
    FILE="$2"
    [[ -z "$FILE" ]] && { echo "Usage: $0 --file <path>"; exit 1; }
    "$EXE" --file "$FILE"
else
    "$EXE" "$MODE"
fi

# Python tests
echo ""
echo "[3/3] Python tests"
echo "----------------------------------------------"
SCRIPT_DIR="$(dirname "$0")"
if [[ "$MODE" == "--file" ]]; then
    python3 "$SCRIPT_DIR/run_agent_tests.py" --file "$FILE"
else
    python3 "$SCRIPT_DIR/run_agent_tests.py" "$MODE"
fi

echo ""
echo "=============================================="
echo "  All tests passed."
echo "=============================================="
