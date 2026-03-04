#!/bin/bash
# GPUWorkLib: проверка реализации профилирования по GPU_Profiling_Mechanism.md
# Usage: ./check_profiling.sh [module_name]
#        ./check_profiling.sh --all

set -e
cd "$(dirname "$0")/.."
python scripts/check_profiling.py "$@"
