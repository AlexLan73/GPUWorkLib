#!/bin/bash
# PyPanelAantenns — запуск Field Viewer (Debian venv)
#
# Порядок: 1) сервер данных (udp_server_test) → 2) Field Viewer
# Сервер формирует демо-сценарии (Recovery, Random Fill, Full Green).
# Viewer слушает UDP :5005 и отображает.
#
# Режимы: ./run_example.sh         — main_dock (все окна: Cell List, Field, Detail, Table, Controls, Log)
#         ./run_example.sh simple  — main.py  (Field + Color scale)

set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
    echo "Создаю venv и устанавливаю зависимости..."
    python3 -m venv .venv
    .venv/bin/pip install dearpygui numpy
fi

PY=".venv/bin/python"
EXAMPLES="Examples"

# 1. Запускаем сервер данных в фоне
echo ">>> [1/2] Запуск UDP-сервера данных (udp_server_test.py)..."
$PY "$EXAMPLES/udp_server_test.py" --hz 8 --cells 100 --elems 256 &   # ×2 быстрее
UDP_PID=$!
trap "kill $UDP_PID 2>/dev/null" EXIT

sleep 1

# 2. Запускаем Field Viewer (main_dock = все окна, или main = простой)
if [[ "$1" == "simple" ]]; then
    echo ">>> [2/2] Запуск Field Viewer (main.py — Field + Color scale)..."
    shift
    $PY "$EXAMPLES/main.py" --port 5005 "$@"
else
    echo ">>> [2/2] Запуск Field Viewer Docking (все окна: Cell List, Field, Detail, Table, Controls, Log)..."
    $PY "$EXAMPLES/main_dock.py" --port 5005 "$@"
fi
