#!/usr/bin/env python3
"""
run_tests.py — запуск Python-тестов GPUWorkLib
===============================================

Работает на Windows и Linux без изменений.

Примеры:
    python run_tests.py                            # все тесты, авто-поиск build/
    python run_tests.py -m filters                 # только модуль filters
    python run_tests.py -b build/linux/python      # указать путь к .so/.pyd
    python run_tests.py -v -m heterodyne           # verbose, один модуль
    python run_tests.py -- -k "test_fir"           # передать аргументы pytest

Linux ROCm пример:
    python run_tests.py -b build/debian-radeon9070/python

Windows MSVC пример:
    python run_tests.py -b build/python/Release
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Валидные модули тестов
MODULES = [
    "fft_maxima",
    "filters",
    "signal_generators",
    "heterodyne",
    "statistics",
    "lch_farrow",
    "integration",
    "vector_algebra",
    "fm_correlator",
    "strategies",
]

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON_TEST_DIR = PROJECT_ROOT / "Python_test"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Запуск Python-тестов GPUWorkLib (Windows / Linux)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--build-dir", "-b",
        metavar="PATH",
        help="Путь к директории с gpuworklib.so/.pyd. "
             "Может быть абсолютным или относительным от корня проекта.",
    )
    parser.add_argument(
        "--module", "-m",
        choices=MODULES,
        metavar="MODULE",
        help=f"Запустить тесты только одного модуля. Доступны: {', '.join(MODULES)}",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод pytest (-v)",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Показать список модулей и выйти",
    )
    # Всё после -- идёт напрямую в pytest
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Дополнительные аргументы pytest (после --)",
    )
    args = parser.parse_args()

    if args.list_modules:
        print("Доступные модули тестов:")
        for m in MODULES:
            test_dir = PYTHON_TEST_DIR / m
            status = "✓" if test_dir.exists() else "✗"
            print(f"  {status} {m}")
        return 0

    # Настроить окружение
    env = os.environ.copy()

    if args.build_dir:
        build_path = Path(args.build_dir)
        if not build_path.is_absolute():
            build_path = PROJECT_ROOT / build_path
        build_path = build_path.resolve()
        env["GPUWORKLIB_BUILD_DIR"] = str(build_path)
        print(f"[run_tests] GPUWORKLIB_BUILD_DIR = {build_path}")

    # Сформировать команду pytest
    test_target = str(PYTHON_TEST_DIR / args.module) if args.module else str(PYTHON_TEST_DIR)

    cmd = [sys.executable, "-m", "pytest", test_target]

    if args.verbose:
        cmd.append("-v")

    # Убрать ведущий "--" если pytest_args начинается с него
    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd.extend(extra)

    print(f"[run_tests] {' '.join(cmd)}")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
