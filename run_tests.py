#!/usr/bin/env python3
"""
run_tests.py — запуск Python-тестов GPUWorkLib
===============================================

Работает на Windows и Linux без изменений.
Запускает каждый test_*.py напрямую: python file.py

Примеры:
    python run_tests.py                            # все тесты, авто-поиск build/
    python run_tests.py -m filters                 # только модуль filters
    python run_tests.py -b build/linux/python      # указать путь к .so/.pyd
    python run_tests.py -v -m heterodyne           # verbose режим

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


def find_test_files(test_dir: Path) -> list:
    """Найти все test_*.py файлы в директории (не рекурсивно)."""
    if not test_dir.exists():
        return []
    return sorted(test_dir.glob("test_*.py"))


def run_test_file(test_file: Path, env: dict, verbose: bool) -> int:
    """Запустить один test_*.py файл. Возвращает return code."""
    if verbose:
        print(f"    -> python {test_file.name}")
    r = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return r.returncode


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
        help="Подробный вывод (показывать каждый запускаемый файл)",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Показать список модулей и выйти",
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

    # Собрать список модулей для запуска
    if args.module:
        modules_to_run = [args.module]
    else:
        modules_to_run = MODULES

    # Запустить тесты
    total_files = 0
    failed_files = []

    for mod in modules_to_run:
        test_dir = PYTHON_TEST_DIR / mod
        test_files = find_test_files(test_dir)
        if not test_files:
            continue

        print(f"\n[{mod}] {len(test_files)} test file(s):")
        for test_file in test_files:
            total_files += 1
            rc = run_test_file(test_file, env, args.verbose)
            if rc != 0:
                failed_files.append(str(test_file.relative_to(PROJECT_ROOT)))
                print(f"  [FAIL] {test_file.name}")
            else:
                if args.verbose:
                    print(f"  [PASS] {test_file.name}")

    print(f"\n{'='*40}")
    print(f"Total files: {total_files}")
    if failed_files:
        print(f"Failed ({len(failed_files)}):")
        for f in failed_files:
            print(f"  {f}")
        return 1
    else:
        print("All passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
