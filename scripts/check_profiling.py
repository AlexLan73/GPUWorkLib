#!/usr/bin/env python3
"""
GPUWorkLib: проверка реализации профилирования модулей
Сверка с Doc_Addition/GPU_Profiling_Mechanism.md

Usage:
  python scripts/check_profiling.py [module_name]
  python scripts/check_profiling.py --all

Без аргументов — проверяет все модули у которых есть код профилирования в tests/.
"""

from pathlib import Path
from datetime import datetime
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = REPO_ROOT / "modules"
DOC_REF = "Doc_Addition/GPU_Profiling_Mechanism.md"

# Ключевые слова: модуль использует профилирование если хоть одно встречается в tests/
PROFILING_KEYWORDS = ["GPUProfiler", "GpuBenchmarkBase", "hipEvent", "profiler.Record",
                      "profiler.Start", "RecordEvent", "RecordROCmEvent"]


def find_profiling_modules():
    """Все модули у которых есть профилирование в tests/."""
    modules = set()
    for mod_dir in MODULES_DIR.iterdir():
        if not mod_dir.is_dir():
            continue
        tests_dir = mod_dir / "tests"
        if not tests_dir.exists():
            continue
        for hpp in tests_dir.rglob("*.hpp"):
            content = _read(hpp)
            if any(kw in content for kw in PROFILING_KEYWORDS):
                modules.add(mod_dir.name)
                break
    return sorted(modules)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _all_files(directory: Path, pattern: str) -> list:
    if not directory.exists():
        return []
    files = list(directory.rglob(pattern))
    # Если ищем .hpp — добавляем ещё и .h (некоторые модули используют .h)
    if pattern.endswith(".hpp"):
        files += list(directory.rglob(pattern[:-3] + "h"))
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Проверка benchmark-класса: GpuBenchmarkBase vs кастомная реализация
# ─────────────────────────────────────────────────────────────────────────────

def check_benchmark(tests_dir: Path, results: dict):
    if not tests_dir.exists():
        return

    all_hpp = _all_files(tests_dir, "*.hpp")

    # Разбиваем: файлы с классом-бенчмарком (без test_*) и test runner'ы (test_*)
    bench_class_files = [f for f in all_hpp if "benchmark" in f.name and "test_" not in f.name]
    runner_files      = [f for f in all_hpp if f.name.startswith("test_") and "benchmark" in f.name]
    all_bench_files   = [f for f in all_hpp if "benchmark" in f.name]

    # Определяем: использует ли модуль GpuBenchmarkBase
    uses_base = any("GpuBenchmarkBase" in _read(f) for f in all_bench_files)

    if uses_base:
        _check_benchmark_standard(bench_class_files, runner_files, results)
    else:
        _check_benchmark_custom(all_bench_files + runner_files + all_hpp, results)


def _check_benchmark_standard(bench_class_files: list, runner_files: list, results: dict):
    """Стандартный паттерн: наследник GpuBenchmarkBase."""

    for bf in bench_class_files:
        content = _read(bf)
        is_rocm = "_rocm" in bf.name
        label   = "ROCm" if is_rocm else "OpenCL"

        # ExecuteKernel / ExecuteKernelTimed
        has_ek  = "ExecuteKernel()" in content
        has_ekt = "ExecuteKernelTimed()" in content
        if has_ek and has_ekt:
            results["checks"].append(("OK", f"{label} benchmark: ExecuteKernel + ExecuteKernelTimed ✓"))
        else:
            results["ok"] = False
            missing = []
            if not has_ek:  missing.append("ExecuteKernel()")
            if not has_ekt: missing.append("ExecuteKernelTimed()")
            results["checks"].append(
                ("FAIL", f"{label} benchmark {bf.name}: нет {', '.join(missing)}")
            )

        # RecordEvent / RecordROCmEvent в ExecuteKernelTimed
        record_fn = "RecordROCmEvent" if is_rocm else "RecordEvent"
        if record_fn in content:
            results["checks"].append(("OK", f"{label} benchmark: {record_fn} ✓"))
        else:
            results["ok"] = False
            results["checks"].append(
                ("FAIL", f"{label} benchmark {bf.name}: нет {record_fn} в ExecuteKernelTimed")
            )

        # output_dir
        if "Results/Profiler" not in content:
            results["warnings"].append(f"{label} benchmark {bf.name}: output_dir не содержит Results/Profiler")

    # Test runner'ы
    for tr in runner_files:
        content = _read(tr)
        is_rocm = "rocm" in tr.name

        if not is_rocm:
            # OpenCL: обязательно CL_QUEUE_PROFILING_ENABLE
            if "clCreateCommandQueue" in content:
                if "CL_QUEUE_PROFILING_ENABLE" in content:
                    results["checks"].append(("OK", f"OpenCL runner {tr.name}: CL_QUEUE_PROFILING_ENABLE ✓"))
                else:
                    results["ok"] = False
                    results["checks"].append(
                        ("FAIL", f"OpenCL runner {tr.name}: нет CL_QUEUE_PROFILING_ENABLE!")
                    )

        # IsProfEnabled перед Run()
        if "bench.Run()" in content or ".Run()" in content:
            if "IsProfEnabled" in content:
                results["checks"].append(("OK", f"Runner {tr.name}: IsProfEnabled ✓"))
            else:
                results["warnings"].append(f"Runner {tr.name}: Run() без проверки IsProfEnabled")


def _check_benchmark_custom(all_files: list, results: dict):
    """Кастомная реализация без GpuBenchmarkBase."""
    combined = "\n".join(_read(f) for f in all_files)

    has_hipEvent   = "hipEvent" in combined
    has_record     = "profiler.Record" in combined or "profiler.record" in combined
    has_print      = "PrintReport" in combined
    has_export     = "ExportMarkdown" in combined or "ExportJSON" in combined
    has_set_gpu    = "SetGPUInfo" in combined

    # Кастомная реализация — не ошибка, но нужен INFO-отчёт
    results["is_custom"] = True
    results["warnings"].append(
        "Кастомная реализация (без GpuBenchmarkBase): нет разбивки Upload/Kernel/Download"
    )
    # pipeline_diagram строится позже в check_module (нужен mod_dir)

    # hipEvent для GPU-таймера
    if has_hipEvent:
        results["checks"].append(("OK", "Custom: hipEvent GPU-таймер ✓"))
        results["info"].append("Замер: hipEvent оборачивает ВЕСЬ pipeline (одно монолитное время)")
        results["info"].append("  Стадии НЕ разделены — в отчёте одна запись на конфиг (N/K/S)")
        results["info"].append("  H2D входных данных НЕ включён (генерация на GPU через kernel)")
        results["info"].append("  H2D референса НЕ включён (PrepareReference вызывается до замера)")
        results["info"].append("  Включено: generate_inputs + R2C FFT + multiply_conj + IFFT + extract + D2H peaks")
    else:
        results["warnings"].append("Custom: hipEvent не найден — возможно CPU chrono (неточный замер)")

    # GPUProfiler.Record
    if has_record:
        results["checks"].append(("OK", "Custom: profiler.Record ✓"))
    else:
        results["ok"] = False
        results["checks"].append(("FAIL", "Custom: нет profiler.Record — данные не попадут в GPUProfiler"))

    # SetGPUInfo перед Start
    if has_set_gpu:
        results["checks"].append(("OK", "Custom: SetGPUInfo ✓"))
    else:
        results["warnings"].append("Custom: SetGPUInfo не найден (в отчёте будет Unknown GPU/driver)")

    # PrintReport + Export
    if has_print:
        results["checks"].append(("OK", "Custom: PrintReport ✓"))
    else:
        results["ok"] = False
        results["checks"].append(("FAIL", "Custom: нет PrintReport — вывод профилирования отсутствует"))

    if has_export:
        results["checks"].append(("OK", "Custom: ExportMarkdown/JSON ✓"))
    else:
        results["warnings"].append("Custom: нет ExportMarkdown/ExportJSON")


# ─────────────────────────────────────────────────────────────────────────────
# Построение ASCII pipeline для кастомной реализации
# ─────────────────────────────────────────────────────────────────────────────

def _extract_measured_block(tests_dir: Path) -> tuple[str, str, str]:
    """
    Возвращает (block_code, warmup_n, bench_n):
      block_code — код между hipEventRecord(ev_start) и hipEventRecord(ev_stop)
    """
    for f in _all_files(tests_dir, "*.hpp"):
        content = _read(f)
        if "hipEventRecord" not in content:
            continue
        m = re.search(
            r'hipEventRecord[^;]+ev_start[^;]*;(.*?)hipEventRecord[^;]+ev_stop',
            content, re.DOTALL
        )
        if not m:
            continue
        block = m.group(1)
        mw = re.search(r'k?[Ww]armup\w*\s*=\s*(\d+)|warmup\s*,\s*(\d+)', content)
        mb = re.search(r'k?[Bb]ench\w*[Rr]uns\s*=\s*(\d+)|runs\s*=\s*(\d+)', content)
        warmup = (mw.group(1) or mw.group(2)) if mw else "?"
        bench  = (mb.group(1) or mb.group(2)) if mb else "?"
        return block, warmup, bench
    return "", "?", "?"


def _extract_fn_body(src_content: str, fn_name: str) -> str:
    """
    Извлекает тело функции fn_name из C++ кода по подсчёту скобок.
    Если есть несколько определений (фасад + реализация) — берёт самое длинное.
    """
    # Лимит [^;{]{0,300} — защита от катастрофического backtracking
    pattern = re.compile(
        rf'\b{re.escape(fn_name)}\s*\([^;{{]{{0,300}}\)\s*(?:const\s*)?\{{',
        re.MULTILINE
    )
    # Объединяем тела ВСЕХ перегрузок — видим все внутренние вызовы
    all_bodies: list[str] = []
    for m in pattern.finditer(src_content):
        start = src_content.rfind('{', m.start(), m.end())
        if start == -1:
            continue
        depth = 0
        body_end = -1
        for i in range(start, len(src_content)):
            if src_content[i] == '{':
                depth += 1
            elif src_content[i] == '}':
                depth -= 1
                if depth == 0:
                    body_end = i
                    break
        if body_end != -1:
            all_bodies.append(src_content[start + 1 : body_end])
    return '\n'.join(all_bodies)


def _gpu_ops_from_body(body: str, stream_hint: str = "") -> list:
    """
    Извлекает GPU-операции из тела функции в порядке появления.
    Возвращает [(stream, label, api), ...]
    """
    ops = []
    seen = set()  # дедупликация по (label, api)

    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        # Определяем stream из переменной
        stream = stream_hint
        m_s = re.search(r'stream(\d+)_', line)
        if m_s:
            stream = f"Stream {m_s.group(1)}"

        entry = None

        if re.search(r'hipModuleLaunchKernel\b', line):
            m_k = re.search(r'fn_(\w+?)_[,\s\)]', line)
            kname = m_k.group(1).replace('_', ' ') if m_k else "kernel"
            entry = (stream or "GPU", kname, "hipModuleLaunchKernel")

        elif re.search(r'hipLaunchKernelGGL|hipLaunchKernel\b', line):
            m_k = re.search(r'(?:GGL|Kernel)\s*\(\s*(\w+)', line)
            kname = m_k.group(1) if m_k else "kernel"
            entry = (stream or "GPU", kname, "hipLaunchKernel")

        elif 'hipfftExecR2C' in line:
            entry = (stream or "GPU", "R2C FFT", "hipfftExecR2C")
        elif 'hipfftExecC2R' in line:
            entry = (stream or "GPU", "C2R IFFT", "hipfftExecC2R")
        elif 'hipfftExecC2C' in line:
            entry = (stream or "GPU", "C2C FFT", "hipfftExecC2C")

        elif re.search(r'hipMemcpyHtoDAsync|hipMemcpyAsync', line):
            kind = "H2D (async)" if ("HtoD" in line or "HostToDevice" in line) else "D2H (async)"
            entry = (stream or "GPU", kind, "hipMemcpyAsync")
        elif re.search(r'hipMemcpyHtoD\b', line):
            entry = (stream or "GPU", "H2D (sync)", "hipMemcpyHtoD")
        elif re.search(r'hipMemcpyDtoH\b', line):
            entry = (stream or "GPU", "D2H (sync)", "hipMemcpyDtoH")

        elif 'hipStreamSynchronize' in line:
            entry = ("⟲", "stream sync", "hipStreamSynchronize")

        elif re.search(r'(rocblas_|rocsolver_)\w+\s*\(', line):
            m_rb = re.search(r'((?:rocblas_|rocsolver_)\w+)\s*\(', line)
            entry = (stream or "GPU", m_rb.group(1) if m_rb else "rocblas", "rocBLAS/rocSOLVER")

        if entry:
            key = (entry[1], entry[2])
            if key not in seen:
                seen.add(key)
                ops.append(entry)

    return ops


def _build_pipeline_diagram(mod_dir: Path) -> list[str]:
    """Строит ASCII pipeline diagram для кастомной реализации."""
    tests_dir = mod_dir / "tests"
    src_dir   = mod_dir / "src"

    block, warmup, bench = _extract_measured_block(tests_dir)
    if not block:
        return []

    # Что вызывается внутри измеряемого блока
    measured_calls = re.findall(r'(\w+(?:\.\w+)?)\s*\(', block)
    measured_calls = [c for c in measured_calls
                      if not c.startswith(("hip", "void", "auto", "int", "float",
                                           "drv_gpu_lib", "std", "static"))]
    measured_label = measured_calls[0] if measured_calls else "Process()"

    # GPU ops: извлекаем только из тела измеряемой функции и её sub-функций
    all_src_files = _all_files(src_dir, "*.cpp")
    combined_src  = "\n".join(_read(f) for f in all_src_files)

    # Фильтруем C++ ключевые слова
    CPP_KEYWORDS = {"if", "for", "while", "switch", "return", "static", "const",
                    "auto", "void", "int", "float", "double", "bool", "char",
                    "new", "delete", "nullptr", "true", "false", "sizeof",
                    "reinterpret_cast", "static_cast", "dynamic_cast"}
    # Имя метода (без объекта): "corr.RunTestPattern" → "RunTestPattern"
    fn_short = measured_label.split(".")[-1].split("(")[0]
    if fn_short.lower() in CPP_KEYWORDS or len(fn_short) < 3:
        # Ищем следующий подходящий вызов
        for c in measured_calls[1:]:
            short = c.split(".")[-1].split("(")[0]
            if short.lower() not in CPP_KEYWORDS and len(short) >= 3:
                measured_label = c
                fn_short = short
                break

    # Тело основной функции
    main_body = _extract_fn_body(combined_src, fn_short)

    # Sub-функции, вызываемые внутри (ищем вызовы вида FnName()  без аргументов-примитивов)
    # Исключаем fn_short (рекурсия) и служебные CamelCase слова
    _SKIP_SUB = {"InputData", "OutputData", "ResolveMatrixSize", "ExportClBufferToFd",
                 "ImportFromOpenCl", "GetHipPtr", "Allocate", "MemcpyDeviceToDevice",
                 "GetNativeQueue", "CheckInfo", "Synchronize", "CompileKernels",
                 "SetGPUInfo", "PrintReport", "ExportMarkdown", "ExportJSON"}
    sub_fn_calls = re.findall(r'\b([A-Z]\w+)\s*\(', main_body)
    sub_fn_calls = [c for c in sub_fn_calls
                    if c != fn_short and c not in _SKIP_SUB]
    sub_fn_calls = list(dict.fromkeys(sub_fn_calls))[:6]  # max 6, без дублей

    # Собираем ops: сначала из main, потом из sub
    main_ops = _gpu_ops_from_body(main_body)
    sub_ops_map: dict = {}
    for sfn in sub_fn_calls:
        body = _extract_fn_body(combined_src, sfn)
        if body:
            ops = _gpu_ops_from_body(body)
            if ops:
                sub_ops_map[sfn] = ops

    # Строим диаграмму
    D = "  "
    lines = [
        "",
        f"{D}hipEventRecord(ev_start)",
        f"{D}  │",
        f"{D}  ▼",
        f"{D}  {measured_label}()",
    ]

    # Ops из основной функции (до вызова sub-функций)
    if main_ops or sub_ops_map:
        has_sub = bool(sub_ops_map)
        all_entries = []  # (is_subfn, name_or_none, ops)

        # Простая логика: main_ops → затем sub_ops
        if main_ops:
            all_entries.append((False, None, main_ops))
        for sfn, sops in sub_ops_map.items():
            all_entries.append((True, sfn, sops))

        for entry_idx, (is_sub, sub_name, ops) in enumerate(all_entries):
            is_last_entry = (entry_idx == len(all_entries) - 1)

            if is_sub:
                lines.append(f"{D}    │")
                lines.append(f"{D}    └── {sub_name}()")
                op_indent = f"{D}          "
            else:
                op_indent = f"{D}    "

            for i, (stream, label, api) in enumerate(ops):
                is_last_op = (i == len(ops) - 1) and (is_last_entry or not is_sub)
                connector  = "└──" if (is_last_op and is_sub) else "├──"
                s_tag      = f"[{stream}] " if stream not in ("⟲", "", None) else ""
                sync_mark  = "⟲ " if stream == "⟲" else ""
                lines.append(f"{op_indent}{connector} {s_tag}{sync_mark}{label}  ({api})")
    else:
        lines.append(f"{D}    (GPU операции не найдены в src/)")

    lines += [
        f"{D}  │",
        f"{D}  ▼",
        f"{D}hipEventRecord(ev_stop)",
        f"{D}hipEventElapsedTime → avg_ms  "
        f"(warmup={warmup}, runs={bench}, среднее за {bench} прогонов)",
        "",
        f"{D}В GPUProfiler: одна запись на конфиг — монолитное суммарное время",
        f"{D}Для разбивки по стадиям — переход на GpuBenchmarkBase",
        "",
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Проверка production-класса (только для стандартного паттерна)
# ─────────────────────────────────────────────────────────────────────────────

def check_production(mod_dir: Path, results: dict):
    """
    Проверяет production-класс модуля:
    - prof_events* / ROCmProfEvents* в сигнатурах методов (include)
    - CollectOrRelease в src (OpenCL)
    - MakeROCmDataFromEvents / MakeROCmDataFromClock в src (ROCm)
    """
    tests_dir   = mod_dir / "tests"
    include_dir = mod_dir / "include"
    src_dir     = mod_dir / "src"

    # Проверяем только для стандартного паттерна (GpuBenchmarkBase)
    all_bench = _all_files(tests_dir, "*.hpp")
    if not any("GpuBenchmarkBase" in _read(f) for f in all_bench):
        return  # Кастомная реализация — production-класс чистый по дизайну

    all_headers = _all_files(include_dir, "*.hpp")
    all_src     = _all_files(src_dir, "*.cpp")

    # ── OpenCL production ─────────────────────────────────────────────────
    ocl_headers = [f for f in all_headers if "_rocm" not in f.name]
    ocl_src     = [f for f in all_src if "_rocm" not in f.name]
    has_ocl_bench = any(
        "_rocm" not in f.name and "GpuBenchmarkBase" in _read(f)
        for f in all_bench
    )

    if has_ocl_bench:
        # prof_events в сигнатуре
        ocl_headers_content = "\n".join(_read(f) for f in ocl_headers)
        if "prof_events" in ocl_headers_content:
            results["checks"].append(("OK", "OpenCL production: prof_events* в сигнатуре ✓"))
        else:
            results["ok"] = False
            results["checks"].append(
                ("FAIL", "OpenCL production: нет prof_events* в сигнатуре метода (include/)")
            )

        # CollectOrRelease в src
        ocl_src_content = "\n".join(_read(f) for f in ocl_src)
        if "CollectOrRelease" in ocl_src_content:
            results["checks"].append(("OK", "OpenCL production: CollectOrRelease ✓"))
        else:
            results["ok"] = False
            results["checks"].append(
                ("FAIL", "OpenCL production: нет CollectOrRelease в src/ (cl_event не собирается)")
            )

    # ── ROCm production ───────────────────────────────────────────────────
    rocm_headers = [f for f in all_headers if "_rocm" in f.name]
    rocm_src     = [f for f in all_src if "_rocm" in f.name]
    has_rocm_bench = any(
        "_rocm" in f.name and "GpuBenchmarkBase" in _read(f)
        for f in all_bench
    )

    if has_rocm_bench:
        # ROCmProfEvents в сигнатуре
        rocm_headers_content = "\n".join(_read(f) for f in rocm_headers)
        if "ROCmProfEvents" in rocm_headers_content or "prof_events" in rocm_headers_content:
            results["checks"].append(("OK", "ROCm production: ROCmProfEvents* в сигнатуре ✓"))
        else:
            results["ok"] = False
            results["checks"].append(
                ("FAIL", "ROCm production: нет ROCmProfEvents* в сигнатуре метода (include/)")
            )

        # MakeROCmDataFromEvents / MakeROCmDataFromClock в src
        rocm_src_content = "\n".join(_read(f) for f in rocm_src)
        has_helpers = ("MakeROCmDataFromEvents" in rocm_src_content or
                       "MakeROCmDataFromClock"  in rocm_src_content)
        if has_helpers:
            results["checks"].append(("OK", "ROCm production: MakeROCmDataFrom* хелперы ✓"))
        else:
            results["ok"] = False
            results["checks"].append(
                ("FAIL", "ROCm production: нет MakeROCmDataFromEvents/Clock в src/ (timing не реализован)")
            )

        # if (prof_events) блоки в src
        has_if_prof = bool(re.search(r"if\s*\(\s*prof_events\s*\)", rocm_src_content))
        if has_if_prof:
            results["checks"].append(("OK", "ROCm production: if(prof_events) guard ✓"))
        else:
            results["warnings"].append(
                "ROCm production: не найден if(prof_events) — hipEvents могут создаваться всегда"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Проверка запрещённых паттернов
# ─────────────────────────────────────────────────────────────────────────────

def check_forbidden(tests_dir: Path, src_dir: Path, results: dict):
    """ЗАПРЕЩЕНО: GetStats() + цикл + con.Print/cout для вывода профилирования."""
    scan = _all_files(tests_dir, "*.hpp") + _all_files(src_dir, "*.cpp")

    for f in scan:
        content = _read(f)
        if not re.search(r"GetStats\s*\(", content):
            continue
        if "PrintReport" in content or "bench.Report" in content:
            continue  # Есть правильный вывод

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "GetStats" not in line:
                continue
            ctx = "\n".join(lines[max(0, i - 3): i + 12])
            if re.search(r"(for|while)\s*\(", ctx) and ("con.Print" in ctx or "std::cout" in ctx):
                results["ok"] = False
                results["checks"].append(
                    ("FAIL", f"{f.name}: ЗАПРЕЩЕНО GetStats + цикл + Print для вывода профилирования")
                )
                break


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция проверки модуля
# ─────────────────────────────────────────────────────────────────────────────

def check_module(module_name: str) -> dict:
    results = {"module": module_name, "ok": True, "checks": [], "warnings": [],
               "info": [], "is_custom": False, "pipeline_diagram": []}
    mod_dir = MODULES_DIR / module_name

    if not mod_dir.exists():
        results["ok"] = False
        results["checks"].append(("ERROR", f"Модуль не найден: {mod_dir}"))
        return results

    tests_dir = mod_dir / "tests"
    src_dir   = mod_dir / "src"

    check_benchmark(tests_dir, results)
    check_production(mod_dir, results)
    check_forbidden(tests_dir, src_dir, results)

    if results.get("is_custom"):
        results["pipeline_diagram"] = _build_pipeline_diagram(mod_dir)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def _format_results(module_results: list) -> tuple[list[str], int, int]:
    """Форматирует вывод. Возвращает (строки, кол-во ошибок, кол-во предупреждений)."""
    lines = []
    total_fail = 0
    total_warn = 0
    for r in module_results:
        status = "✅" if r["ok"] else "❌"
        lines.append(f"\n{status} {r['module']}")
        for check_type, msg in r["checks"]:
            prefix = "  ✓" if check_type == "OK" else "  ✗"
            lines.append(f"{prefix} {msg}")
            if check_type != "OK":
                total_fail += 1
        for w in r["warnings"]:
            lines.append(f"  ⚠ {w}")
            total_warn += 1
        for info_msg in r.get("info", []):
            lines.append(f"  ℹ {info_msg}")
        if r.get("pipeline_diagram"):
            lines.append(f"\n  {'─' * 54}")
            lines.append(f"  Измеряемый GPU pipeline ({r['module']}):")
            for diag_line in r["pipeline_diagram"]:
                lines.append(diag_line)
            lines.append(f"  {'─' * 54}")
    return lines, total_fail, total_warn


def _save_report(module_results: list, total_fail: int, total_warn: int):
    """Сохраняет отчёт в Results/profiling_audit/ при наличии ошибок или предупреждений."""
    audit_dir = REPO_ROOT / "Results" / "profiling_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "FAIL" if total_fail > 0 else "WARN"
    report_path = audit_dir / f"{timestamp}_{suffix}.txt"

    lines = []
    lines.append("GPU Profiling Implementation Audit")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Ref: {DOC_REF}")
    lines.append("=" * 60)

    for r in module_results:
        status = "FAIL" if not r["ok"] else ("CUSTOM" if r.get("is_custom") else "OK")
        lines.append(f"\n[{status}] {r['module']}")

        for check_type, msg in r["checks"]:
            prefix = "  [OK]  " if check_type == "OK" else "  [FAIL]"
            lines.append(f"{prefix} {msg}")
        for w in r["warnings"]:
            lines.append(f"  [WARN] {w}")
        for info_msg in r.get("info", []):
            lines.append(f"  [INFO] {info_msg}")

        # Pipeline diagram для кастомных реализаций
        if r.get("pipeline_diagram"):
            lines.append("")
            lines.append("  ┌─ Измеряемый GPU pipeline ─────────────────────────────┐")
            for diag_line in r["pipeline_diagram"]:
                lines.append(diag_line)
            lines.append("  └────────────────────────────────────────────────────────┘")

    lines.append("\n" + "=" * 60)
    lines.append(f"Modules checked : {len(module_results)}")
    lines.append(f"Errors          : {total_fail}")
    lines.append(f"Warnings        : {total_warn}")
    if any(r.get("is_custom") for r in module_results):
        custom = [r["module"] for r in module_results if r.get("is_custom")]
        lines.append(f"Custom impl     : {', '.join(custom)}")
        lines.append("  → Монолитный замер всего pipeline (нет разбивки по стадиям)")
        lines.append("  → Для детальных данных рекомендуется GpuBenchmarkBase (см. GPU_Profiling_Mechanism.md §2)")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("\nМодули с профилированием:", ", ".join(find_profiling_modules()))
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        modules = find_profiling_modules()
    elif len(sys.argv) > 1:
        modules = [sys.argv[1]]
    else:
        modules = find_profiling_modules()

    print("=" * 60)
    print("  GPU Profiling Implementation Check")
    print(f"  Ref: {DOC_REF}")
    print("=" * 60)

    module_results = [check_module(m) for m in modules]
    output_lines, total_fail, total_warn = _format_results(module_results)

    for line in output_lines:
        print(line)

    print("\n" + "=" * 60)
    print(f"  Проверено модулей: {len(modules)}")

    need_report = total_fail > 0 or total_warn > 0
    if need_report:
        if total_fail > 0:
            print(f"  Ошибок: {total_fail}")
        if total_warn > 0:
            print(f"  Предупреждений: {total_warn}")
        report_path = _save_report(module_results, total_fail, total_warn)
        print(f"  Отчёт: {report_path.relative_to(REPO_ROOT)}")
        print("=" * 60)
        if total_fail > 0:
            sys.exit(1)
    else:
        print("  Всё ОК ✓")
        print("=" * 60)


if __name__ == "__main__":
    main()
