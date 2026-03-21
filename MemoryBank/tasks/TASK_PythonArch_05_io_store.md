# TASK_PythonArch_05 — common/io/ (ResultStore)

> **Фаза**: 2 (приоритет MEDIUM)
> **Зависимости**: TASK_Arch_04 (validators — для сохранения TestResult)
> **Статус**: ⬜ TODO
> **Оценка**: ~1.5 часа
> **Паттерны**: GoF Repository, GoF Strategy, GRASP Information Expert/High Cohesion, SOLID SRP

---

## 🎯 Цель

Создать `Python_test/common/io/` — унифицированный слой I/O.

**Проблема сейчас**:
- `JSONReporter` пишет в `Results/JSON/{name}.json` (через reporters.py)
- PlotConfig сохраняет в `Results/Plots/{module}/`
- Нет единого места для numpy arrays (входных данных, промежуточных результатов)
- Нет стандартного способа "сохранить данные для отладки"

**Решение**: `ResultStore` — единая точка входа для всего I/O тестов.
```python
store = ResultStore()
store.save_array(gpu_output, "cw_test_output", module="signal_generators")
store.save_test_result(result, module="signal_generators")
```

---

## 📁 Создаваемые файлы (5 штук)

```
Python_test/common/io/
├── __init__.py          ← 1. пакет + export
├── base.py              ← 2. IDataStore (ABC)
├── numpy_store.py       ← 3. NumpyStore (.npy, .npz)
├── json_store.py        ← 4. JsonStore (.json)
└── result_store.py      ← 5. ResultStore (координатор + пути)
```

---

## 📝 Детальное ТЗ

### 2. `common/io/base.py` — IDataStore (ABC)

```python
from abc import ABC, abstractmethod
from pathlib import Path

class IDataStore(ABC):
    """
    GoF Strategy: интерфейс для хранилища данных.
    SOLID ISP: только операции I/O, ничего лишнего.
    """

    @abstractmethod
    def save(self, data, name: str, subdir: str = "") -> Path:
        """
        Сохраняет данные.

        Args:
            data:   данные для сохранения (зависит от реализации)
            name:   имя файла (без расширения)
            subdir: подкаталог внутри base_dir (например, "signal_generators")

        Returns:
            Path: абсолютный путь к сохранённому файлу
        """

    @abstractmethod
    def load(self, name: str, subdir: str = ""):
        """
        Загружает данные по имени.

        Returns:
            данные (тип зависит от реализации)
        """

    @abstractmethod
    def exists(self, name: str, subdir: str = "") -> bool:
        """Проверяет наличие данных."""

    @abstractmethod
    def list(self, subdir: str = "") -> list[str]:
        """Список всех сохранённых имён в subdir."""
```

---

### 3. `common/io/numpy_store.py` — NumpyStore

```python
import numpy as np
from pathlib import Path
from .base import IDataStore

class NumpyStore(IDataStore):
    """
    Хранилище numpy arrays в .npy / .npz файлах.
    SRP: только работа с numpy arrays.

    Структура файлов:
        {base_dir}/{subdir}/{name}.npy      ← одиночный массив
        {base_dir}/{subdir}/{name}.npz      ← пакет массивов
    """

    def __init__(self, base_dir: str | Path = "Results/Arrays"):
        self._base = Path(base_dir)

    def save(self, data: np.ndarray, name: str, subdir: str = "") -> Path:
        """
        Сохраняет numpy array как .npy.

        Args:
            data: np.ndarray любого dtype и shape
            name: имя файла (без расширения)
        """
        path = self._resolve(name, subdir, ext=".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
        return path

    def save_many(self, arrays: dict[str, np.ndarray],
                  name: str, subdir: str = "") -> Path:
        """
        Сохраняет несколько массивов в .npz (сжатый).

        Args:
            arrays: {"key1": array1, "key2": array2, ...}

        Example:
            store.save_many({"gpu": gpu_out, "ref": ref_out}, "comparison")
            # → Results/Arrays/module/comparison.npz
        """
        path = self._resolve(name, subdir, ext=".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def load(self, name: str, subdir: str = "") -> np.ndarray:
        """Загружает .npy файл."""
        path = self._resolve(name, subdir, ext=".npy")
        if not path.exists():
            raise FileNotFoundError(f"Array not found: {path}")
        return np.load(path)

    def load_many(self, name: str, subdir: str = "") -> dict[str, np.ndarray]:
        """Загружает .npz файл → dict массивов."""
        path = self._resolve(name, subdir, ext=".npz")
        if not path.exists():
            raise FileNotFoundError(f"NPZ not found: {path}")
        npz = np.load(path)
        return {k: npz[k] for k in npz.files}

    def exists(self, name: str, subdir: str = "") -> bool:
        return self._resolve(name, subdir, ext=".npy").exists()

    def list(self, subdir: str = "") -> list[str]:
        """Список всех .npy/.npz файлов в subdir (без расширения)."""
        d = self._base / subdir if subdir else self._base
        if not d.exists():
            return []
        names = set()
        for f in d.iterdir():
            if f.suffix in (".npy", ".npz"):
                names.add(f.stem)
        return sorted(names)

    def _resolve(self, name: str, subdir: str, ext: str) -> Path:
        base = self._base / subdir if subdir else self._base
        return base / f"{name}{ext}"
```

---

### 4. `common/io/json_store.py` — JsonStore

```python
import json
from pathlib import Path
from datetime import datetime
from .base import IDataStore

class JsonStore(IDataStore):
    """
    Хранилище dict / TestResult в .json файлах.
    SRP: только JSON I/O.

    Структура:
        {base_dir}/{subdir}/{name}.json
    """

    def __init__(self, base_dir: str | Path = "Results/JSON"):
        self._base = Path(base_dir)

    def save(self, data: dict, name: str, subdir: str = "") -> Path:
        """
        Сохраняет dict в JSON.

        Автоматически добавляет "saved_at": ISO timestamp.
        """
        path = self._resolve(name, subdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().isoformat(),
            **data,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return path

    def load(self, name: str, subdir: str = "") -> dict:
        """Загружает JSON → dict."""
        path = self._resolve(name, subdir)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self, name: str, subdir: str = "") -> bool:
        return self._resolve(name, subdir).exists()

    def list(self, subdir: str = "") -> list[str]:
        d = self._base / subdir if subdir else self._base
        if not d.exists():
            return []
        return sorted(f.stem for f in d.iterdir() if f.suffix == ".json")

    def _resolve(self, name: str, subdir: str) -> Path:
        base = self._base / subdir if subdir else self._base
        return base / f"{name}.json"
```

---

### 5. `common/io/result_store.py` — ResultStore (Repository)

```python
from pathlib import Path
from .numpy_store import NumpyStore
from .json_store import JsonStore
import numpy as np

class ResultStore:
    """
    GoF Repository: единая точка доступа к результатам тестов.
    GRASP Information Expert: знает правила именования и расположения файлов.
    GRASP High Cohesion: весь I/O тестов — здесь.

    Структура Results/:
        Results/
        ├── Arrays/{module}/{name}.npy     ← numpy данные (GPU output, ref)
        ├── JSON/{module}/{name}.json      ← TestResult, конфигурации
        └── Plots/{module}/*.png           ← графики (через PlotterFactory)

    Профилировщик (Results/Profiler/) — через GPUProfiler в C++.

    Пример:
        store = ResultStore()

        # Сохранить GPU output для отладки:
        store.save_array(gpu_signal, "cw_4096", module="signal_generators")

        # Сохранить сравнение GPU vs NumPy:
        store.save_comparison(gpu_out, ref_out, "cw_test", module="signal_generators")

        # Загрузить позже:
        prev = store.load_array("cw_4096", module="signal_generators")
    """

    # Абсолютный путь: корень проекта / Results
    # Не ломается если тест запущен из другого каталога
    _PROJECT_ROOT = Path(__file__).parents[2]

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base = self._PROJECT_ROOT / "Results"
        else:
            base = Path(base_dir)
        self._numpy = NumpyStore(base / "Arrays")
        self._json  = JsonStore(base  / "JSON")

    # ── numpy arrays ──────────────────────────────────────────────────────────

    def save_array(self, data: np.ndarray,
                   name: str, module: str = "") -> Path:
        """
        Сохраняет numpy array.
        Путь: Results/Arrays/{module}/{name}.npy
        """
        return self._numpy.save(data, name, subdir=module)

    def load_array(self, name: str, module: str = "") -> np.ndarray:
        """Загружает numpy array."""
        return self._numpy.load(name, subdir=module)

    def save_comparison(self,
                        gpu_output: np.ndarray,
                        reference: np.ndarray,
                        name: str, module: str = "") -> Path:
        """
        Сохраняет GPU-вывод И эталон вместе в .npz для удобного сравнения.
        Путь: Results/Arrays/{module}/{name}.npz

        Загрузка:
            data = store.load_comparison("cw_test", "signal_generators")
            gpu = data["gpu"]
            ref = data["ref"]
        """
        return self._numpy.save_many(
            {"gpu": gpu_output, "ref": reference},
            name, subdir=module
        )

    def load_comparison(self, name: str, module: str = "") -> dict:
        """Загружает .npz с GPU-выводом и эталоном."""
        return self._numpy.load_many(name, subdir=module)

    # ── TestResult / JSON ─────────────────────────────────────────────────────

    def save_test_result(self, result, module: str = "") -> Path:
        """
        Сохраняет TestResult.
        Путь: Results/JSON/{module}/{result.test_name}.json

        Args:
            result: TestResult или dict
        """
        if hasattr(result, "to_dict"):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            data = {"result": str(result)}
        name = data.get("test_name", "unknown")
        return self._json.save(data, name, subdir=module)

    def save_config(self, config, name: str, module: str = "") -> Path:
        """
        Сохраняет конфигурацию теста.
        Путь: Results/JSON/{module}/{name}_config.json
        """
        if hasattr(config, "__dict__"):
            data = vars(config)
        elif hasattr(config, "__dataclass_fields__"):
            import dataclasses
            data = dataclasses.asdict(config)
        else:
            data = dict(config)
        return self._json.save(data, f"{name}_config", subdir=module)

    def save_benchmark(self, data: dict, name: str, module: str = "") -> Path:
        """
        Сохраняет результаты бенчмарка.
        Путь: Results/JSON/{module}/bench_{name}.json
        """
        return self._json.save(data, f"bench_{name}", subdir=module)

    def load_json(self, name: str, module: str = "") -> dict:
        """Загружает любой JSON файл."""
        return self._json.load(name, subdir=module)

    # ── утилиты ───────────────────────────────────────────────────────────────

    def list_arrays(self, module: str = "") -> list[str]:
        """Список сохранённых массивов в модуле."""
        return self._numpy.list(subdir=module)

    def list_results(self, module: str = "") -> list[str]:
        """Список сохранённых JSON в модуле."""
        return self._json.list(subdir=module)

    def array_exists(self, name: str, module: str = "") -> bool:
        return self._numpy.exists(name, subdir=module)
```

---

### 1. `common/io/__init__.py`

```python
"""
I/O слой тестов GPUWorkLib.

Использование:
    from common.io import ResultStore

    store = ResultStore()
    store.save_array(gpu_out, "my_test", module="signal_generators")
    store.save_comparison(gpu_out, ref, "cw_vs_numpy", module="signal_generators")
    store.save_test_result(result, module="signal_generators")
"""

from .base import IDataStore
from .numpy_store import NumpyStore
from .json_store import JsonStore
from .result_store import ResultStore

__all__ = [
    "IDataStore",
    "NumpyStore",
    "JsonStore",
    "ResultStore",
]
```

---

## 📋 Что обновить

### `common/__init__.py` — добавить:
```python
from .io import ResultStore, NumpyStore, JsonStore
```

### `common/reporters.py` — обновить JSONReporter (опционально):
```python
# Можно переключить JSONReporter на использование JsonStore
# Это не обязательно сразу — но желательно для унификации
from .io import JsonStore

class JSONReporter(IReporter):
    def __init__(self, module: str = ""):
        self._store = JsonStore()
        self._module = module

    def on_suite_finished(self, results):
        for r in results:
            self._store.save(r.to_dict(), r.test_name, subdir=self._module)
```

---

## ✅ Критерии завершения

- [ ] Все 5 файлов созданы
- [ ] `from common.io import ResultStore` работает
- [ ] `store.save_array(np.ones(100), "test", "module")` → создаёт файл
- [ ] `store.load_array("test", "module")` → загружает обратно
- [ ] `store.save_comparison(gpu, ref, "name")` → .npz с ключами "gpu", "ref"
- [ ] `store.save_test_result(result, "module")` → JSON файл
- [ ] `store.save_benchmark({"ms": 1.5}, "fft", "module")` → bench_fft.json
- [ ] `store.list_arrays("module")` → список имён
- [ ] Создаются правильные пути: `Results/Arrays/{module}/{name}.npy`

## 🧪 Проверка без GPU

```python
# python Python_test/common/io/test_io_smoke.py
import numpy as np
import tempfile
from pathlib import Path
from common.runner import TestRunner
from common.result import TestResult, ValidationResult
from common.io import ResultStore

class TestIOSmoke:
    """Smoke-тест I/O — работает без GPU."""

    def test_save_load_array(self):
        result = TestResult(test_name="save_load_array")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            data = np.random.randn(100).astype(np.float32)
            p = store.save_array(data, "test_arr", "mymodule")
            loaded = store.load_array("test_arr", "mymodule")
            ok = p.exists() and np.allclose(data, loaded)
            result.add(ValidationResult(
                passed=ok,
                metric_name="roundtrip",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0,
                message=f"path={p}"
            ))
        return result

    def test_save_comparison(self):
        result = TestResult(test_name="save_comparison")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            data = np.random.randn(100).astype(np.float32)
            ref = np.ones(100, np.float32)
            store.save_comparison(data, ref, "comparison", "mymodule")
            cmp = store.load_comparison("comparison", "mymodule")
            ok = "gpu" in cmp and "ref" in cmp
            result.add(ValidationResult(
                passed=ok,
                metric_name="keys",
                actual_value=1.0 if ok else 0.0,
                threshold=1.0
            ))
        return result

    def test_save_benchmark_json(self):
        result = TestResult(test_name="benchmark_json")
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(base_dir=tmp)
            store.save_benchmark({"ms_per_call": 1.23}, "fft", "mymodule")
            j = store.load_json("bench_fft", "mymodule")
            ok = j["ms_per_call"] == 1.23
            result.add(ValidationResult(
                passed=ok,
                metric_name="json_roundtrip",
                actual_value=j.get("ms_per_call", 0),
                threshold=1.23
            ))
        return result

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run(TestIOSmoke())
    runner.print_summary(results)
```

---

*Создан: 2026-03-21 | Кодо | Фаза 2*
