"""
result_store.py — ResultStore (Repository)
============================================

Единая точка доступа к артефактам тестов (массивы, JSON, бенчмарки).

Маршрутизация:
    store.save_array(gpu_out, "cw_4096", module="signal_generators")
        → Results/Arrays/signal_generators/cw_4096.npy
    store.save_comparison(gpu_out, ref_out, "cw_vs_numpy", module=...)
        → Results/Arrays/signal_generators/cw_vs_numpy.npz
    store.save_test_result(test_result, module="signal_generators")
        → Results/JSON/signal_generators/{test_result.test_name}.json
    store.save_benchmark({"ms": 1.5}, "fft", module="fft_processor")
        → Results/JSON/fft_processor/bench_fft.json

Корень `Results/` — в корне репозитория (ищется по маркеру `.git`),
что устойчиво к запуску из любого рабочего каталога.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..result import TestResult
from .json_store import JsonStore
from .numpy_store import NumpyStore


def _find_repo_root() -> Path:
    """Найти корень репозитория (маркер `.git`).

    Не используем hardcoded ``parents[N]`` — он молча ломается, если файл
    переедет. Проверяем ``.git`` как директорию (обычный clone) и как файл
    (git worktree, submodule). Fallback — 4 уровня вверх
    (``Python_test/common/io/result_store.py`` → корень).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        git_entry = parent / ".git"
        if git_entry.exists():
            return parent
    return here.parents[3]


class ResultStore:
    """Repository для артефактов тестов.

    Структура Results/ (в корне репо):
        Results/
        ├── Arrays/{module}/{name}.npy     ← numpy данные
        ├── Arrays/{module}/{name}.npz     ← пакеты (save_comparison)
        ├── JSON/{module}/{name}.json      ← TestResult, конфиги, бенчмарки
        └── Plots/{module}/*.png           ← графики (через PlotterFactory)

    Профилировщик (Results/Profiler/) — через GPUProfiler в C++.
    """

    _PROJECT_ROOT: Path = _find_repo_root()

    def __init__(self, base_dir: str | Path | None = None):
        """
        Args:
            base_dir: корень для Results/. По умолчанию — {repo_root}/Results.
                      Тесты передают tempfile.TemporaryDirectory для изоляции.
        """
        if base_dir is None:
            base = self._PROJECT_ROOT / "Results"
        else:
            base = Path(base_dir)
        self._numpy = NumpyStore(base / "Arrays")
        self._json  = JsonStore(base  / "JSON")

    # ── numpy arrays ────────────────────────────────────────────────────────

    def save_array(self, data: np.ndarray,
                   name: str, module: str = "") -> Path:
        """Сохранить numpy array → Results/Arrays/{module}/{name}.npy."""
        return self._numpy.save(data, name, subdir=module)

    def load_array(self, name: str, module: str = "") -> np.ndarray:
        """Загрузить numpy array."""
        return self._numpy.load(name, subdir=module)

    def save_comparison(self,
                        gpu_output: np.ndarray,
                        reference: np.ndarray,
                        name: str, module: str = "") -> Path:
        """Сохранить GPU-вывод и эталон в одном .npz.

        Ключи в .npz: ``"gpu"`` и ``"ref"``.
        """
        return self._numpy.save_many(
            {"gpu": gpu_output, "ref": reference},
            name, subdir=module,
        )

    def load_comparison(self, name: str,
                        module: str = "") -> dict[str, np.ndarray]:
        """Загрузить .npz с ключами gpu/ref."""
        return self._numpy.load_many(name, subdir=module)

    # ── TestResult / JSON ───────────────────────────────────────────────────

    def save_test_result(self, result, module: str = "") -> Path:
        """Сохранить TestResult → Results/JSON/{module}/{test_name}.json.

        Args:
            result: TestResult (из common.result) или dict с ключом ``test_name``.

        Raises:
            ValueError: если dict без ``test_name``.
            TypeError:  если не TestResult и не dict.
        """
        # isinstance — fail-loud: если TestResult потеряет to_dict/test_name
        # при рефакторинге, ошибка возникнет здесь, а не молча через dict-ветку.
        if isinstance(result, TestResult):
            data = result.to_dict()
            name = result.test_name
        elif isinstance(result, dict):
            data = result
            name = data.get("test_name")
            if not name:
                raise ValueError(
                    "dict result должен содержать ключ 'test_name'"
                )
        else:
            raise TypeError(
                f"save_test_result принимает TestResult или dict, "
                f"получено: {type(result).__name__}"
            )
        return self._json.save(data, name, subdir=module)

    def save_config(self, config, name: str, module: str = "") -> Path:
        """Сохранить конфиг → Results/JSON/{module}/{name}_config.json.

        Принимает dataclass, plain class (__dict__) или dict.
        """
        if hasattr(config, "__dataclass_fields__"):
            import dataclasses
            data = dataclasses.asdict(config)
        elif isinstance(config, dict):
            data = dict(config)
        elif hasattr(config, "__dict__"):
            data = vars(config)
        else:
            data = {"value": str(config)}
        return self._json.save(data, f"{name}_config", subdir=module)

    def save_benchmark(self, data: dict, name: str,
                       module: str = "") -> Path:
        """Сохранить бенчмарк → Results/JSON/{module}/bench_{name}.json."""
        return self._json.save(data, f"bench_{name}", subdir=module)

    def load_json(self, name: str, module: str = "") -> dict:
        """Загрузить любой JSON файл по имени."""
        return self._json.load(name, subdir=module)

    # ── утилиты ─────────────────────────────────────────────────────────────

    def list_arrays(self, module: str = "") -> list[str]:
        """Список сохранённых массивов в модуле."""
        return self._numpy.list(subdir=module)

    def list_results(self, module: str = "") -> list[str]:
        """Список сохранённых JSON в модуле."""
        return self._json.list(subdir=module)

    def array_exists(self, name: str, module: str = "") -> bool:
        """True если есть .npy ИЛИ .npz с таким именем."""
        return self._numpy.exists(name, subdir=module)

    def json_exists(self, name: str, module: str = "") -> bool:
        return self._json.exists(name, subdir=module)
