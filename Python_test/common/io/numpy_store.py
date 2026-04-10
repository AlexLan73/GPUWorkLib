"""
numpy_store.py — NumpyStore (I/O для numpy arrays)
====================================================

Хранит numpy-массивы в .npy (одиночные) и .npz (пакеты).
SRP: только работа с numpy, никакого JSON/графиков.

Структура файлов:
    {base_dir}/{subdir}/{name}.npy      ← одиночный массив
    {base_dir}/{subdir}/{name}.npz      ← пакет массивов (save_many)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import IDataStore


class NumpyStore(IDataStore):
    """Хранилище numpy arrays."""

    def __init__(self, base_dir: str | Path = "Results/Arrays"):
        self._base = Path(base_dir)

    # ── save / load одиночного .npy ──────────────────────────────────────────

    def save(self, data: np.ndarray, name: str, subdir: str = "") -> Path:
        """Сохранить numpy array в .npy."""
        path = self._resolve(name, subdir, ext=".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
        return path

    def load(self, name: str, subdir: str = "") -> np.ndarray:
        """Загрузить .npy файл."""
        path = self._resolve(name, subdir, ext=".npy")
        if not path.exists():
            raise FileNotFoundError(f"Array not found: {path}")
        return np.load(path)

    # ── save / load нескольких массивов (.npz) ───────────────────────────────

    def save_many(self, arrays: dict[str, np.ndarray],
                  name: str, subdir: str = "") -> Path:
        """Сохранить несколько массивов в .npz (сжатый).

        Example:
            store.save_many({"gpu": gpu_out, "ref": ref_out}, "comparison")
            # → Results/Arrays/{module}/comparison.npz
        """
        path = self._resolve(name, subdir, ext=".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    def load_many(self, name: str, subdir: str = "") -> dict[str, np.ndarray]:
        """Загрузить .npz файл как dict массивов."""
        path = self._resolve(name, subdir, ext=".npz")
        if not path.exists():
            raise FileNotFoundError(f"NPZ not found: {path}")
        with np.load(path) as npz:
            return {key: npz[key] for key in npz.files}

    # ── навигация ────────────────────────────────────────────────────────────

    def exists(self, name: str, subdir: str = "") -> bool:
        """True если существует .npy ИЛИ .npz с таким именем."""
        return (self._resolve(name, subdir, ext=".npy").exists()
                or self._resolve(name, subdir, ext=".npz").exists())

    def list(self, subdir: str = "") -> list[str]:
        """Список всех .npy/.npz файлов в subdir (без расширения)."""
        directory = self._base / subdir if subdir else self._base
        if not directory.exists():
            return []
        names: set[str] = set()
        for f in directory.iterdir():
            if f.suffix in (".npy", ".npz"):
                names.add(f.stem)
        return sorted(names)

    # ── private ──────────────────────────────────────────────────────────────

    def _resolve(self, name: str, subdir: str, ext: str) -> Path:
        base = self._base / subdir if subdir else self._base
        return base / f"{name}{ext}"
