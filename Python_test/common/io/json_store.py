"""
json_store.py — JsonStore (I/O для dict / TestResult)
=======================================================

SRP: только работа с JSON.

Структура:
    {base_dir}/{subdir}/{name}.json

Параметр add_timestamp=True добавляет "saved_at" ISO в начало файла.
Для стабильных diff между прогонами (CI / регрессии) — отключить.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .base import IDataStore


class JsonStore(IDataStore):
    """Хранилище JSON файлов."""

    def __init__(self,
                 base_dir: str | Path = "Results/JSON",
                 add_timestamp: bool = True):
        """
        Args:
            base_dir:      корень хранилища.
            add_timestamp: добавлять ли ``saved_at`` в payload.
        """
        self._base = Path(base_dir)
        self._add_ts = add_timestamp

    def save(self, data: dict, name: str, subdir: str = "") -> Path:
        """Сохранить dict в JSON."""
        path = self._resolve(name, subdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._add_ts:
            # Наш штамп — ПОСЛЕДНИЙ, чтобы он перезаписывал возможное
            # поле "saved_at" из ранее загруженного JSON (round-trip).
            payload: dict = {**data, "saved_at": datetime.now().isoformat()}
        else:
            payload = dict(data)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return path

    def load(self, name: str, subdir: str = "") -> dict:
        """Загрузить JSON файл → dict."""
        path = self._resolve(name, subdir)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self, name: str, subdir: str = "") -> bool:
        return self._resolve(name, subdir).exists()

    def list(self, subdir: str = "") -> list[str]:
        directory = self._base / subdir if subdir else self._base
        if not directory.exists():
            return []
        return sorted(f.stem for f in directory.iterdir() if f.suffix == ".json")

    def _resolve(self, name: str, subdir: str) -> Path:
        base = self._base / subdir if subdir else self._base
        return base / f"{name}.json"
