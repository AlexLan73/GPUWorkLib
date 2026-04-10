"""
common.io — унифицированный I/O слой тестов
=============================================

Единая точка сохранения/загрузки всех артефактов тестов.

    from common.io import ResultStore

    store = ResultStore()
    store.save_array(gpu_out, "cw_4096", module="signal_generators")
    store.save_comparison(gpu_out, ref, "cw_vs_numpy", module="signal_generators")
    store.save_test_result(test_result, module="signal_generators")
    store.save_benchmark({"ms_per_call": 1.5}, "fft", module="fft_processor")
"""

from .base import IDataStore
from .json_store import JsonStore
from .numpy_store import NumpyStore
from .result_store import ResultStore

__all__ = [
    "IDataStore",
    "NumpyStore",
    "JsonStore",
    "ResultStore",
]
