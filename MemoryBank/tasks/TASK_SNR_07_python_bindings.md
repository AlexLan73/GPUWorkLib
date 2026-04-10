# TASK SNR_07: Python bindings (pybind11) для SNR-estimator

> **Дата**: 2026-04-09
> **Модуль**: `modules/statistics/python/statistics_bindings.cpp`
> **Приоритет**: Medium
> **Статус**: BACKLOG
> **Зависимости**: **[SNR_06](TASK_SNR_06_facade.md)**
> **Ревьюер**: Кодо
>
> 📐 **План**: **Часть 3** в [snr_estimator_statistics_plan.md](../specs/snr_estimator_statistics_plan.md)
> 📋 **Индекс**: [`TASK_SNR_INDEX.md`](TASK_SNR_INDEX.md)

---

## 🎯 Цель

Добавить Python bindings (pybind11) для типов и методов SNR-estimator'а в существующий файл `statistics_bindings.cpp`. **Не создавать новый файл** — дополнить существующий.

---

## 📝 Экспортировать в Python

### Типы
- `SnrEstimationConfig` (struct → class с атрибутами)
- `SnrEstimationResult` (struct → class с атрибутами)
- `BranchThresholds` (struct → class)
- `BranchType` (enum: Low, Mid, High)
- `BranchSelector` (class с методами Select, Current, Reset)

### Методы `StatisticsProcessor`
- `compute_snr_db(data_np, n_antennas, n_samples, config)` — с CPU данными из numpy

> ⚠️ **GPU overload НЕ экспортируется** в Python (void* не маппится в Python естественно).

---

## 📝 Пример кода для bindings

```cpp
// В существующем statistics_bindings.cpp, в функции PYBIND11_MODULE:

void BindSnrEstimator(py::module_& m) {
  using namespace statistics;

  // --- Enum BranchType ---
  py::enum_<BranchType>(m, "BranchType")
    .value("Low",  BranchType::Low)
    .value("Mid",  BranchType::Mid)
    .value("High", BranchType::High)
    .export_values();

  // --- BranchThresholds ---
  py::class_<BranchThresholds>(m, "BranchThresholds")
    .def(py::init<>())
    .def_readwrite("low_to_mid_db",  &BranchThresholds::low_to_mid_db)
    .def_readwrite("mid_to_high_db", &BranchThresholds::mid_to_high_db)
    .def_readwrite("hysteresis_db",  &BranchThresholds::hysteresis_db);

  // --- SnrEstimationConfig ---
  py::class_<SnrEstimationConfig>(m, "SnrEstimationConfig")
    .def(py::init<>())
    .def_readwrite("target_n_fft",         &SnrEstimationConfig::target_n_fft)
    .def_readwrite("step_samples",         &SnrEstimationConfig::step_samples)
    .def_readwrite("step_antennas",        &SnrEstimationConfig::step_antennas)
    .def_readwrite("guard_bins",           &SnrEstimationConfig::guard_bins)
    .def_readwrite("ref_bins",             &SnrEstimationConfig::ref_bins)
    .def_readwrite("search_full_spectrum", &SnrEstimationConfig::search_full_spectrum)
    .def_readwrite("with_dechirp",         &SnrEstimationConfig::with_dechirp)
    .def_readwrite("thresholds",           &SnrEstimationConfig::thresholds)
    .def("validate", &SnrEstimationConfig::Validate);

  // --- SnrEstimationResult ---
  py::class_<SnrEstimationResult>(m, "SnrEstimationResult")
    .def(py::init<>())
    .def_readonly("snr_db_global",         &SnrEstimationResult::snr_db_global)
    .def_readonly("snr_db_per_antenna",    &SnrEstimationResult::snr_db_per_antenna)
    .def_readonly("used_antennas",         &SnrEstimationResult::used_antennas)
    .def_readonly("used_bins",             &SnrEstimationResult::used_bins)
    .def_readonly("actual_step_samples",   &SnrEstimationResult::actual_step_samples)
    .def_readonly("n_actual",              &SnrEstimationResult::n_actual);
    // БЕЗ branch — он в BranchSelector!

  // --- BranchSelector ---
  py::class_<BranchSelector>(m, "BranchSelector")
    .def(py::init<>())
    .def("select",  &BranchSelector::Select,
         py::arg("snr_db"), py::arg("thresholds"),
         "Select branch with hysteresis. Updates internal state.")
    .def("current", &BranchSelector::Current,
         "Get current branch without updating state")
    .def("reset",   &BranchSelector::Reset,
         py::arg("to") = BranchType::Low);

  // --- StatisticsProcessor method ---
  // Найти существующую py::class_<StatisticsProcessor> и добавить:
  // .def("compute_snr_db", ...)
  // См. ниже.
}
```

**Привязка метода `compute_snr_db`** в существующем `py::class_<StatisticsProcessor>`:

```cpp
.def("compute_snr_db",
     [](StatisticsProcessor& self,
        py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data_np,
        uint32_t n_antennas,
        uint32_t n_samples,
        const SnrEstimationConfig& config) {
       // Validate shape
       if (data_np.ndim() != 2) {
         throw std::invalid_argument("compute_snr_db: expected 2D array");
       }
       if ((uint32_t)data_np.shape(0) != n_antennas ||
           (uint32_t)data_np.shape(1) != n_samples) {
         throw std::invalid_argument(
             "compute_snr_db: shape mismatch");
       }

       // Convert to std::vector<complex<float>>
       auto buf = data_np.request();
       const std::complex<float>* ptr =
           static_cast<const std::complex<float>*>(buf.ptr);
       std::vector<std::complex<float>> data(ptr, ptr + data_np.size());

       return self.ComputeSnrDb(data, n_antennas, n_samples, config);
     },
     py::arg("data"), py::arg("n_antennas"),
     py::arg("n_samples"), py::arg("config"),
     "Compute SNR (dB) from numpy complex64 array via CA-CFAR.")
```

**В конце `PYBIND11_MODULE` вызвать:**
```cpp
BindSnrEstimator(m);
```

---

## ✅ Definition of Done

- [ ] Экспортированы: `BranchType`, `BranchThresholds`, `SnrEstimationConfig`, `SnrEstimationResult`, `BranchSelector`
- [ ] Метод `StatisticsProcessor.compute_snr_db(data_np, n_antennas, n_samples, config)` экспортирован
- [ ] `SnrEstimationResult` в Python **НЕ содержит** атрибут `branch` (он в BranchSelector)
- [ ] `compute_snr_db` принимает `numpy.ndarray` с `dtype=complex64`, shape `(n_antennas, n_samples)`
- [ ] Валидация shape внутри lambda (throw `std::invalid_argument`)
- [ ] Существующие bindings не изменены
- [ ] Код компилируется на Debian (понедельник)
- [ ] Кодо провёл ревью

---

## ⚠️ Критерии ревью (Кодо проверит)

- ✅ Имена атрибутов в Python: `target_n_fft`, `search_full_spectrum`, `n_actual` (snake_case lowercase)
- ✅ `SnrEstimationResult` без `branch` атрибута
- ✅ `BranchSelector.select(snr_db, thresholds)` — обновляет state
- ✅ `BranchSelector.current()` — только чтение
- ✅ `BranchSelector.reset(to=BranchType.Low)` — с default
- ✅ `compute_snr_db` шейп-валидация + понятное исключение при несоответствии
- ✅ numpy `dtype=complex64` обязателен (не `complex128`!)
- ✅ GPU overload НЕ экспортируется

---

## 🚫 Запреты

- ❌ НЕ создавать новый файл — дополнять `statistics_bindings.cpp`
- ❌ НЕ экспортировать GPU overload `ComputeSnrDb(void* gpu_data, ...)` в Python
- ❌ НЕ менять существующие bindings `ComputeMean`/`ComputeMedian`/etc
- ❌ НЕ добавлять `BranchType` в `SnrEstimationResult`

---

## 🔗 Связанные таски

- **Требует:** [SNR_06](TASK_SNR_06_facade.md) (методы фасада)
- **Блокирует:** [SNR_10](TASK_SNR_10_python_e2e.md) (Python e2e тест)

---

*Created 2026-04-09 | Кодо*
