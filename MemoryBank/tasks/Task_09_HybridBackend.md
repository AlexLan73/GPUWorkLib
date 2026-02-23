# Task_09_HybridBackend — OPENCLandROCm гибридный режим (опционально)

> **Памятка для ИИ**: Если работаешь под Windows — пиши весь код + тесты. **Тестировать будем только под Linux** (Debian/Ubuntu). ROCm не поддерживается на Windows.

---

## 1. Цель

Реализовать гибридный режим `BackendType::OPENCLandROCm`: на одном GPU — OpenCL + ROCm. Рекомендация: **Вариант B** — два DrvGPU на одну GPU, ZeroCopyBridge связывает память. Альтернатива: **Вариант A** — HybridBackend обёртка.

---

## 2. Зависимости

- Task_00_DrvGPU
- Task_08_ZeroCopy (для передачи памяти между OpenCL и ROCm)

---

## 3. Вариант A: HybridBackend

```cpp
class HybridBackend : public IBackend {
    std::unique_ptr<OpenCLBackend> opencl_;
    std::unique_ptr<ROCmBackend> rocm_;
public:
    IBackend* GetOpenCL() { return opencl_.get(); }
    IBackend* GetROCm() { return rocm_.get(); }
    void* Allocate(size_t size_bytes, unsigned int flags) override;  // по умолчанию OpenCL с экспортом
};
```

- `CreateBackend()` для `OPENCLandROCm`: `backend_ = std::make_unique<HybridBackend>()`
- `HybridBackend::Initialize(device_index)` создаёт оба под-backend

---

## 4. Вариант B: Два DrvGPU

- `GPUManager::InitializeAll(OPENCLandROCm)` создаёт для каждой GPU два DrvGPU (или один с двумя backend-указателями)
- `GetOpenCLBackend()`, `GetROCmBackend()`
- ZeroCopyBridge связывает cl_mem и hip_ptr

---

## 5. Задачи

- [ ] Выбрать Вариант A или B
- [ ] Реализовать HybridBackend или логику двух DrvGPU
- [ ] Интеграция в GPUManager
- [ ] Тесты
- [ ] Компиляция

---

## Ссылки

- [PLAN_ROCm_DrvGPU_Full.md](PLAN_ROCm_DrvGPU_Full.md) — Часть 3
