# Signal Generators — Полная документация

> Генерация сигналов на GPU (CW, LFM, Noise, Script, FormSignal)

**Namespace**: `signal_gen`
**Каталог**: `modules/signal_generators/`
**Зависимости**: DrvGPU (`IBackend*`), OpenCL

---

## Обзор

| Генератор | Класс | Описание |
|-----------|-------|----------|
| CW | `CwGenerator` | `s(t) = A*exp(j*(2πf*t + φ))` |
| LFM | `LfmGenerator` | ЛЧМ chirp |
| Noise | `NoiseGenerator` | Gaussian (Philox + Box-Muller) |
| Script | `ScriptGenerator` | DSL → OpenCL kernel |
| FormSignal | `FormSignalGenerator` | Мультиканальный, getX формула |
| DelayedFormSignal | `DelayedFormSignalGenerator` | FormSignal + Farrow задержка |

---

## Фабрика и сервис

- `SignalGeneratorFactory` — создание генераторов
- `SignalService` — фасад (GenerateCpu, GenerateGpu)

---

## Файлы

- [README](README.md) | [API](API.md) | [ScriptGenerator](ScriptGenerator.md)
- Python: [signal_generators_api.md](../../Python/signal_generators_api.md)

---

*Обновлено: 2026-02-17*
