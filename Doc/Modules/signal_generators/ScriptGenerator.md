# ScriptGenerator — Text DSL -> OpenCL Kernel Compiler

## Обзор

`ScriptGenerator` позволяет задавать формулы сигналов в текстовом формате (DSL), которые компилируются в OpenCL kernel и выполняются на GPU в runtime.

**Namespace**: `signal_gen`
**Файл**: `include/generators/script_generator.hpp`

---

## Формат скрипта

```
[Params]
ANTENNAS = 256       # Количество антенн/лучей (ID: 0..255)
POINTS = 10000       # Количество отсчётов (T: 0..9999)

[Defs]
// Промежуточные переменные (зависят от ID антенны)
var_A = 1.0 + (float)ID * 0.01
var_W = 0.1 + (float)ID * 0.0005
var_P = (float)ID * 3.14 / 180.0

[Signal]
// Финальный сигнал (зависит от T — номера отсчёта)
res = var_A * sin(var_W * (float)T + var_P);
```

### Секции

| Секция | Обязательная | Описание |
|--------|-------------|----------|
| `[Params]` | Да | Размеры: ANTENNAS/BEAMS, POINTS/LENGTH/SAMPLES |
| `[Defs]` | Нет | Промежуточные переменные (OpenCL C syntax) |
| `[Signal]` | Да | Вычисление выходного сигнала |

### Встроенные переменные

| Переменная | Тип | Описание |
|------------|-----|----------|
| `ID` | `uint` | Индекс антенны (0 .. ANTENNAS-1) |
| `T` | `uint` | Индекс отсчёта (0 .. POINTS-1) |
| `M_PI_F` | `float` | Константа pi |

### Выходные переменные

| Формат | Переменные | Результат |
|--------|-----------|-----------|
| **Real** | `res` | `output[gid] = (float2)(res, 0.0f)` |
| **Complex IQ** | `res_re`, `res_im` | `output[gid] = (float2)(res_re, res_im)` |
| **Real only re** | `res_re` | `output[gid] = (float2)(res_re, 0.0f)` |

### Автоматические преобразования

- Переменным без типа автоматически добавляется `float`
- Строкам без `;` добавляется точка с запятой
- Комментарии: `//` и `#`
- Операторы сравнения (`==`, `!=`, `<=`, `>=`) корректно обрабатываются

---

## API

```cpp
class ScriptGenerator {
public:
    explicit ScriptGenerator(IBackend* backend);

    // Загрузка и компиляция
    void LoadScript(const std::string& script_text);
    void LoadFile(const std::string& file_path);

    // Генерация
    cl_mem Generate();                                    // GPU buffer
    std::vector<std::complex<float>> GenerateToCpu();    // CPU vector

    // Информация
    uint32_t GetAntennas() const;
    uint32_t GetPoints() const;
    size_t   GetTotalSamples() const;
    const std::string& GetKernelSource() const;  // для отладки
    bool IsReady() const;
};
```

---

## Примеры

### 1. Простой CW сигнал

```
[Params]
ANTENNAS = 256
POINTS = 10000

[Defs]
var_A = 1.0 + (float)ID * 0.01
var_W = 0.1 + (float)ID * 0.0005
var_P = (float)ID * 3.14 / 180.0

[Signal]
res = var_A * sin(var_W * (float)T + var_P)
```

### 2. Комплексный IQ сигнал

```
[Params]
ANTENNAS = 8
POINTS = 4096

[Defs]
float freq = 0.05f + (float)ID * 0.01f
float phase = (float)ID * 0.785f

[Signal]
float angle = 2.0f * M_PI_F * freq * (float)T + phase
res_re = cos(angle)
res_im = sin(angle)
```

### 3. Условный сигнал (тернарный оператор)

```
[Params]
ANTENNAS = 16
POINTS = 2048

[Defs]
float freq = (ID % 2 == 0) ? 0.1f : 0.05f
float amp = 1.0f / (1.0f + (float)ID)

[Signal]
res = amp * sin(freq * (float)T)
```

### 4. C++ использование

```cpp
signal_gen::ScriptGenerator gen(backend);

gen.LoadScript(R"(
    [Params]
    ANTENNAS = 8
    POINTS = 4096
    [Signal]
    res = sin(0.1f * (float)T + (float)ID * 0.5f)
)");

// GPU генерация
cl_mem gpu_buf = gen.Generate();
// ... передать в FFT или другую обработку ...
clReleaseMemObject(gpu_buf);

// CPU генерация
auto cpu_data = gen.GenerateToCpu();
// cpu_data.size() == 8 * 4096
```

### 5. Python использование

```python
import gpuworklib

ctx = gpuworklib.GPUContext(0)
sg = gpuworklib.ScriptGenerator(ctx)

sg.load("""
[Params]
ANTENNAS = 16
POINTS = 4096
[Signal]
res = sin(0.1f * (float)T + (float)ID * 0.5f)
""")

data = sg.generate()  # numpy array (16, 4096), dtype=complex64
print(data.shape)     # (16, 4096)
print(sg.kernel_source)  # посмотреть сгенерированный OpenCL kernel
```

---

## Pipeline: ScriptGenerator + FFTProcessor

```python
# Генерация сигнала
sg = gpuworklib.ScriptGenerator(ctx)
sg.load("""
[Params]
ANTENNAS = 16
POINTS = 4096
[Defs]
float freq = 50.0f + (float)ID * 25.0f
[Signal]
float angle = 2.0f * M_PI_F * freq / 1000.0f * (float)T
res_re = cos(angle)
res_im = sin(angle)
""")
signal = sg.generate()

# GPU FFT
fft = gpuworklib.FFTProcessor(ctx)
spectrum = fft.process(signal, sample_rate=1000.0, output_mode='complex')
```

---

## Как это работает внутри

1. **Parse**: Текст разбирается на `[Params]`, `[Defs]`, `[Signal]`
2. **Transform**: Выражения преобразуются (добавление `float`, `;`)
3. **Generate**: Создаётся OpenCL kernel source с `get_global_id(0)` → `ID`, `T`
4. **Compile**: `clCreateProgramWithSource` + `clBuildProgram`
5. **Execute**: `clEnqueueNDRangeKernel` с `global_size = ANTENNAS * POINTS`

При ошибке компиляции выбрасывается `std::runtime_error` с:
- Сгенерированным kernel source
- Build log от OpenCL компилятора

---

## Производительность

| Конфигурация | Время | Throughput |
|-------------|-------|------------|
| 256 x 10000 = 2.56M samples | ~8 ms | 320 Msamples/s |
| 512 x 16384 = 8.39M samples | ~32 ms | 262 Msamples/s |

*Тестировано на NVIDIA GeForce RTX 2080 Ti*

---

*Обновлено: 2026-02-13*
