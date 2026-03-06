<!-- ДАЛЬШЕ ПЕРЕНОСИТЬ: Doc_Addition/Info_Double_Buffering.md -->

# Double Buffering для GPU Pipeline

> Источник: MemoryBank/specs/Precpectiva/double_buffering_analysis.md  
> Дата: 2026-02-11 | Статус: ИССЛЕДОВАНИЕ (не реализовано)

---

## Концепция

Скрытие задержки Upload за временем FFT+Post. Без DB: Upload → FFT → Download по очереди. С DB: Upload B идёт параллельно с FFT A.

---

## Вывод по GPUWorkLib

- Upload уже ~12% от GPU времени (Pinned Memory).
- Для 2× буферов нужно 12+ GB — на 12 GB карте не поместится.
- Экономия порядка 0.5%; сложность высокая (два плана clFFT, синхронизация).
- **Рекомендация**: не реализовывать сейчас. Рассмотреть при росте доли Upload или при появлении памяти.

Полный текст с метриками и условиями «когда DB нужен» — в исходном файле.
