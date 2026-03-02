Цель: Создать агента и комманду для тестирования.

**Реализовано** (2026-03-02):
- Cursor Agent: `.cursor/skills/run-gpu-tests/SKILL.md`
- Команда: `./scripts/run_agent_tests.sh [all | <module> | --file <path>]`
- config: `config/tests_order.txt`
- его дйствия 
1. смотрит с какой gpu работаем 
    если с стоит AMD GPU то не тестируем все что имеет вызов библиотеки clFFT 
      (может прописать в cmake отключить clFFT  если стоит AMD GPU)
    если стоит NVIDIA то не тестируем все что всязано с ROCm

2.  задавать что тестируем 
  2.1. all все тесты знзачит все модули и все тесты относыщиеся к модулю (модули) порядок очень важен и мы пропишим отдельно
  2.2. задать конкретный модуль fft_processor, fft_maxima, ... порядок важен и обязателен
  2.3. задать что рестировать через файл - с соблюдением порядка

3. Порядок тестирования если all
  3.0. DrvGPU. 
  3.1. modules/fft_processor
  3.2. modules/statistics
  3.3. modules/vector_algebra
  3.4. modules/fft_maxima
  3.5. modules/filters
  3.6. modules/signal_generators
  3.7. modules/lch_farrow
  3.8. modules/heterodyne

4. Тест делать на c++ & python. Во время теста все сохранять таблицы профилирования и сформированные картинки (python)  

