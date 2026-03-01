---
description: Аудит модуля GPUWorkLib на соответствие эталону vector_algebra
---

Используй агент module-auditor для аудита модуля: $ARGUMENTS

Если модуль не указан — спроси какой модуль проверять.

Проверь:
- modules/$ARGUMENTS/ — C++ production и тесты
- python/py_$ARGUMENTS.hpp — Python binding
- Python_test/$ARGUMENTS/ — Python тесты
- Doc/Modules/$ARGUMENTS/ и Doc/Python/ — документация
- MASTER_INDEX.md и CLAUDE.md — актуальность статуса

Выдай таблицу расхождений с приоритетами и список конкретных задач.
