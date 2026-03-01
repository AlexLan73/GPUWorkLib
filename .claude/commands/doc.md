---
description: Создать или обновить Doc/Modules/{module}/Full.md по эталону heterodyne
---

Используй агент module-doc-writer для модуля: $ARGUMENTS

Если модуль не указан — спроси какой документировать.

Алгоритм:
1. Прочитай Doc/Modules/heterodyne/Full.md как эталон формата
2. Изучи реальный код modules/$ARGUMENTS/ (include, src, kernels, tests)
3. Изучи python/py_$ARGUMENTS.hpp и Python_test/$ARGUMENTS/
4. Создай ДВА файла (оба обязательны!):
   - Doc/Modules/$ARGUMENTS/Full.md — полная документация
   - Doc/Modules/$ARGUMENTS/Quick.md — шпаргалка (образец: Doc/Modules/heterodyne/Quick.md)

   Full.md содержит:
   - Математикой алгоритма (LaTeX)
   - Pipeline ASCII + mermaid диаграммой
   - C4 диаграммами (C1-C4)
   - C++ и Python примерами вызова
   - Таблицей тестов (C++ + Python)
   - Файловым деревом модуля
   - Важными нюансами

После создания обнови MASTER_INDEX.md если нужно.
