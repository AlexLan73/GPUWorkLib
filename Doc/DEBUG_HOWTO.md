# Как отлаживать GPUWorkLib

## Способ 1: В Cursor (если есть cppvsdbg)

1. Поставь **точки останова**: клик слева от номера строки в `main.cpp` или другом файле (или F9).
2. Нажми **F5** или выбери **Run → Start Debugging**.
3. Если Cursor пишет, что тип `cppvsdbg` не поддерживается — нажми **«Install cppvsdbg Extension»** в диалоге, дождись установки и снова нажми F5.

Путь к exe в конфиге: `build/Debug/GPUWorkLib.exe`.  
Если у тебя exe лежит в `build/src/Debug/GPUWorkLib.exe` — поправь в `.vscode/launch.json` поле `"program"`.

---

## Способ 2: В Visual Studio (всегда работает для MSVC)

1. Собери проект в Cursor/терминале:
   ```bash
   cmake --build build --config Debug
   ```
2. Открой в **Visual Studio** решение:
   ```
   build\GPUWorkLib.sln
   ```
3. В обозревателе решений выбери проект **GPUWorkLib** (или **src**) как запускаемый (ПКМ → Set as Startup Project).
4. Поставь точки останова (F9), нажми **F5** — запустится отладчик MSVC.

Так отлаживать можно всегда, независимо от поддержки cppvsdbg в Cursor.
