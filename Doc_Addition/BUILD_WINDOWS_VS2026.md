# Сборка GPUWorkLib на Windows (ветка nvidia)

Ninja в `tools/ninja.exe`. Если нет — скачать [ninja-win.zip](https://github.com/ninja-build/ninja/releases) и распаковать в `tools/`. Один пресет `windows-nvidia`.

**Запускать из «x64 Native Tools Command Prompt for VS 2026»:**

```cmd
cd E:\C++\GPUWorkLib
rmdir /s /q build
cmake --preset windows-nvidia
cmake --build build
```

Debug: при configure добавить `-DCMAKE_BUILD_TYPE=Debug`.
