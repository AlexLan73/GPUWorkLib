Продумать как принудительно сохранить ихформацию при сбои Error

Для глобального перехвата критических сбоев (Access Violation, ошибки GPU, деление на ноль) без использования `try-catch` в каждой функции, тебе нужно работать на уровне операционной системы. 

В C++ есть два основных механизма для этого: **Сигналы** (для Linux/кроссплатформ) и **Structured Exception Handling** (SEH, для Windows).

### 1. Windows (MSVC): Vectorized Exception Handling (VEH)
Это самый мощный способ для Windows. VEH перехватывает исключение *до того*, как система начнет искать блоки `try-catch`.

```cpp
#include <windows.h>
#include <iostream>
#include <fstream>

// Обработчик, который вызовется при любом падении
LONG WINAPI GlobalErrorHandler(PEXCEPTION_POINTERS pExceptionInfo) {
    std::ofstream log("crash_log.txt", std::ios::app);
    
    // Код ошибки (например, 0xC0000005 — Access Violation)
    DWORD code = pExceptionInfo->ExceptionRecord->ExceptionCode;
    // Адрес инструкции, где произошел сбой
    PVOID address = pExceptionInfo->ExceptionRecord->ExceptionAddress;

    log << "CRASH DETECTED!" << std::endl;
    log << "Error Code: 0x" << std::hex << code << std::dec << std::endl;
    log << "Address: " << address << std::endl;

    // Здесь можно вызвать StackWalker или аналоги для записи стека вызовов
    log.close();

    // EXCEPTION_CONTINUE_SEARCH — передать управление дальше (программа упадет)
    // EXCEPTION_EXECUTE_HANDLER — "проглотить" ошибку (опасно)
    return EXCEPTION_CONTINUE_SEARCH; 
}

int main() {
    // Регистрируем глобальный перехватчик (1 = вызывать первым)
    AddVectoredExceptionHandler(1, GlobalErrorHandler);

    // Пример ошибки: обращение по кривому адресу (имитация сбоя GPU драйвера)
    int* p = nullptr;
    *p = 10; 

    return 0;
}
```

### 2. Linux / Cross-platform: Signals
В Linux (и Windows частично) ошибки процессора и памяти прилетают в виде сигналов (`SIGSEGV`, `SIGFPE`, `SIGILL`).

```cpp
#include <csignal>
#include <iostream>
#include <execinfo.h> // Только для Linux/macOS

void signalHandler(int signum) {
    std::cerr << "Signal received: " << signum << std::endl;

    // Получаем стек вызовов (backtrace)
    void* array [oroboro](https://oroboro.com/stack-trace-on-crash/);
    size_t size = backtrace(array, 10);
    backtrace_symbols_fd(array, size, 2); // Пишет в stderr

    exit(signum);
}

int main() {
    std::signal(SIGSEGV, signalHandler); // Ошибка сегментации
    std::signal(SIGFPE,  signalHandler); // Арифметика (деление на 0)
    

