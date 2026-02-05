
🔧 Kernels → Кто вызывает

| Kernel (функция в hpp) | Вызывается из | Функция | Строка |

|------------------------|---------------|---------|--------|

| GetPaddingKernelSource() | antenna_fft_debug.cpp | CreatePaddingKernel() | :525 |

| GetDebugPostKernelSource() | antenna_fft_debug.cpp | CreatePostKernel() | :552 |

| GetPreCallbackSource32() | antenna_fft_release.cpp | CreateFFTPlanWithCallbacks() | :225 |

| GetPostCallbackSource() | antenna_fft_release.cpp | CreateFFTPlanWithCallbacks() | :234 |

| GetPostKernelSource() | — | — | ❌ НЕ ИСПОЛЬЗУЕТСЯ |

| GetPreCallbackSource() | — | — | ❌ НЕ ИСПОЛЬЗУЕТСЯ |


