Рефактинг кода modules/fft_maxima/include/spectrum_maxima_finder.h 
 Перенеси эти class & struct  в папку modules/fft_maxima/include/interface  
  и настрой cMake  modules/fft_maxima/CMakeLists.txt
  исправь файл modules/fft_maxima/tests/cpu_fft_reference.hpp на использование общих structur & class из modules/fft_maxima/include/interface
  - structur & class:
      SpectrumParams
      MaxValue
    SpectrumResult
    ProfilingData
Используй sequential-thinking (MCP server).