### У тебя должнополучиться получиться в API два метода
1. то что ты наверно сделал
  - передается данные  и опмсание в шавблоне и кертнел паралельно (сразу) ищет максимумы
2. также передаются даннве и там реализуетсч алгоритм как в   
  // НОВЫЙ API: Process(InputData, PeakSearchMode, DriverType)
        auto gpu_vector = finder.Process(input, peak_mode, DriverType::OPENCL);
описание в 
E:\C++\GPUWorkLib\Doc\Modules\fft_maxima\README.md
E:\C++\GPUWorkLib\Doc\Modules\fft_maxima\
посмотри алгоритм формирования данных для FFT в Process(...)
посмотри кернал там тоже писание 
E:\C++\GPUWorkLib\modules\fft_maxima\include\kernels\fft_kernel_sources.hpp
напоминаю приходят данные по всем лучам 1000 точек приводим их к =>2^n 1024 
затем добавляем нули 1024+1024 = 2048 затеи есть коеф. k= 2, 4, 8 - добавить нулей до числа 2048*k
от этого берем FFT потом вызываем кернел с поиском 1/2 максим или всех. если все до вызывакернел одинаково может нет смысло аовторять использовать одну шапку

3. вызов kernal из FFT я это делал вот рабочий пример 
E:\C++\Cuda\OpenCLProd\LOpenCl - это каталог там все
E:\C++\Cuda\OpenCLProd\LOpenCl\src\Native3Dcustom.cpp
в этом примере из fft два calback посмотри 

### Используй sequential-thinking, context7, github,