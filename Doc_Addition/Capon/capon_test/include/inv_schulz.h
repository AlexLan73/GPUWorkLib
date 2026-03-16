#pragma once
#include <arrayfire.h>


//Выводит ошибку нахождения обратной матрицы
//A - изначальная матрица
//A1 - найденная обратная матрица
void print_shulz_error(af::array A,af::array A1);


// Обращение методом Шульца
//[in]
//A - входная матрица
//epsilon - точность
//max_iter - максимальное число итерации

//[out]
//iter_num - число итераций
//Возвращает обратную матрицу
af::array inv_shulz(af::array A,double epsiolon,int max_iter, int &iter_num);


// Функция Ряда Неймена
af::array matrix_INV_Neumann_batch(const af::array& A0, int order);



// Обращение методом Шульца c поддержкой batch
//[in]
//A - входная матрица
//epsilon - точность
//iter_num_max - максимальное число итерации

//[out]
//iter_num - число итераций
//epsilon_cur2 - массив квадратов ошибок на каждой итерации
//Возвращает обратную матрицу
af::array matrix_INV_Schulz_fast_batch( const af::array& A, double epsilon, int iter_num_max, int& iter_num,
                           std::vector<double>& epsilon_cur2);
