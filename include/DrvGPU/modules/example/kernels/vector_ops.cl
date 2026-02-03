/**
 * @file vector_ops.cl
 * @brief OpenCL kernels для примитивных операций с векторами
 * 
 * Реализует базовые операции:
 * - Добавление скаляра (A[] + 1)
 * - Вычитание скаляра (A[] - 1)
 * - Сложение двух векторов (A[] + B[])
 * 
 * Два режима:
 * 1. Out-of-place: результат записывается в новый вектор C[]
 * 2. In-place: результат перезаписывает входной вектор A[]
 * 
 * @author DrvGPU Team
 * @date 2026-02-03
 */

// ════════════════════════════════════════════════════════════════════════════
// 1. Добавление скаляра к вектору
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Добавить 1 к каждому элементу вектора (out-of-place)
 * 
 * C[i] = A[i] + 1
 * 
 * @param input  Входной вектор A[]
 * @param output Выходной вектор C[]
 * @param n      Размер вектора
 */
__kernel void vector_add_one_out(
    __global const float* input,
    __global float* output,
    const int n)
{
    int gid = get_global_id(0);
    
    if (gid < n) {
        output[gid] = input[gid] + 1.0f;
    }
}

/**
 * @brief Добавить 1 к каждому элементу вектора (in-place)
 * 
 * A[i] = A[i] + 1
 * 
 * @param data Вектор для модификации A[]
 * @param n    Размер вектора
 */
__kernel void vector_add_one_inplace(
    __global float* data,
    const int n)
{
    int gid = get_global_id(0);
    
    if (gid < n) {
        data[gid] = data[gid] + 1.0f;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Вычитание скаляра из вектора
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Вычесть 1 из каждого элемента вектора (out-of-place)
 * 
 * C[i] = A[i] - 1
 * 
 * @param input  Входной вектор A[]
 * @param output Выходной вектор C[]
 * @param n      Размер вектора
 */
__kernel void vector_sub_one_out(
    __global const float* input,
    __global float* output,
    const int n)
{
    int gid = get_global_id(0);
    
    if (gid < n) {
        output[gid] = input[gid] - 1.0f;
    }
}

/**
 * @brief Вычесть 1 из каждого элемента вектора (in-place)
 * 
 * A[i] = A[i] - 1
 * 
 * @param data Вектор для модификации A[]
 * @param n    Размер вектора
 */
__kernel void vector_sub_one_inplace(
    __global float* data,
    const int n)
{
    int gid = get_global_id(0);
    
    if (gid < n) {
        data[gid] = data[gid] - 1.0f;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Сложение двух векторов
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Сложить два вектора (out-of-place)
 * 
 * C[i] = A[i] + B[i]
 * 
 * @param input_a Первый входной вектор A[]
 * @param input_b Второй входной вектор B[]
 * @param output  Выходной вектор C[]
 * @param n       Размер векторов
 */
__kernel void vector_add_vectors_out(
    __global const float* input_a,
    __global const float* input_b,
    __global float* output,
    const int n)
{
    int gid = get_global_id(0);
    
    if (gid < n) {
        output[gid] = input_a[gid] + input_b[gid];
    }
}

/**
 * @brief Сложить два вектора (in-place)
 * 
 * A[i] = A[i] + B[i]
 * 
 * @param data_a  Первый вектор A[] (будет модифицирован)
 * @param input_b Второй входной вектор B[]
 * @param n       Размер векторов
 */
__kernel void vector_add_vectors_inplace(
    __global float* data_a,
    __global const float* input_b,
    const int n)
{
    int gid = get_global_id(0);
    
    if (gid < n) {
        data_a[gid] = data_a[gid] + input_b[gid];
    }
}
