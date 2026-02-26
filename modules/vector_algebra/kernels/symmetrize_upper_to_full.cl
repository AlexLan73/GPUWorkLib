

// ============================================================================
// symmetrize_upper_to_full — заполнить нижний треугольник (conjugate)
// ============================================================================
// float2 = complex<float>: .x = real, .y = imag
// conjugate: {v.x, -v.y}
//
// Запускать с grid (ceil(n/16), ceil(n/16)), block (16, 16).
// Каждый поток обрабатывает один элемент (row, col).
// Если col > row → записать conj в (col, row).

extern "C" __global__ void symmetrize_upper_to_full(
    float2* __restrict__ data,
    unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    // Только верхний треугольник: col > row
    if (col > row) {
        float2 v = data[row * n + col];
        data[col * n + row] = {v.x, -v.y};  // conjugate
    }
}

