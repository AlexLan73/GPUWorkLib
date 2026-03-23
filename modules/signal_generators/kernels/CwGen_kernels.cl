

struct float2_t {
    float x;
    float y;
};

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __global__ __launch_bounds__(256)
void generate_cw(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,
    const float amplitude,
    const float initial_phase)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = amplitude * cos_val;
    out.y = amplitude * sin_val;
    output[gid] = out;
}

extern "C" __global__ __launch_bounds__(256)
void generate_cw_real(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float base_freq,
    const float freq_step,
    const float amplitude,
    const float initial_phase)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float freq = base_freq + (float)beam_id * freq_step;
    const float t = (float)sample_id / sample_rate;
    const float phase = 2.0f * M_PI_F * freq * t + initial_phase;

    float2_t out;
    out.x = amplitude * __cosf(phase);
    out.y = 0.0f;
    output[gid] = out;
}

