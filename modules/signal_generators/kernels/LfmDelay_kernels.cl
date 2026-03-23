

struct float2_t {
    float x;
    float y;
};

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __global__ __launch_bounds__(256)
void generate_lfm(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float t = (float)sample_id / sample_rate;
    const float phase = M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t;

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = amplitude * cos_val;
    out.y = amplitude * sin_val;
    output[gid] = out;
}

extern "C" __global__ __launch_bounds__(256)
void generate_lfm_real(
    float2_t* __restrict__ output,
    const unsigned int beam_count,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int beam_id   = blockIdx.y;
    if (sample_id >= n_point || beam_id >= beam_count) return;

    const unsigned int gid = beam_id * n_point + sample_id;

    const float t = (float)sample_id / sample_rate;
    const float phase = M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t;

    float2_t out;
    out.x = amplitude * __cosf(phase);
    out.y = 0.0f;
    output[gid] = out;
}

// Conjugate LFM: s_ref*(t) = exp(-j[pi*mu*t^2 + 2*pi*f_start*t])
// Used as reference for dechirp: s_dc = s_rx * s_ref*
// Difference from generate_lfm: negative phase sign (conjugate)
extern "C" __global__ __launch_bounds__(256)
void generate_lfm_conjugate(
    float2_t* __restrict__ output,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_point) return;

    const float t = (float)i / sample_rate;
    // Negative phase → conjugate of LFM
    const float phase = -(M_PI_F * chirp_rate * t * t + 2.0f * M_PI_F * f_start * t);

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = cos_val;  // amplitude = 1.0 for reference signal
    out.y = sin_val;
    output[i] = out;
}

// LFM with analytical per-antenna delay:
//   s(t) = A * exp(j * [pi*mu*(t-tau)^2 + 2*pi*f_start*(t-tau)])
//   output = 0 when t < tau (signal hasn't arrived yet)
// 2D grid: (n_point, antennas)
extern "C" __global__ __launch_bounds__(256)
void generate_lfm_analytical_delay(
    float2_t* __restrict__ output,
    const float* __restrict__ delay_us,
    const unsigned int n_ant,
    const unsigned int n_point,
    const float sample_rate,
    const float f_start,
    const float chirp_rate,
    const float amplitude)
{
    const unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int ant_id    = blockIdx.y;
    if (sample_id >= n_point || ant_id >= n_ant) return;

    const unsigned int gid = ant_id * n_point + sample_id;
    const float t = (float)sample_id / sample_rate;
    const float tau = delay_us[ant_id] * 1e-6f;

    if (t < tau) {
        float2_t zero; zero.x = 0.0f; zero.y = 0.0f;
        output[gid] = zero;
        return;
    }

    const float t_local = t - tau;
    const float phase = M_PI_F * chirp_rate * t_local * t_local
                      + 2.0f * M_PI_F * f_start * t_local;

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    float2_t out;
    out.x = amplitude * cos_val;
    out.y = amplitude * sin_val;
    output[gid] = out;
}

