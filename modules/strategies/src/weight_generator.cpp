#include "weight_generator.hpp"
#include "interface/i_backend.hpp"

#include <cmath>

namespace strategies {

std::vector<std::complex<float>> WeightGenerator::generate_delay_and_sum(
    const WeightParams& params)
{
  const uint32_t n = params.n_ant;
  const double f0 = params.f0;
  const double norm = 1.0 / std::sqrt(static_cast<double>(n));
  constexpr double two_pi = 2.0 * 3.14159265358979323846;

  std::vector<std::complex<float>> W(n * n);

  for (uint32_t beam = 0; beam < n; ++beam) {
    for (uint32_t ant = 0; ant < n; ++ant) {
      double tau_ant = params.tau_base + ant * params.tau_step;
      double phase = -two_pi * f0 * tau_ant;
      W[beam * n + ant] = std::complex<float>(
          static_cast<float>(norm * std::cos(phase)),
          static_cast<float>(norm * std::sin(phase)));
    }
  }

  return W;
}

void* WeightGenerator::upload_to_gpu(
    void* backend_ptr,
    const std::vector<std::complex<float>>& weights)
{
  auto* backend = static_cast<drv_gpu_lib::IBackend*>(backend_ptr);
  size_t bytes = weights.size() * sizeof(std::complex<float>);
  void* d_W = backend->Allocate(bytes);
  backend->MemcpyHostToDevice(d_W, weights.data(), bytes);
  return d_W;
}

}  // namespace strategies
