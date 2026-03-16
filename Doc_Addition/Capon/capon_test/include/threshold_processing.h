#pragma once
#include <arrayfire.h>

std::vector<af::array> threashold_keipon(af::array z, float h, af::array u, af::array v, float Nu, float Nv);
