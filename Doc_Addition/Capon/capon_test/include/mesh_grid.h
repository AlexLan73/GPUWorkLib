#pragma once
#include <iostream>
#include <arrayfire.h>

std::pair<af::array, af::array> mesh_grid(af::array u0, af::array v0);
