#include <iostream>
#include <arrayfire.h>
#include "mesh_grid.h"

using namespace af;

std::pair<af::array, af::array> mesh_grid(af::array u0, af::array v0) {
    const dim_t Nx = u0.dims(0);
    const dim_t Ny = v0.dims(0);

    const dim_t N = Nx * Ny;

    array n = af::range(af::dim4(N));

 

    array nx = floor(n / Ny);
    array ny = mod(n, Ny);

    array u = u0(nx);
    array v = v0(ny);

	return std::make_pair(u, v);
}
