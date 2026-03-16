//#define NOMINMAX
#include <iostream>
#include "capon_relief.h"
#include <arrayfire.h>
#include "threshold_processing.h"
#include <ctime>


std::vector<af::array> threashold_keipon(af::array z, float h, af::array u0, af::array v0, float Nu, float Nv) {
	try {
        //af::timer t = af::timer::start();

		const dim_t N = Nu * Nv;
        int batch_x = z.dims(2);
        int batch_y = z.dims(3);

        af::array i = af::range(af::dim4(N), 0, f32);




		//af_print(i);

		af::array ix = af::floor(i / Nv);
		//af_print(ix);

		af::array iy = af::mod(i, Nv);
		//af_print(iy);

        //std::cout << "n_dims = " << i.dims() << std::endl;

		af::array cond0 = z > h;
		//af_print(cond0);









        af::array cond1 = z > z(0,(af::min)(iy + 1.0, Nv - 1.0) + ix * Nv, af::span,af::span);
		//af_print(cond1);
        af::array cond2 = z > z(0,(af::max)(iy - 1.0, 0.0) + ix * Nv, af::span,af::span);
        af::array cond3 = z > z(0,iy + (af::min)(ix + 1.0, Nu - 1.0) * Nv, af::span,af::span);
        af::array cond4 = z > z(0,iy + (af::max)(ix - 1.0, 0.0) * Nv, af::span,af::span);

		af::array cond = cond0 && cond1 && cond2 && cond3 && cond4;
		//af::array cond = cond0;

        //print_dims2(u0,"u=");

        af::array u = af::tile(u0,1,1,batch_x,batch_y);
        af::array v = af::tile(v0,1,1,batch_x,batch_y);





		af::array up = u(cond);
		af::array vp = v(cond);
		af::array zp = z(cond);


		

		std::vector<af::array> res = { up, vp, zp };


		return res;
	}
	catch (const af::exception& e) {
		std::cerr << "ArrayFire error: " << e.what() << std::endl;

		throw;
	}
}
