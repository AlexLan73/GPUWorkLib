// OpenCL_.cpp: определяет точку входа для приложения.
//

/*
#ifdef _DEBUG
	#undef _DEBUG
	#include <python.h>
	#define _DEBUG
#else
    #include <python>
#endif
*/

#define _USE_MATH_DEFINES


#include <arrayfire.h>
#include <complex>
#include <random>
#include <fstream>
#include <math.h>
#include <cmath>
#include "capon_relief.h"
#include "threshold_processing.h"
#include "mesh_grid.h"
//#include "matplotlibcpp.h"
#include <ctime>
#include <regex>
#include <cctype>
#include <algorithm>
#include <iostream>
#include "inv_schulz.h"


//#include <windows.h>

using namespace std;
using namespace af;
//namespace plt = matplotlibcpp;

//1. Графика
//2. Посмотреть как работает на стенде
//3. Встраивать программу

static af::array linspace(float start, float end, int n) {
	//vector<double> res;

	af::array res;

	/*
	if (n == 0) {
		res =  af::array(res.size(), res.data());
	}
	*/

	if (n == 1) {
		res =  end;
	}
	else {
		
		/*
		double step = (end - start) / (n - 1);

		for (int i = 0; i < n; ++i) {
			res.push_back(start + step * i);
		}
		*/

        af::array k = af::transpose(af::range(dim4(n), f32));
		
		res = start + (end - start) * k/(n-1);

	}

	return res;
}

std::pair<std::vector<double>, std::vector<double>> parseComplex(const std::string& filename) {
	std::vector<double> real;
	std::vector<double> img;

	std::ifstream file(filename);

	if (!file.is_open()) {
		throw std::runtime_error("Cannot open file: " + filename);
	}

	std::string line;

	while (std::getline(file, line)) {
		std::stringstream ss(line);

		double real_num;
		std::string imag_str;

		while (ss >> real_num >> imag_str) {
			if (imag_str.back() != 'i') {
				throw std::invalid_argument("Invalid imag format");
			}

			imag_str.pop_back();

			double imag = std::stod(imag_str);

			real.push_back(real_num);
			img.push_back(imag);
		}
	}

	file.close();

	return std::make_pair(real, img);
}

int main(int argc, char* argv[])
{

	try {
		
        int device_default = 4;

        int device = argc > 1 ? atoi(argv[1]) : device_default;
        setDevice(device);

		af::info();

		string file_x = "x_data.txt";
		string file_y = "y_data.txt";

		ifstream inputFile_x(file_x);
		ifstream inputFile_y(file_y);

		if (!inputFile_x.is_open()) {
			std::cerr << "Error: Could not open file " << file_x << std::endl;
			return 1;
		}

		if (!inputFile_y.is_open()) {
			std::cerr << "Error: Could not open file " << file_y << std::endl;
			return 1;
		}

        vector<float> x_values;
        vector<float> y_values;

		double num;

		while (inputFile_x >> num) {
			x_values.push_back(num);
		}

		inputFile_x.close();

        //cout << "x_values size: " << x_values.size() << endl;

		while (inputFile_y >> num) {
			y_values.push_back(num);
		}

		inputFile_y.close();

        //cout << "y_values size: " << y_values.size() << endl;



		double f0 = 3918e+06 + 3.15e+06;
		double Fc = f0;

		double fc = 1e+06;
		double fs = 9.5e+06;

         //std::vector<float> ui_vector = {0.0};
         //std::vector<float> vi_vector = {0.0};
        // std::vector<float> Pi_log_vector = {-50.0};

         float Pi_log = 30.0;


         af::array ui, vi;
		//координаты помехи
        //af::array ui(ui_vector.size(),ui_vector.data());
        //af::array vi(vi_vector.size(),vi_vector.data());
        //af::array Pi_log(Pi_log_vector.size(),Pi_log_vector.data());



        bool timeClacFlag = true;

		//600
		//1200
		//3000
		//6000
		unsigned int N = 1000;   // Число временных отсчетов





        short typeCalc = 4;    //Алгоритм обращения 0  - linsolve, 1 - linsolve + chol, 2 - inv, 3 - итерации Шульца, 4 - chol + inv,  5 - модерн Шульц, 6 - спектральное разложение
        int iter_num = 12;            //число итераций Шульца



        int P = 85;   //Число каналов

        double u_step = 0.0078 / 2.5;
        double ulim = std::sin((6.5 / 2.0) * (M_PI / 180.0));

        int batch_x(1), batch_y(1);
        int direction_count(1);

        int batch_adapt_x(2),batch_adapt_y(2);
        int direction_count_adapt(4);










        std::cout << "Число каналов: ";
        std::cin >> P;


        std::cout << "Число отсчетов: ";
        std::cin >> N;



        std::cout << "Алгоритм обращения матрицы 0  - linsolve, 1 - linsolve + chol," << std::endl << " 2 - inv,  3 - итерации Шульца, 4 - chol + inv, 5 - модерн Шульц, 6 - спектральное разложение:  ";
        std::cin >> typeCalc;


        int kk;
        std::cout << "Расчет времени всей функции или ее частей (1 - всей, 0 - частей): ";
        std::cin >> kk;

        timeClacFlag = kk == 0;


        if(!timeClacFlag)
        {
        if((typeCalc == 5 )||(typeCalc == 3 ))
        {
            std::cout << "Колличество направлений по x: ";
            std::cin >> batch_x;
            std::cout << "Колличество направлений по y: ";
            std::cin >> batch_y;

        }else{

            std::cout << "Всего колличество направлений: ";
            std::cin >> direction_count;
        }
        }




        std::cout << "Способ расчета: ";
        switch (typeCalc){
            case 0: std::cout << "linsolve" << std::endl; break;
        case 1: std::cout << "linsolve + chol" << std::endl; break;
        case 2: std::cout << "inv" << std::endl; break;
        case 3: std::cout << "Schulz" << std::endl; break;
        case 4: std::cout << "chol + inv" << std::endl; break;
        case 5: std::cout << "Modern Schulz" << std::endl; break;
        case 6: std::cout << "svd" << std::endl; break;
        default: std::cout << "Неправильный ввод" << std::endl; return 0; break;


        }







        if (timeClacFlag)
              std::cout << "Расчет времени ЧАСТЕЙ keypon_relief" << endl;
        else
              std::cout << "Расчет времени ВСЕЙ keypon_relief" << endl;








        af::array x_sub_prm(x_values.size(), x_values.data());
		af::array y_sub_prm(y_values.size(), y_values.data());



        x_sub_prm = x_sub_prm(range(P));
        y_sub_prm = y_sub_prm(range(P));

        int Ni = 25;  //Число помех


 std::cout <<"Interf count Ni = " << Ni << std::endl;
std::cout <<"Interf power Pi_log = " << Pi_log << std::endl;




		

        const bool is_from_file = false;

        af::array signal,noise;
		af::array interf0;

		if (!is_from_file) {

            float Rmin = u_step*4;
            float Rmax = sinf(6.5/2.0*M_PI/180.0);
            //Равномерное плоское распределение 25 помех в пределах радиусов Rmin до Rmax
            uniformCirclePolar(Ni, Rmin, Rmax,ui, vi);

            //print_dims2(ui,"ui");


            af::array Pint = af::constant(pow(10,(Pi_log / 10)),1,Ni,f32);

            FormSignal(N, x_sub_prm, y_sub_prm, ui, vi, f0, Pint,signal,noise);



		}
		else {
			string file_signal = "signal_matlab.txt";
			string file_interf0 = "interf0_matlab.txt";

			std::pair<std::vector<double>, std::vector<double>> real_img = parseComplex(file_signal);

			std::vector<double> realv = real_img.first;
			std::vector<double> imgv = real_img.second;

			af::array re1(realv.size(), realv.data());
			af::array im1(imgv.size(), imgv.data());

			std::cout << "re dims: " << re1.dims() << std::endl;
			std::cout << "im dims: " << im1.dims() << std::endl;

			re1 = af::transpose(af::moddims(re1, 1000, 341));
			im1 = af::transpose(af::moddims(im1, 1000, 341));

			signal = af::complex(re1, im1);


			real_img = parseComplex(file_interf0);

			realv = real_img.first;
			imgv = real_img.second;

			af::array re2(realv.size(), realv.data());
			af::array im2(imgv.size(), imgv.data());

			std::cout << "re dims: " << re2.dims() << std::endl;
			std::cout << "im dims: " << im2.dims() << std::endl;

			re2 = af::transpose(af::moddims(re2, 1000, 341));
			im2 = af::transpose(af::moddims(im2, 1000, 341));

			interf0 = af::complex(re2, im2);
		}



		//int Nu = 101;
		//int Nv = 101;



        af::array u0_seq = af::seq(-ulim, ulim, u_step);
        af::array u0 = u0_seq;


        af::array v0 = u0;
		//af::array u0 = std::sin(1.0 * (M_PI / 180.0)) * linspace(-1, 1, Nu);
		//af::array v0 = std::sin(1.0 * (M_PI / 180.0)) * linspace(-1, 1, Nv);

		double Nu = u0.elements();
		double Nv = v0.elements();

        std::pair<af::array, af::array> uv = mesh_grid(u0.as(f32), v0.as(f32));

        af::array u = transpose(uv.first);
        af::array v = transpose(uv.second);


        double mu = 1000.0;

		af::array z0;

        double tol_Shulz = 1e-05;    //Точность вычисление Шульца
        int iter_num_out;





        af::array U = getU(x_sub_prm, y_sub_prm, u, v, f0);

		//!!Время!! DONE


        af::array signal_batch = af::tile(signal,1,1,batch_x,batch_y);
        af::array U_batch = af::tile(U,1,1,batch_x,batch_y);

        //Для прогрева
        z0 = keypon_relief(signal_batch, U_batch, mu, typeCalc,iter_num,tol_Shulz, iter_num_out,false);




        af::sync();



        //Вычислени рельефа Кейпона
         GET_TIME(z0 = keypon_relief(signal_batch, U_batch, mu, typeCalc,iter_num,tol_Shulz, iter_num_out,timeClacFlag);,"keypon_relief",direction_count,!timeClacFlag)







		af::array vals, idx;
		(af::max)(vals, idx, af::real(z0), 1);

        double max_val = vals.scalar<float>();
		int row0 = idx.scalar<unsigned>();

        //std::cout << "max=" << max_val << " at row0=" << row0 << std::endl;

		//af_print(z0);

		af::array z0_test = af::real(z0);

        std::vector<float> z0_values;
		std::ofstream out;

		z0_values.resize(z0_test.elements());

		z0_test.host(z0_values.data());

		out.open("z_values.txt");

		if (!out.is_open()) {
			std::cerr << "Can not open file!";

			throw;
		}

		for (double elem : z0_values) {
			out << elem << std::endl;
		}

		out.close();

		//af_print(z0);

        float h = 1000;

		//!!Время!!
        std::vector<af::array> threashold;



        //Для прогрева
        threashold = threashold_keipon(af::abs(z0), h, u, v, Nu, Nv);
        af::sync();

        //print_dims2(u,"u");
        //print_dims2(v,"v");






        GET_TIME(threashold = threashold_keipon(af::abs(z0), h, u, v, Nu, Nv);,"threashold_keipon",direction_count,!timeClacFlag)


        u = ulim*linspace(-1, 1, 320);
        v = af::constant(0.0,1,u.elements(),f32);


        U = getU(x_sub_prm, y_sub_prm, u, v, f0);


        af::array z_signal_DO;

        //Для прогрева
        z_signal_DO = DO_beams(signal,   U);

        //ПОмеховый канал на лучах
        GET_TIME(z_signal_DO = DO_beams(signal,   U);,"DO_beams",1,true)



         af::array Y_beams;


        //Адаптация!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        float delta_uv = 0.0025;
        std::vector<float> ur00 ={1, -1, -1, 1, 0};
        std::vector<float> vr00 ={1, 1, -1, -1, 0};

        af::array ur(1,5,ur00.data());
        af::array vr(1,5,vr00.data());

        //целевые лучи
        ur = ur*delta_uv;
        vr = vr*delta_uv;




        af::array Uc = getU(x_sub_prm, y_sub_prm, ur, vr, f0);


        //print_dims2(Uc,"Uc");

        af::array Uc_batch;
        if ((batch_x*batch_y) > 1)
        {
            signal_batch = af::tile(signal,1,1,batch_adapt_x,batch_adapt_y);
            Uc_batch = af::tile(Uc,1,1,batch_adapt_x,batch_adapt_y);
        }else
            Uc_batch = Uc;

        int direction_count_adapt1 = 1;
        if( direction_count > 1)
            direction_count_adapt1 =  direction_count_adapt;






        //для прогрева
        af::array Y2 =  adapt(signal_batch,  Uc_batch, mu,iter_num, tol_Shulz, iter_num_out,typeCalc);
        af::sync();

        GET_TIME(Y_beams =  adapt(signal_batch,  Uc_batch, mu,iter_num, tol_Shulz, iter_num_out, typeCalc);,"adapt",direction_count_adapt1,true)



        af::array Ui = getU(x_sub_prm, y_sub_prm, ui, vi, f0);

        af::array Uc_beams = af::tile(Uc,1,direction_count_adapt1);


        //Адаптация на лучах
        af::array z_noise = adapt_beams(noise, Uc_beams,  Ui, mu, iter_num,  tol_Shulz, iter_num_out,  typeCalc);

        af::array z_signal;
        GET_TIME(z_signal = adapt_beams(signal, Uc_beams,  Ui, mu, iter_num,  tol_Shulz, iter_num_out,  typeCalc);,"adapt beams",1,true)

        af::array z_noise0 = DO_beams(noise,   Uc_beams);

        af::array z_signal0 = DO_beams(signal,   Uc_beams);


        //Вычисление ОПШ
        float qi_log = cal_SNI(z_signal, z_noise);

        float qi_log0 = cal_SNI(z_signal0, z_noise);


        std::cout << "ОПШ после адаптации на лучах = " << qi_log << std::endl;
        std::cout << "ОПШ после ДО = " << qi_log0 << std::endl;




        return 0;


		af::array up = threashold[0];
		af::array vp = threashold[1];
		af::array zp = threashold[2];

		af_print(up);
		af_print(vp);
		af_print(zp);


		af::array phase1 = 2.0 * M_PI * (af::matmulNT(x_sub_prm, u.as(f64)) + af::matmulNT(y_sub_prm, v.as(f64))) * f0 / LIGHT_SPEED;

		printf("phase1 dimensions: %lld x %lld\n", phase1.dims(0), phase1.dims(1));

		af::array product = af::matmul(af::transpose(U, true), interf0); 
		af::array magnitude = af::real(product);
		af::array z2 = af::stdev(magnitude, 1);

		af::array Z = af::moddims(z0, Nv, Nu);

		af::array Z_log = 10 * af::log10(af::abs(Z));

		af::array Z2 = af::moddims(z2, Nv, Nu);
		af::array Z_log2 = 10 * af::log10(af::abs(Z2) * af::abs(Z2) + 1e-08);

		std::cout << "\nmatrix v0: " << v0.dims() << std::endl;
		std::cout << "matrix u0: " << u0.dims() << std::endl;
		std::cout << "matrix Z_log: " << Z_log.dims() << std::endl;



		return 0;

	}
	catch (const af::exception& e) {
		std::cerr << "ArrayFire error: " << e.what() << std::endl;

		throw;
	}


	
}


