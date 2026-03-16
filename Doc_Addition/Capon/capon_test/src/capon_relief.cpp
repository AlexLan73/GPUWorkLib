#define _USE_MATH_DEFINES

#include <arrayfire.h>
#include <iostream>
#include <math.h>
#include "capon_relief.h"
#include <fstream>
#include <ctime>
#include "inv_schulz.h"

//#include <windows.h>

using namespace af;






//Вывод размерности матрицы
void print_dims2(af::array in,std::string name_str)
{


    std::cout << "Тип " << name_str << ": ";

    switch (in.type()){
        case f32: std::cout << "f32"; break;
        case f64: std::cout << "f64"; break;
        case c32: std::cout << "c32"; break;
        case c64: std::cout << "c64"; break;
        case s32: std::cout << "s32"; break;
        case s64: std::cout << "s64"; break;
        case u32: std::cout << "u32"; break;
        case u64: std::cout << "u64"; break;
     default: std::cout << "другой"; break;

    }

    std::cout << std::endl;

    if ((in.dims(2) > 1)||(in.dims(3) > 1))
    {
        std::cout << "размерность " << name_str << " :" <<  in.dims(0) << ", " << in.dims(1);

        std::cout << ", " <<  in.dims(2) << ", " << in.dims(3) << std::endl;
    }else{

        std::cout << "размерность " << name_str << " :" <<  in.dims(0) << ", " << in.dims(1) << std::endl;
    }


}


//Вычисление вектра распределения сигнала по пространственным каналам (фазовые множители)
//[in]
//x_sub_prm, y_sub_prm  - массив координат антенных секций
//u, v - обобщенные координаты помехи
//f0 - несущая частота

//[out]
// матрица распределения сигнала помехи по пространственным каналам [число пространственных каналов]*[число направлений/помех]
af::array getU(af::array x_sub_prm, af::array y_sub_prm, af::array u, af::array v,double f0)
{


    af::array x_sub_prm_mult_u = af::matmul(x_sub_prm, u);
            af::array y_sub_prm_mult_v = af::matmul(y_sub_prm, v);


            af::array phase = 2.0 * af::Pi * (x_sub_prm_mult_u + y_sub_prm_mult_v) * f0/ LIGHT_SPEED;


            af::cfloat I(0.0, 1.0);
            //af::array Iarr = af::constant(I, phase.dims(), c64);
            //af::array Iarr = af::constant(I,1, c32);


            af::array U;
            U = af::exp( I * phase.as(c32) ) / sqrt(y_sub_prm.numdims());

            return U;

}



//Функция вычисление сигнала помеха+шума
//[in]
//N - число отсчетов
//x_sub_prm, y_sub_prm  - массив координат антенных секций
//ui, vi - обобщенные координаты помехи
//f0 - несущая частота

//[out]
//Матрица принятого сигнала (шум + помеха) hразмерностью [число приемных каналов]*[число отсчетов]
void FormSignal(int N, af::array x_sub_prm, af::array y_sub_prm, af::array ui, af::array vi,double f0,af::array Pint,af::array &signal,af::array &noise)
{
    int P = x_sub_prm.elements();  //Число каналов

    af::array Ui = getU(x_sub_prm, y_sub_prm, ui, vi, f0);

    noise = af::complex(randn(P, N, f32), randn(P, N, f32)) / sqrt(2);


    //af_print(sqrt(transpose(Pint) / 2.0).as(c64));

    af::array A = sqrt(Pint / 2.0);
    af::array B = af::complex(af::randn(ui.elements(), N, f32), af::randn(ui.elements(), N, f32));



    //print_dims2(A,"A");
    //print_dims2(B,"B");

    af::array D = af::diag(transpose(A).as(c32),0,false);

    //af_print(D);



    af::array interf0 = af::matmul(D,B);

    //print_dims2(Ui,"Ui");
    //print_dims2(interf0,"interf0");

    af::array interf = af::matmul(Ui, interf0);


    //print_dims2(x_sub_prm,"x_sub_prm");
    //print_dims2(interf,"interf");
    //print_dims2(noise,"noise");



    signal = interf + noise;

}



//Функция вычислет рельеф Кейпона
//[in]
//signal - Матрица принятого сигнала (шум + помеха) hразмерностью [число приемных каналов]*[число отсчетов]
//U - Матрица распределения сигнала помехи по пространственным каналам [число пространственных каналов]*[число направлений]
//mu - коэффициент регуляризации
//typeCalc - Алгоритм обращения 0  - linsolve, 1 - linsolve + chol, 2 - inv, 3 - итерации Шульца, 4 - chol + inv, 5 - спектральное разложение
//timeClacFlag - вычисление времени компонентов функции
//iter_num - число итераций Шульца
//tol_Shulz - точность вычисления Шульца

//[out]
//iter_num_out - число итераций Шульца
//массив величин рельфа Кепона. Размер массива равен числу направлений
af::array keypon_relief(af::array signal,  af::array U, double mu, short typeCalc,int iter_num, double tol_Shulz, int &iter_num_out,  bool timeClacFlag)
{
	try {



        //std::cout << "\tkeypon_relief:   " << std::endl;
        int Nelem = signal.dims(0);
        int batch_x = signal.dims(2);
        int batch_y = signal.dims(3);

		//std::cout << "Nelem size: " << Nelem;



        af::array R,I;

        I = af::identity(Nelem, Nelem, batch_x, batch_y, c32);
        GET_TIME(R = matmul(signal, af::transpose(signal, true)) + mu * I;,"R",1,timeClacFlag)

        //print_dims2(R,"R");



		af::array U1;
		af::array U2;
		af::array U3;

		af::array z;
		af::array L;
		af::array R1;

		af::array R2;

		array vals, idx;
		double max_val;      
		unsigned row0;

		std::vector<double> z_values;
		std::ofstream out;
        af::array UR;

		double elapsed;

        af::array uu,ss,vv,vv2,L1;

        std::vector<double> epsilon_cur(0);

		switch (typeCalc) {
		case 0:
            GET_TIME(U1 = af::solve(R, U),"solve(R, U)",1,timeClacFlag);

            GET_TIME(z = 1.0 / af::sum(conjg(U) * U1, 0),"z",1,timeClacFlag)

            return z;
			break;
		case 1:

            GET_TIME(cholesky(L,R,false);,"chol",1,timeClacFlag ) //Нижняя треугольная

            //print_dims2(L,"L");
            //print_dims2(U,"U");

            GET_TIME(U1 = solve(L, U, AF_MAT_LOWER);,"solve",1,timeClacFlag)

            GET_TIME(z = 1.0 / sum(conjg(U1) * U1, 0),"z",1,timeClacFlag)

            return z;
			break;
		case 2:
            GET_TIME(R1 = af::inverse(R);,"R1",1,timeClacFlag)

            GET_TIME(UR =  matmul(R1, U);,"UR",1,timeClacFlag)

            GET_TIME(z = 1.0 / sum(conjg(U) * UR, 0);,"z",1,timeClacFlag)

            return z;
            break;

        case 3:

            GET_TIME(R1 = inv_shulz(R,tol_Shulz,iter_num,iter_num_out);,"R1",1,timeClacFlag)



            GET_TIME(UR =  matmul(R1, U);,"UR",1,timeClacFlag)

            GET_TIME(z = 1.0 / sum(conjg(U) * UR, 0);,"z",1,timeClacFlag)

            //print_shulz_error(R,R1);


            return z;
            break;

        case 4:

            GET_TIME(cholesky(L,R,false);,"chol",1,timeClacFlag);  //Нижняя треугольная
            GET_TIME(L1 = solve(L,I,AF_MAT_LOWER);,"solve",1,timeClacFlag);   //Обратная к Холецкому

            GET_TIME(U1 =af::matmul(L1,U);,"L1*U",1,timeClacFlag);

            GET_TIME(z = 1.0 / sum(conjg(U1) * U1, 0),"z",1,timeClacFlag)
            return z;
            break;

        case 5:


            GET_TIME(R1 = matrix_INV_Schulz_fast_batch( R,tol_Shulz,iter_num,iter_num_out,epsilon_cur);,"R1",1,timeClacFlag)



            GET_TIME(UR =  matmul(R1, U);,"UR",1,timeClacFlag)

            GET_TIME(z = 1.0 / sum(conjg(U) * UR, 0);,"z",1,timeClacFlag)

            //print_shulz_error(R,R1);


            return z;
            break;


        case 6:
            GET_TIME(af::svd(uu,ss,vv,R);,"[u,s,v] = svd",1,timeClacFlag)

            //print_dims2(ss,"ss");
            //print_dims2(af::diag(1/sqrt(ss),0,false),"1/sqrt(ss)");

            GET_TIME(vv2 = af::matmul(af::diag(1/sqrt(ss.as(c64)),0,false),af::transpose(vv, true));,"diag(1/s)*v",1,timeClacFlag)
            GET_TIME(U1 = af::matmul(vv2,U);,"м*U",1,timeClacFlag)

            GET_TIME(z = 1.0 / sum(conjg(U1) * U1, 0),"z",1,timeClacFlag)

            return z;
            break;


		default:
			std::cout << "Incorrect type calc!" << std::endl;

            //std::cout << "\n\tВремя выполнения keypon_relief: " << elapsed << " секунд" << std::endl;
			return z;
			break;
		}
	}
	catch (const af::exception& e) {
		std::cerr << "ArrayFire error: " << e.what() << std::endl;

		throw;
	}
}

//Функция вычислет адаптивное ДО
//[in]
//Y - Матрица принятого сигнала (шум + помеха) hразмерностью [число приемных каналов]*[число отсчетов]
//U - Матрица распределения сигнала цели по пространственным каналам [число пространственных каналов]*[число лучей]
//mu - коэффициент регуляризации
//typeCalc - Алгоритм обращения 0  - linsolve, 1 - linsolve + chol, 2 - inv, 3 - итерации Шульца, 4 - chol + inv
//iter_num - число итераций Шульца
//tol_Shulz - точность вычисления Шульца


//[out]
//iter_num_out - число итераций Шульца
//Выход матрица сигнала в лучах [число лучей]*[число отсчетов]
af::array adapt(af::array Y,  af::array U, double mu, int iter_num, double tol_Shulz, int &iter_num_out,int typeCalc)
{
    int Nelem = Y.dims(0);


    int batch_x = Y.dims(2);
    int batch_y = Y.dims(3);

    af::array R, I;
    I = af::identity(Nelem, Nelem, batch_x, batch_y, c32);
    R = matmul(Y, af::transpose(Y, true)) + mu * I;
    std::vector<double> epsilon_cur(0);

    af::array w, Yout, L, U1, R1,L1;
    switch (typeCalc) {
    case 0:
        w = af::solve(R, U);

        break;
    case 1:

        cholesky(L,R,false);  //Нижняя треугольная

        //print_dims2(L,"L");
        //print_dims2(U,"U");

        U1 = solve(L, U, AF_MAT_LOWER);
        w = solve(af::transpose(L, true), U1, AF_MAT_UPPER);


        break;
    case 2:
        R1 = af::inverse(R);

        w = matmul(R1,U);

        break;
    case 3:


        R1 = inv_shulz(R,tol_Shulz,iter_num,iter_num_out);

        w = matmul(R1,U);

        //print_shulz_error(R,R1);


        break;

    case 4:

        cholesky(L,R,false);  //Нижняя треугольная
        L1 = solve(L,I,AF_MAT_LOWER);   //Обратная к Холецкому

        U1 =af::matmul(L1,U);
        w =af::matmulTN(conjg(L1),U1);

        break;
    case 5:
        R1 = matrix_INV_Schulz_fast_batch( R,tol_Shulz,iter_num,iter_num_out,epsilon_cur);

        w = matmul(R1,U);

        //print_shulz_error(R,R1);


        break;
    default:
        std::cout << "Incorrect type calc!" << std::endl;

        //std::cout << "\n\tВремя выполнения keypon_relief: " << elapsed << " секунд" << std::endl;
        break;
    }

    Yout = af::matmulTN(af::conjg(w),Y);  //Проверить здесь только транспонирование или еще комплексное сопряжения


    return Yout;




}









//Функция вычислет адаптивное ДО (адаптация на лучах)
//[in]
//Y - Матрица принятого сигнала (шум + помеха) hразмерностью [число приемных каналов]*[число отсчетов]
//Uс - Матрица распределения сигнала цели по пространственным каналам [число пространственных каналов]*[число целевых луче]
//Ui - Матрица распределения сигнала помехи по пространственным каналам [число пространственных каналов]*[число помеховых луче]
//mu - коэффициент регуляризации
//typeCalc - Алгоритм обращения 0  - linsolve, 1 - linsolve + chol, 2 - inv, 3 - итерации Шульца, 4 - chol + inv
//iter_num - число итераций Шульца


//[out]
//Выход матрица сигнала в лучах [число целевых лучей]*[число отсчетов]
af::array adapt_beams(af::array Y,  af::array Uc,  af::array Ui, double mu, int iter_num,  double tol_Shulz, int &iter_num_out, int typeCalc)
{

    //Целевые лучи
    af::array Yout,Yc = af::matmulTN(conjg(Uc),Y);

     std::vector<double> epsilon_cur(0);

    int Ni = 0;

    if (!(Ui.isempty()))
    {
        af::array Yi = af::matmulTN(conjg(Ui),Y);
        Ni = Ui.dims(1);

        af::array I  = af::identity(Ni, Ni, c32);

        //print_dims2(Y,"Y");
        //print_dims2(I,"I");

        af::array  R = matmulNT(Yi, conjg(Yi)) + mu * I;

        af::array Pc = matmulNT(Yi, conjg(Yc));


        af::array w,  L1,L, U1, R1;
        switch (typeCalc) {
        case 0:
            w = af::solve(R, Pc);

            break;
        case 1:
            cholesky(L,R,false);  //Нижняя треугольная

            U1 = solve(L, Pc, AF_MAT_LOWER);

            w = solve(af::transpose(L, true), U1, AF_MAT_UPPER);



            break;
        case 2:
            R1 = af::inverse(R);

            w = af::matmul(R1,Pc);

            break;
        case 3:

            //R1 = inv_shulz(R,iter_num);
            R1 = inv_shulz(R,tol_Shulz,iter_num,iter_num_out);

            w = af::matmul(R1,Pc);

            //print_shulz_error(R,R1);


            break;
        case 4:

            cholesky(L,R,false);  //Нижняя треугольная
            L1 = solve(L,I,AF_MAT_LOWER);   //Обратная к Холецкому

            U1 =af::matmul(L1,Pc);
            w =af::matmulTN(conjg(L1),U1);

            break;
        case 5:

            R1 = matrix_INV_Schulz_fast_batch( R,tol_Shulz,iter_num,iter_num_out,epsilon_cur);

            w = af::matmul(R1,Pc);

            //print_shulz_error(R,R1);



            break;
        default:
            std::cout << "Incorrect type calc!" << std::endl;

            //std::cout << "\n\tВремя выполнения keypon_relief: " << elapsed << " секунд" << std::endl;
            break;
        }




        Yout = Yc - af::matmulTN(af::conjg(w),Yi);  //Проверить здесь только транспонирование или еще комплексное сопряжения



    }else{

        Yout = Yc;

    }








    return Yout;




}




//Функция вычислет адаптивное ДО (адаптация на лучах)
//[in]
//Y - Матрица принятого сигнала (шум + помеха) hразмерностью [число приемных каналов]*[число отсчетов]
//Uс - Матрица распределения сигнала цели по пространственным каналам [число пространственных каналов]*[число целевых луче]

//[out]
//Выход матрица сигнала в лучах [число целевых лучей]*[число отсчетов]
af::array DO_beams(af::array Y,  af::array U)
{
    af::array z = af::matmulTN(conjg(U),Y);
    return z;

}



//Вычисление ОПШ
float cal_SNI(af::array z_signal, af::array z_noise)
{
    af::array sigma_signal = af::mean(real(z_signal*conjg(z_signal)),1);
    af::array sigma_noise = af::mean(real(z_noise*conjg(z_noise)),1);

    af::array qic = 10*af::log10(sigma_signal/sigma_noise);

    //af_print(qic)

    float qi = qic(z_signal.dims(0)-1).scalar<float>();
    return qi;


}


//Равномерное плоское распределение в пределах радиусов Rmin до Rmax
//[in] n - число точек
// Rmin - внутренний радиус
// Rmax - внешний радиус

//[out]
//x,y - обобщенные координаты

void uniformCirclePolar(int n, float Rmin,float Rmax,af::array &x,af::array &y)
{
    // Генерируем равномерные углы
    af::array theta = 2.0 * af::Pi *af::randu(1,n,f32);



    float a = (Rmin/Rmax); a = a*a;

    //Генерируем равномерные радиусы
    af::array r = Rmax * sqrt(a + (1-a)*af::randu(1,n,f32));


    // Преобразуем в декартовы координаты
    x = r* cos(theta);
    y = r* sin(theta);


}













