#include "inv_schulz.h"
#include <arrayfire.h>
#include  <iostream>

// Обращение методом Шульца
//[in]
//A - входная матрица
//epsilon - точность
//max_iter - максимальное число итерации

//[out]
//iter_num - число итераций
//Возвращает обратную матрицу
af::array inv_shulz(af::array A,double epsiolon,int max_iter, int &iter_num)
{
    int P = A.dims(0);
    int batch_size_x = A.dims(2);
    int batch_size_y = A.dims(3);

    af::array aa;

    af::array  V(A.dims(0),A.dims(1),A.dims(2),A.dims(3),c32);
    //af::array  V;


    //print_dims2(A,"A");

    float norm1,norm2;
    af::array norms;

    for(int i=0; i < batch_size_x; i++)
    for(int j=0; j < batch_size_y; j++)
    {
        aa  = A(af::span,af::span,i,j);

        norm1 = af::norm(aa,AF_NORM_VECTOR_1);
        norm2 = af::norm(aa,AF_NORM_VECTOR_INF);

        //print_dims2(aa,"aa");


        float mult  = 1/(norm1*norm2);



        //V(af::span,af::span,i) = transpose(aa)/norms;
        V(af::span,af::span,i,j) = transpose(aa)*mult;
    }




     //af::array V = transpose(A)/(af::norm(A,AF_NORM_VECTOR_1)*af::norm(A,AF_NORM_VECTOR_INF));



    af::array I = af::identity(P, P, batch_size_x, batch_size_y,  c32);
    //af::array I = af::identity(P, P,  c32);


    af::array V1,AV,VAV,Yn,error,error_one;

    double error_float;

    iter_num = max_iter;

  for(int i = 0; i < max_iter; i++)
  {

      AV = af::matmul(A,V);

      error = af::max(af::flat(af::abs(I - AV)));
      error_float = error.scalar<float>();

      //std::cout << error_float << std::endl;

      if (error_float < epsiolon)
      {


          iter_num = i;

          break;
      }



        //V = 2*V - af::matmul(V,A,V);


        //V1 =(2*I - af::matmul(A,V));          // Формула №1
        //V = af::matmul(V,V1);


        //VAV = af::matmul(V,A,V);
        //V = 3*V-3*VAV + af::matmul(VAV,A,V);






        V1 = (3*I-3*AV + af::matmul(AV,AV));
        V = af::matmul(V,V1);


        /*
        AV = af::matmul(A,V);
        V = matmul(V,(7*I + matmul(AV,(-21*I + matmul(AV,(35*I + matmul(AV,(-35*I + matmul(AV,(21*I +   matmul(AV,(-7*I+AV))))))))))));
        */


        //Yn=I-matmul(A,V);
        //V = matmul(V,(I + matmul(Yn,(I + matmul(Yn ,(I + matmul(Yn,(I + matmul(Yn,(I + matmul(Yn,(I + matmul(Yn,(I+Yn))))))))))))));  // Формула №6



        //V=V*(3*I-3*AV+AV*AV);

        //V=V*(3*I-3*A*V+(A*V)^2);% Формула №2
        //V=V*(3*I-A*V*(3*I-A*V));% Формула №3
        //V=(I+1/4*(I-V*A)*(3*I-V*A)^2)*V;% Формула №4
        //V=V*(7*I+A*V*(-21*I+A*V*(35*I+A*V*(-35*I+A*V*(21*I+A*V*(-7*I+A*V))))));% Формула №5

        //Yn=I-A*V;
        //V=V*(I+Yn*(I+Yn*(I+Yn*(I+Yn*(I+Yn*(I+Yn*(I+Yn)))))));% Формула №6
        //pn=A*V; en=17*I+pn*(-28*I+pn*(22*I+pn*(-8*I+pn))); V=-1/16*V*en*(-8*I+pn*en);% Формула №7







    };

   return V;
}



/*
af::array inv_triangle(af::array L)
{
    af::solve(af::transpose(L, true), U, AF_MAT_LOWER)

}
*/


// Функция Ряда Неймена с поддержкой batch
af::array matrix_INV_Neumann_batch(const af::array& A0, int order)
{

    dim_t n = A0.dims(0);
    int batch_x = A0.dims(2);
    int batch_y = A0.dims(3);

    af::array norm = af::max(af::sum(af::abs(A0)));

    af::array norm1 = af::tile(1/norm,n,n);


    af::array A = A0*norm1;
    af::array I = af::identity(n, n, batch_x, batch_y, A.type());

    // Простейшая аппроксимация ряда Неймана
    af::array V = I;
    af::array term = I - A;
    af::array D = I - A;


    for (int i = 1; i <= order; i++) {
        V = V + term;
        term = af::matmul(term, D);
    }

    V = V*norm1;

    return V;
}











// Обращение методом Шульца c поддержкой batch
//[in]
//A - входная матрица
//epsilon - точность
//iter_num_max - максимальное число итерации

//[out]
//iter_num - число итераций
//epsilon_cur2 - массив квадратов ошибок на каждой итерации
//Возвращает обратную матрицу
af::array matrix_INV_Schulz_fast_batch( const af::array& A, double epsilon, int iter_num_max, int& iter_num,
                           std::vector<double>& epsilon_cur2)
{

    // Получаем размеры матрицы
    dim_t n = A.dims(0);
    //dim_t m = A.dims(1);

    dim_t n2 =n*n;

    dim_t n_batch = A.dims(2);
    dim_t m_batch = A.dims(3);

    // Сначала делаем Неймена
    af::array V = matrix_INV_Neumann_batch(A, 4);



    //Начальное приближение
    //af::array V = transpose(A)/(af::norm(A,AF_NORM_VECTOR_1)*af::norm(A,AF_NORM_VECTOR_INF));



    // Единичная матрица
    af::array I = af::identity(n, n, n_batch, m_batch, A.type());

    // Вычисляем AV
    af::array AV = af::matmul(A, V);

    // Квадрат эпсилон для сравнения
    double epsilon2 = epsilon * epsilon;

    // Вычисляем матрицы для определения коэффициента al
    af::array C = AV - I;
    af::array D = 2.0 * I - 3.0 * AV + af::matmul(AV, AV);
    af::array B = af::matmul(AV, D);

    // Вычисляем коэффициент al
    // sum(C.*conj(B), 'all') в Matlab
    af::array C_conjB = C * af::conjg(B);
    af::array numerator = -af::sum(af::moddims(C_conjB,n2,1,n_batch,m_batch));


    // sum(real(B).^2 + imag(B).^2, 'all') в Matlab
    af::array B_real = af::real(B);
    af::array B_imag = af::imag(B);
    af::array B_sq = af::pow(B_real, 2) + af::pow(B_imag, 2);
    af::array denominator = af::sum(moddims(B_sq,n2,1,n_batch,m_batch));

    af::array al = af::tile(numerator / denominator,n,n,1,1);





    // Обновляем V и AV
    V = V + al * af::matmul(V, D);
    AV = AV + al * B;

    //af_print(V);

    // Резервируем память для epsilon_cur2
    epsilon_cur2.reserve(iter_num_max);

    // Далее вычисляем стандартного Шульца
    for (int n = 0; n < iter_num_max; n++) {
        C = AV - I;

        // max(real(C).^2 + imag(C).^2,[],'all') в Matlab
        af::array C_real = af::real(C);
        af::array C_imag = af::imag(C);
        af::array C_sq = af::pow(C_real, 2) + af::pow(C_imag, 2);






        double current_epsilon = af::max<double>(C_sq);
        epsilon_cur2.push_back(current_epsilon);

        if (current_epsilon < epsilon2) {
            iter_num = n + 1; // +1 потому что в Matlab индексация с 1
            return V;
        }

        // Формула №1: V = V*(2*I - AV)
        V = af::matmul(V, 2.0 * I - AV);
        AV = af::matmul(A, V);
    }

    iter_num = iter_num_max;

    return V;
}














void print_shulz_error(af::array A,af::array A1)
{
    int P = A.dims(0);
    af::array I = af::identity(P, P,  c32);

    af::array error = af::max(af::max(af::abs(I - af::matmul(A,A1))));
    float zz;
    error.host(&zz);





     std::cout << "ошибка Шульца " << zz << std::endl;


}
