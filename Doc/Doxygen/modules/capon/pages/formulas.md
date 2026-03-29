@page capon_formulas capon — Математические формулы

@tableofcontents

@section capon_formulas_intro Введение

Алгоритм Capon (MVDR) минимизирует выходную мощность при ограничении единичного усиления
в целевом направлении. Все матричные операции выполняются на GPU через rocBLAS/rocsolver.

@section capon_formulas_covariance Ковариационная матрица

Оценка пространственной ковариационной матрицы по \f$ N \f$ отсчётам с \f$ P \f$ каналами:

\f[
R = \frac{1}{N} Y Y^H + \mu I \in \mathbb{C}^{P \times P}
\f]

где:
- \f$ Y \in \mathbb{C}^{P \times N} \f$ — матрица сигналов (\f$ P \f$ каналов, \f$ N \f$ отсчётов)
- \f$ Y^H \f$ — эрмитово сопряжение
- \f$ \mu \f$ — параметр диагональной загрузки (regularization), предотвращает вырожденность
- \f$ I \f$ — единичная матрица

@note Диагональная загрузка \f$ \mu I \f$ критична для численной стабильности:
типичные значения \f$ \mu \in [10^{-4}, 10^{-1}] \f$.

Реализация: `CovarianceMatrixOp` — rocBLAS CGEMM для \f$ Y Y^H \f$, затем HIP kernel
для нормализации \f$ 1/N \f$ и добавления \f$ \mu I \f$.

@section capon_formulas_cholesky Cholesky-обращение

Обращение ковариационной матрицы через Cholesky-разложение (положительно определённая матрица):

\f[
R = L L^H \quad \text{(POTRF — Cholesky factorization)}
\f]

\f[
R^{-1} \quad \text{(POTRI — inversion from Cholesky factor)}
\f]

где \f$ L \f$ — нижнетреугольная матрица.

@warning Если \f$ R \f$ не является положительно определённой (например, при малом \f$ \mu \f$),
POTRF вернёт ошибку. Увеличьте \f$ \mu \f$.

Реализация: `CaponInvertOp` делегирует обращение модулю `vector_algebra::CholeskyInverterROCm`.

@section capon_formulas_relief Relief (пространственный спектр)

Пространственный спектр Capon (MVDR Power Spectrum):

\f[
z[m] = \frac{1}{\text{Re}(u_m^H\, R^{-1}\, u_m)}, \quad m = 0, \ldots, M-1
\f]

где:
- \f$ u_m \in \mathbb{C}^{P} \f$ — вектор управления (steering vector) для направления \f$ m \f$
- \f$ M \f$ — количество направлений сканирования
- \f$ R^{-1} \in \mathbb{C}^{P \times P} \f$ — обращённая ковариационная матрица

Реализация: `CaponReliefOp` — HIP kernel, вычисляет \f$ u^H R^{-1} u \f$ для всех \f$ M \f$ направлений параллельно.

@section capon_formulas_beamforming Adaptive Beamforming

Вычисление адаптивных весов и формирование выходных сигналов:

\f[
W = R^{-1} U \in \mathbb{C}^{P \times M}
\f]

\f[
Y_{\text{out}} = W^H Y \in \mathbb{C}^{M \times N}
\f]

где:
- \f$ U \in \mathbb{C}^{P \times M} \f$ — матрица steering-векторов для \f$ M \f$ направлений
- \f$ W \f$ — матрица адаптивных весов (MVDR weights)
- \f$ Y_{\text{out}} \f$ — выходные сигналы по \f$ M \f$ направлениям, \f$ N \f$ отсчётов каждый

Реализация:
- `ComputeWeightsOp` — rocBLAS CGEMM: \f$ W = R^{-1} U \f$
- `AdaptBeamformOp` — rocBLAS CGEMM: \f$ Y_{\text{out}} = W^H Y \f$

@section capon_formulas_steering ULA Steering Vector

Для равномерной линейной решётки (ULA) с \f$ d/\lambda = 0.5 \f$:

\f[
u_p(\theta) = \exp\!\big(j \cdot 2\pi \cdot p \cdot 0.5 \cdot \sin\theta\big), \quad p = 0, \ldots, P-1
\f]

где:
- \f$ p \f$ — номер антенного элемента
- \f$ \theta \f$ — угол прихода (от нормали к решётке)
- \f$ d/\lambda = 0.5 \f$ — расстояние между элементами в долях длины волны (критерий Найквиста)

@note Для 2D решётки (плоской) steering vector обобщается на два угла (азимут и угол места).

@section capon_formulas_summary Сводка операций

| Операция | Формула | GPU библиотека |
|----------|---------|----------------|
| Ковариационная матрица | \f$ R = YY^H/N + \mu I \f$ | rocBLAS CGEMM |
| Cholesky-разложение | \f$ R = LL^H \f$ | rocsolver POTRF |
| Обращение | \f$ R^{-1} \f$ | rocsolver POTRI |
| Весовые коэффициенты | \f$ W = R^{-1}U \f$ | rocBLAS CGEMM |
| Relief | \f$ z[m] = 1/\text{Re}(u^H R^{-1} u) \f$ | HIP kernel |
| Beamforming | \f$ Y_{\text{out}} = W^H Y \f$ | rocBLAS CGEMM |

@see capon_overview
@see capon_tests
