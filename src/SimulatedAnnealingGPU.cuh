#ifndef B951BB41_D03D_4461_AB77_CCD976C9C275
#define B951BB41_D03D_4461_AB77_CCD976C9C275
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <math.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <src/cublas_utils.h>
#include <src/helper_cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// template<typename T>
// using func_t = double (*) (double *);

__global__ void function(double * a, double * res, int dim, int size);
__global__ void print(double * a, int size);
__global__ void Ones(double * a, int size);
__global__ void compareAndAssign(double * bestFuncVals, double * funVals, double *x, double *pb, int pNum, int dim);
__global__ void minusGroupBest(double * x, double * tp, double * group_best, double factor, int dim, int size);
__global__ void Add(double *A, double *B, double *C, int numElements);
__global__ void Scale(double *A, double factor, int size);
__global__ void Minus(double *A, double *B, double *C,  double factor,int numElements);
__global__ void Norm(double *A, double *B, int pNum, int dim);
__global__ void SumRowwise(double * A, double * res, int pNum, int dim);
// __device__ void atomicMin_double(double* const address, const double value);
__device__ double atomicMin_double(double* address, double val);
__global__ void reduceMinIdxOptimized(const double* __restrict__ input, const int size, double* minOut, int* minIdxOut);
__global__ void updateXNew(double *x, double * x_new, double * u, double t, int size);
__global__ void cudaRand(double *d_out, double lb, double ub, int size);
__global__ void Anneal(double *diff, double * x, double * x_new, double* y_new, double* y, double* minVal, double * group_best, double t, double * u, double * delta_f, int xNum, int dim);



template<std::size_t L>
class SimulatedAnnealingGPU{
public:
    SimulatedAnnealingGPU(int initialTrails=1e5, int MaxIters=1500,int innerIter=300, double T_max=100., double T_min = 1e-5, double tolerance=1e-10, int threadsNum=256);
    void run();
    double getOptimal();
    double * getSol();
private:
    double t, tMax, tMin;
    int maxIter, innerMaxIter, iter, stay, threads, size;
    double tol;
    double optimal;
    int xNum;
    int blocks, block_for_xNum;
    double * y;
    double * d_y;
    double * d_y_new;
    double * tmp_y;
    double * group_best;
    double * x;
    double * u;
    double * d_x;
    double * d_x_new;
    double * ddiff;
    double * delta_f;
    double * df;
    double *minValue;
    int * minIndex;
    double* sol;
    int * sol_id;
    void _updateTemp();
    void _updateXNew();
    void _initX();
    void _initCuda();
    void _updateCurrentFuncValues();
    double _randGen(double ub = 1.0, double lb = 0.0);
    void _randInit(double * data, int size, double scale);
    void findGroupBest();
    void evaluate();  
    void evaluateXNew();  
    void anneal();
    double minFunVal(int * id);  
    // double minFunValCPU(int * id);
};

// template<std::size_t L>
template<std::size_t L>
SimulatedAnnealingGPU<L>::SimulatedAnnealingGPU(int initialTrails, int MaxIters ,int innerIter, double T_max, double T_min, double tolerance, int threadsNum){
    // function = func;
    tol = tolerance;
    maxIter = MaxIters;
    innerMaxIter= innerIter;
    iter = 0;
    stay = 0;
    xNum = initialTrails;
    size = xNum * L;
    threads = threadsNum;
    blocks = size/threads + 1;
    block_for_xNum = xNum / threads + 1;
    tMax = T_max;
    tMin = T_min;
    t = T_max;
    sol = new double[L];
    sol_id = new int;
    df = new double;
    _initX();
    _initCuda();
    evaluate();
    // optimal = minFunVal(sol_id);
    // findGroupBest();
    // checkCudaErrors(cudaMemcpy(sol, group_best, L * sizeof(double), cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(bestFuncVals, d_y, xNum * sizeof(double), cudaMemcpyDeviceToDevice)); 
    // bestFuncVals = d_y;
    optimal = minFunVal(sol_id);   
    printf("optimal = %lf, minId = %d pos=%d\n", optimal, *sol_id, (*sol_id) * L);
    checkCudaErrors(cudaMemcpy(group_best, d_x + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToDevice)); 
    checkCudaErrors(cudaMemcpy(sol, group_best, L * sizeof(double), cudaMemcpyDeviceToHost)); 
    // printf("group best\n");
    // print<<<blocks,threads>>>(group_best, L);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::run(){
    for(int ii=0; ii < maxIter; ii++){
        for(int i=0; i < innerMaxIter; i++){
            _updateXNew();
            evaluateXNew();
            Minus<<<block_for_xNum, threads>>>(d_y_new, d_y, ddiff, 1.0, xNum);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            anneal();
        }
        checkCudaErrors(cudaMemcpy(df, delta_f, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&optimal, minValue, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(sol, group_best, L*sizeof(double), cudaMemcpyDeviceToHost));
        printf("Iter%d: Optimal=%lf Temp=%lf ∆f=%lf x=[", ii, optimal, t, *df);
        for(int j = 0; j < L-1; j++){
                printf("%lf ", sol[j]);
        }
        printf("%lf]\n", sol[L-1]);
        iter ++;
        _updateTemp();
        if( *df < tol ) stay++;
        else stay = 0;
        if (t < tMin) break;
        if (stay > 150) break;
    }
}

template<std::size_t L>
double SimulatedAnnealingGPU<L>::minFunVal(int * id) {
    double * res = new double;
    checkCudaErrors(cudaMemcpy(minValue, d_y, sizeof(double), cudaMemcpyDeviceToDevice));
    reduceMinIdxOptimized<<<1, threads>>>(d_y, xNum, minValue, minIndex);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // (*id)-=1;
    checkCudaErrors(cudaMemcpy(res, minValue, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(id, minIndex, sizeof(int), cudaMemcpyDeviceToHost));
    return *res;
}


template<std::size_t L>
void SimulatedAnnealingGPU<L>::_initX() {
    x = new double[size];
    _randInit(x, size, 100.0);
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::anneal() {
    cudaRand<<<block_for_xNum, threads>>>(u, 0.0, 1.0, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize());
    Anneal<<<block_for_xNum, threads>>>(ddiff, d_x, d_x_new, d_y, d_y_new, minValue, group_best, t, u, delta_f, xNum, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize());
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::_updateTemp(){
    t = tMax / log(iter + 1);
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::_updateXNew(){
    cudaRand<<<blocks, threads>>>(u, -1.0, 1.0, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize());
    updateXNew<<<blocks, threads>>>(d_x, d_x_new, u, t, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize());
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::findGroupBest() {
    checkCudaErrors(cudaMemcpy(group_best, d_x + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToDevice)); 
}


template<std::size_t L>
void SimulatedAnnealingGPU<L>::_initCuda() {
    //TODO: Use cudaMemcpyAsync
    checkCudaErrors(cudaMalloc((void **) &d_x, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &u, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &tmp_y, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_x_new, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_y, xNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &ddiff, xNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_y_new, xNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &group_best, L * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &minValue, sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &delta_f, sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &minIndex, sizeof(int)));
    // CUBLAS_CHECK(cublasCreate(&handle));
}


template<std::size_t L>
void SimulatedAnnealingGPU<L>::evaluate() {
    int dim = L;
    function<<<blocks, threads>>>(d_x, tmp_y, dim, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    SumRowwise<<<block_for_xNum, threads>>>(tmp_y, d_y, xNum, dim);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::evaluateXNew() {
    int dim = L;
    function<<<blocks, threads>>>(d_x_new, tmp_y, dim, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    SumRowwise<<<block_for_xNum, threads>>>(tmp_y, d_y_new, xNum, dim);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // printf("d_y\n");
    // print<<<blocks, threads>>>(d_y, xNum);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
double SimulatedAnnealingGPU<L>::_randGen(double ub, double lb) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distribution(lb, ub);
    return distribution(eng);
}

template<std::size_t L>
void SimulatedAnnealingGPU<L>::_randInit(double * data, int size, double scale) {
    for (int i = 0; i < size; i++){
        data[i] = scale * _randGen();
    }
}

template<std::size_t L>
double SimulatedAnnealingGPU<L>::getOptimal() {
    return optimal;
}

template<std::size_t L>
double* SimulatedAnnealingGPU<L>::getSol() {
    return sol;
}

__global__ void function(double * a, double * res, int dim, int size){
    // printf("Entered\n");
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    double tmp = a[index];
    if (index < size){
        res[index] = tmp * (tmp - (double)(index % dim + 1));
        // printf("index = %d tmp = %lf tmp1 = %lf res = %lf\n", index, tmp, (tmp - (double)(index % dim + 1)), res[index]);
    }
}

__global__ void print(double * a, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        printf("index[%d]=%lf\n", index, a[index]);
    }
}

__global__ void Ones(double * a, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        a[index] = 1.0;
    }
}

__global__ void compareAndAssign(double * bestFuncVals, double * funVals, double *x, double *pb, int pNum, int dim){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < pNum){
        if(funVals[index] < bestFuncVals[index]) {
            bestFuncVals[index] = funVals[index];
            for(int i = 0; i < dim; i++){
                pb[index * dim + i] = x[index * dim + i];
            }
        }
    }
}

__global__ void minusGroupBest(double * x, double * tp, double * group_best, double factor, int dim, int pNum){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < pNum){
        for(int i = 0; i < dim; i++){
            tp[index * dim + i] = factor * (group_best[i] - x[index * dim + i]);
        }
    }
}

__global__ void Add(double *A, double *B, double *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void Scale(double *A, double factor, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size){
        A[i] = A[i] * factor;
    }
}

__global__ void Minus(double *A, double *B, double *C, double factor, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = factor * (A[i] - B[i]);
    }
}

__global__ void Norm(double *A, double *B, int pNum, int dim){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < pNum){
        B[index] = 0;
        for(int i = 0; i < dim; i++){
            B[index] += A[index * dim + i] * A[index * dim + i];
        }
    }
}

__global__ void SumRowwise(double * A, double * res, int pNum, int dim){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < pNum){
        res[index] = 0;
        for(int i = 0; i < dim; i++){
            res[index] += A[index * dim + i];
        }
    }
}

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
__global__ void reduceMinIdxOptimized(const double* __restrict__ input, const int size, double* minOut, int* minIdxOut)
{
    double localMin = input[0];
    int localMinIdx = 0;
    if(threadIdx.x<size){
        for (int i = threadIdx.x; i < size; i += blockDim.x)
        {
            double val = input[i];
            // printf("input[%d]=%lf\n", i, val);
            if (localMin > val)
            {
                localMin = val;
                localMinIdx = i;
            }
        }
    }
  
    atomicMin_double(minOut, localMin);
  
    __syncthreads();
  
    if (*minOut == localMin)
    {
        *minIdxOut = localMinIdx;
    }
}

__global__ void cudaRand(double *d_out,  double lb, double ub, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size){
        curandState state;
        curand_init((unsigned long long)clock() + i, 0, 0, &state);
        double tmp = curand_uniform_double(&state);
        d_out[i] = (ub - lb) * tmp + lb;
    }
}

__global__ void updateXNew(double *x, double * x_new, double * u, double t, int size){
    int index = blockDim.x * blockIdx.x + threadIdx.x;;
    if(index < size){
        double uVal = u[index];
        double sign;
        if(uVal>0) sign = 1.;
        if(uVal==0) sign = 0.;
        if(uVal<0) sign = -1.;
        double factor = 1.0 + 1.0/t;
        u[index] = pow(factor, fabs(uVal)) - 1.0;
        x_new[index] = x[index] + 20. * t * sign * u[index];
    }
}

__global__ void Anneal(double *diff, double * x, double * x_new, double* y, double* y_new, double* minVal, double * group_best, double t, double * u, double * delta_f, int xNum, int dim){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < xNum){
        double diff_i = diff[index];
        double diff_over_t = exp(-1 * diff_i / t);
        if(diff_i < 0 || diff_over_t > u[index]){
            for(int i = 0; i < dim; i++){
                x[index * dim + i] = x_new[index * dim + i];
            }
            y[index] = y_new[index];
            if(y[index] < *minVal){
                double tmp = *minVal;
                *minVal = y[index];
                *delta_f = fabs(*minVal-tmp);
                for(int i = 0; i < dim; i++){
                    group_best[i] = x[index * dim + i];
                }
            }
        }
    }
    
}

#endif /* B951BB41_D03D_4461_AB77_CCD976C9C275 */
