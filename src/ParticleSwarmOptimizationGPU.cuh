#ifndef B951BB41_D03D_4461_AB77_CCD976C9C276
#define B951BB41_D03D_4461_AB77_CCD976C9C276
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

__global__ void function(double * a, double * res, int dim, int size);
__global__ void updateParticles(double * a, double * v, double dt, int size);
__global__ void print(double * a, int size);
__global__ void Ones(double * a, int size);
__global__ void compareAndAssign(double * bestFuncVals, double * funVals, double *particles, double *pb, int pNum, int dim);
__global__ void minusGroupBest(double * particles, double * tp, double * group_best, double factor, int dim, int size);
__global__ void Add(double *A, double *B, double *C, int numElements);
__global__ void Scale(double *A, double factor, int size);
__global__ void Minus(double *A, double *B, double *C,  double factor,int numElements);
__global__ void Norm(double *A, double *B, int pNum, int dim);
__global__ void SumRowwise(double * A, double * res, int pNum, int dim);
__device__ double atomicMin_double(double* address, double val);
__global__ void reduceMinIdxOptimized(const double* __restrict__ input, const int size, double* minOut, int* minIdxOut);


template<std::size_t L>
class ParticalSwarmOptimizationGPU{
public:
    ParticalSwarmOptimizationGPU(int NumParticles=1e5, int MaxIters=1500, int Verbose=1, double C1=2.5, double C2=0.5, double weight=0.5, double tolerence=1e-10, double delta_t=1.0, int threadNum=256);
    void run();
    double getOptimal();
    double * getSol();
private:
    double c1;
    double c2;
    double w;
    double tol;
    double dt;
    double * particles;
    double * d_particles;
    double * personal_best;
    double * personal_best_func_values;
    double * current_func_values;
    double * group_best;
    double * v;
    double * d_v;
    double * v_norm;
    double * tmp_v;
    double * tmp_v1;
    double * funcVals;
    double * tmpFuncVals;
    double * bestFuncVals;
    double * sumVec;
    double * colOnes;
    double optimal;
    double *sol;
    double *minValue;
    int * sol_id;
    int particleNum;
    int maxIter;
    int * minIndex;
    int size;
    int verbose;
    int threads;
    int blocks, block_for_particleNum;
    const double alpha = 1.;
    const double beta = 0.;
    void _initParticles();
    void _initVelocity();
    void _initCuda();
    void _updateVelocity();
    void _updateParticles();
    void _updateCurrentFuncValues();
    void _randInit(double * data, int size, double scale);
    void findPersonalBest();
    void findGroupBest();
    void evaluate();  
    double minFunVal(int * id);  
    double _randGen();
};

template<std::size_t L>
ParticalSwarmOptimizationGPU<L>::ParticalSwarmOptimizationGPU(int NumParticles, int MaxIters, int Verbose, double C1, double C2, double weight, double tolerence, double delta_t, int threadNum){
    c1 = C1;
    c2 = C2;
    w = weight;
    tol = tolerence;
    maxIter = MaxIters;
    dt = delta_t;
    particleNum = NumParticles;
    size = L * particleNum;
    threads = threadNum;
    blocks = size/threads + 1;
    block_for_particleNum = particleNum / threads + 1;
    sol = new double[L];
    sol_id = new int;
    verbose = Verbose;
    _initParticles();
    _initVelocity();
    _initCuda();
    evaluate();
    checkCudaErrors(cudaMemcpy(bestFuncVals, funcVals, particleNum * sizeof(double), cudaMemcpyDeviceToDevice)); 
    optimal = minFunVal(sol_id);   
    checkCudaErrors(cudaMemcpy(group_best, d_particles + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToDevice)); 
    checkCudaErrors(cudaMemcpy(sol, group_best, L * sizeof(double), cudaMemcpyDeviceToHost)); 
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::run(){
    double tmpOptimal;
    _updateParticles();
    for(int i = 0; i < maxIter; ++i){
        evaluate();
        tmpOptimal = minFunVal(sol_id);
        double diff = fabs(optimal - tmpOptimal);
        if(diff < tol){
            optimal = tmpOptimal;
            checkCudaErrors(cudaMemcpy(sol, d_particles + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToHost)); 
            if(verbose){
                printf("Iter%d: minFunVal=%lf ∆f=%lf x=[", i, optimal, diff);
                for(int j = 0; j < L-1; j++){
                    printf("%lf ", sol[j]);
                }
                printf("%lf]\n", sol[L-1]);
            }
            break;
        }
        if(tmpOptimal < optimal){
            optimal = tmpOptimal;
            checkCudaErrors(cudaMemcpy(sol, d_particles + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToHost)); 
        }
        _updateVelocity();
        _updateParticles();
        findPersonalBest();
        findGroupBest();
        if(verbose){
            printf("Iter%d: minFunVal=%lf ∆f=%lf x=[", i, optimal, diff);
            for(int j = 0; j < L-1; j++){
                printf("%lf ", sol[j]);
            }
            printf("%lf]\n", sol[L-1]);
        }
    }
}

template<std::size_t L>
double ParticalSwarmOptimizationGPU<L>::minFunVal(int * id) {
    double * res = new double;
    checkCudaErrors(cudaMemcpy(minValue, funcVals, sizeof(double), cudaMemcpyDeviceToDevice));
    reduceMinIdxOptimized<<<1, threads>>>(funcVals, particleNum, minValue, minIndex);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    checkCudaErrors(cudaMemcpy(res, minValue, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(id, minIndex, sizeof(int), cudaMemcpyDeviceToHost));
    return *res;
}


template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initParticles() {
    particles = new double[size];
    _randInit(particles, size, 100.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::findPersonalBest() {
    compareAndAssign<<<block_for_particleNum, threads>>>(bestFuncVals, funcVals, d_particles, personal_best, particleNum, L);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::findGroupBest() {
    checkCudaErrors(cudaMemcpy(group_best, d_particles + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToDevice)); 
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initVelocity() {
    v = new double[size];
    _randInit(v, particleNum * L, 1.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initCuda() {
    checkCudaErrors(cudaMalloc((void **) &d_particles, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &funcVals, particleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &tmpFuncVals, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &bestFuncVals, particleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &personal_best, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &group_best, L * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_v, size * sizeof(double)));
    cudaStream_t stream1, stream2, stream3;
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaMemcpyAsync(d_particles, particles, size * sizeof(double), cudaMemcpyHostToDevice, stream1));
    cudaStreamCreate(&stream2);
    checkCudaErrors(cudaMemcpyAsync(personal_best, particles, size * sizeof(double), cudaMemcpyHostToDevice, stream2));
    cudaStreamCreate(&stream3);
    checkCudaErrors(cudaMemcpyAsync(d_v, v, size * sizeof(double), cudaMemcpyHostToDevice, stream3));
    checkCudaErrors(cudaMalloc((void **) &tmp_v, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &tmp_v1, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &v_norm, particleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &minValue, sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &minIndex, sizeof(int)));
    gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_updateParticles() {
    updateParticles<<<blocks, threads>>>(d_particles, d_v, dt, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_updateVelocity() {
    double r1 = _randGen();
    double r2 = _randGen();
    const double coef1 = c1 * r1 /dt;
    const double coef2 = c2 * r2 /dt;
    const double a = 1.0 * coef1;
    const double b = -1.0 * coef1;
    Scale<<<blocks, threads>>>(d_v, w, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Minus<<<blocks, threads>>>(personal_best, d_particles, tmp_v, coef1, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Add<<<blocks, threads>>>(d_v, tmp_v, tmp_v1, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    minusGroupBest<<<blocks, threads>>>(d_particles, tmp_v, group_best, coef2, L, particleNum);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Add<<<blocks, threads>>>(tmp_v, tmp_v1, d_v, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::evaluate() {
    int dim = L;
    function<<<blocks, threads>>>(d_particles, tmpFuncVals, dim, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    SumRowwise<<<block_for_particleNum, threads>>>(tmpFuncVals, funcVals, particleNum, dim);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
double ParticalSwarmOptimizationGPU<L>::_randGen() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distribution(0, 1);
    return distribution(eng);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_randInit(double * data, int size, double scale) {
    for (int i = 0; i < size; i++){
        data[i] = scale * _randGen();
    }
}

template<std::size_t L>
double ParticalSwarmOptimizationGPU<L>::getOptimal() {
    return optimal;
}

template<std::size_t L>
double* ParticalSwarmOptimizationGPU<L>::getSol() {
    return sol;
}

__global__ void function(double * a, double * res, int dim, int size){
    // printf("Entered\n");
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    double tmp = a[index];
    if (index < size){
        res[index] = tmp * (tmp - (double)(index % dim + 1));
    }
}

__global__ void updateParticles(double * a, double * v, double dt, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        a[index] = a[index] + v[index] * dt;
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

__global__ void compareAndAssign(double * bestFuncVals, double * funVals, double *particles, double *pb, int pNum, int dim){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < pNum){
        if(funVals[index] < bestFuncVals[index]) {
            bestFuncVals[index] = funVals[index];
            for(int i = 0; i < dim; i++){
                pb[index * dim + i] = particles[index * dim + i];
            }
        }
    }
}

__global__ void minusGroupBest(double * particles, double * tp, double * group_best, double factor, int dim, int pNum){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < pNum){
        for(int i = 0; i < dim; i++){
            tp[index * dim + i] = factor * (group_best[i] - particles[index * dim + i]);
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


#endif /* B951BB41_D03D_4461_AB77_CCD976C9C276 */
