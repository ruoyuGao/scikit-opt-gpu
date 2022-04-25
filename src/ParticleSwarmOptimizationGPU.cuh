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


// template<typename T>
// using func_t = double (*) (double *);

__global__ void function(double * a, double * res, int dim, int size);
__global__ void updateParticles(double * a, double * v, double dt, int size);
__global__ void print(double * a, int size);
__global__ void Ones(double * a, int size);
__global__ void compareAndAssign(double * bestFuncVals, double * funVals, double *particles, double *pb, int pNum, int dim);
template<std::size_t L>
class ParticalSwarmOptimizationGPU{
public:
    ParticalSwarmOptimizationGPU(int NumParticles=1e5, int MaxIters=1500, double C1=2.5, double C2=0.5, double weight=0.5, double tolerence=1e-10, double delta_t=0.01, int threadNum=256);
    void run();
    double getOptimal();
    double * getSol();
    // void printParticles(){std::cout << particles << std::endl;}
    // void printV(){std::cout << v << std::endl;}
    // __device__ void function(T(*func)(double *), double * input, double * result);

private:
    double c1;
    double c2;
    double w;
    double tol;
    int maxIter;
    double dt;
    int particleNum;
    double * particles;
    double * d_particles;
    double * personal_best;
    double * personal_best_func_values;
    double * current_func_values;
    double * group_best;
    double * v;
    double * d_v;
    double * tmp_v;
    double * tmp_v1;
    double * funcVals;
    double * tmpFuncVals;
    double * bestFuncVals;
    double * sumVec;
    double * colOnes;
    double optimal;
    double *sol;
    int * sol_id;
    int size;
    const double alpha = 1.;
    const double beta = 0.;

    cublasHandle_t handle;
    int threads;
    int blocks, block_for_fun_vals;
    // double (*function)(double *);
    void _initParticles();
    void _initVelocity();
    void _initCuda();
    void _updateVelocity();
    void _updateParticles();
    void _updateCurrentFuncValues();
    double _randGen();
    void _randInit(double * data, int size, double scale);
    void findPersonalBest();
    void findGroupBest();
    void evaluate();  
    double minFunVal(int * id);  
};

// template<std::size_t L>
template<std::size_t L>
ParticalSwarmOptimizationGPU<L>::ParticalSwarmOptimizationGPU(int NumParticles, int MaxIters, double C1, double C2, double weight, double tolerence, double delta_t, int threadNum){
    // function = func;
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
    block_for_fun_vals = particleNum / threads + 1;
    sol = new double[L];
    sol_id = new int;
    _initParticles();
    _initVelocity();
    _initCuda();
    evaluate();
    checkCudaErrors(cudaMemcpy(bestFuncVals, funcVals, particleNum * sizeof(double), cudaMemcpyDeviceToDevice)); 
    // bestFuncVals = funcVals;
    optimal = minFunVal(sol_id);   
    printf("optimal = %lf, minId = %d pos=%d\n", optimal, *sol_id, (*sol_id) * L);
    checkCudaErrors(cudaMemcpy(group_best, d_particles + (*sol_id) * L, L * sizeof(double), cudaMemcpyDeviceToDevice)); 
    checkCudaErrors(cudaMemcpy(sol, group_best, L * sizeof(double), cudaMemcpyDeviceToHost)); 
    // printf("group best\n");
    // print<<<blocks,threads>>>(group_best, L);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::run(){
    int minId;
    double tmpOptimal;
    _updateParticles();
    // printf("Particles After\n");
    // print<<<blocks,threads>>>(d_particles, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    for(int i = 0; i < maxIter; ++i){
        evaluate();
        // printf("Group Best\n");
        // print<<<blocks,threads>>>(group_best, L);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
        tmpOptimal = minFunVal(&minId);
        double diff = abs(optimal - tmpOptimal);
        if(diff < tol){
            optimal = tmpOptimal;
            checkCudaErrors(cudaMemcpy(sol, d_particles + minId * L, L * sizeof(double), cudaMemcpyDeviceToHost)); 
            break;
        }
        if(tmpOptimal < optimal){
            optimal = tmpOptimal;
            checkCudaErrors(cudaMemcpy(sol, d_particles + minId * L, L * sizeof(double), cudaMemcpyDeviceToHost)); 
        }
        _updateVelocity();
        // printf("Particles After\n");
        // print<<<blocks,threads>>>(d_particles, size);
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );
        _updateParticles();
        findPersonalBest();
        findGroupBest();
        printf("Iter%d: minFunVal=%lf x=[", i, optimal);
        for(int j = 0; j < L-1; j++){
            printf("%lf ", sol[j]);
        }
        printf("%lf]\n", sol[L-1]);
    }
    
}

template<std::size_t L>
double ParticalSwarmOptimizationGPU<L>::minFunVal(int * id) {
    CUBLAS_CHECK(cublasIdamin(handle, particleNum, funcVals, 1, id));
    double * res = new double;
    (*id)-=1;
    // printf("MINID=%d\n", *id);
    checkCudaErrors(cudaMemcpy(res, funcVals+(*id), sizeof(double), cudaMemcpyDeviceToHost));
    return *res;
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initParticles() {
    particles = new double[size];
    _randInit(particles, size, 100.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::findPersonalBest() {
    compareAndAssign<<<block_for_fun_vals, threads>>>(bestFuncVals, funcVals, d_particles, personal_best, particleNum, L);
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
    //TODO: Use cudaMemcpyAsync
    checkCudaErrors(cudaMalloc((void **) &d_particles, size * sizeof(double)));
    // printf("d_particles = \n");
    // print<<<blocks, threads>>>(d_particles, size);
    checkCudaErrors(cudaMalloc((void **) &funcVals, particleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &tmpFuncVals, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &sumVec, L * sizeof(double)));
    Ones<<<blocks, threads>>>(sumVec, L);
    checkCudaErrors(cudaMalloc((void **) &colOnes, particleNum * sizeof(double)));
    Ones<<<blocks, threads>>>(colOnes, particleNum);
    checkCudaErrors(cudaMalloc((void **) &bestFuncVals, particleNum * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &personal_best, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &group_best, L * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_v, size * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_particles, particles, size * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(personal_best, particles, size * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, v, size * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &tmp_v, size * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &tmp_v1, size * sizeof(double)));
    CUBLAS_CHECK(cublasCreate(&handle));
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
    const double c = -1.0;
    // printf("coef1=%lf coef2=%lf\n Velocity before:\n", coef1, coef2);
    // print<<<blocks,threads>>>(d_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // d_v = d_v * w
    CUBLAS_CHECK(cublasDscal(handle, size, &w, d_v, 1));
    // printf("d_v * w\n");
    // print<<<blocks,threads>>>(d_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // tmp_v = coef1 * (personal_best-d_particles)
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, particleNum, L, &a, personal_best, particleNum, &b, d_particles, particleNum, tmp_v, particleNum));
    // printf("personal best\n");
    // print<<<blocks,threads>>>(personal_best, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("d_particles\n");
    // print<<<blocks,threads>>>(d_particles, size);
    // gpuErrchk( cudaPeekAtLastError());
    // gpuErrchk( cudaDeviceSynchronize() );
    // tmp_v1 = d_v + tmp_v  now tmp_v is free to use
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, particleNum, L, &alpha, d_v, particleNum, &alpha, tmp_v, particleNum, tmp_v1, particleNum));
    // copy d_particles to tmp_v and tranpose it (col major for next step)
    // checkCudaErrors(cudaMemcpy(tmp_v, d_particles, size * sizeof(double), cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, particleNum, L, &alpha, d_particles, L, &beta, d_particles, L, tmp_v, particleNum));
    // printf("tmp_v1\n");
    // print<<<blocks,threads>>>(tmp_v1, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("group best\n");
    // print<<<blocks,threads>>>(group_best, L);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // tmp_v = tmp_v - group_best (rowwise)
    CUBLAS_CHECK(cublasDger(handle, particleNum, L, &c, colOnes, 1, group_best, 1, tmp_v, particleNum));
    // printf("tmp_v - group best\n");
    // print<<<blocks,threads>>>(tmp_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    //  tmp_v *=-1
    CUBLAS_CHECK(cublasDscal(handle, size, &c, tmp_v, 1));
    // printf("tmp_v *= -1\n");
    // print<<<blocks,threads>>>(tmp_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize());
    // d_v = tmp_v1 + coef2 * tmp_v
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, particleNum, L, &alpha, tmp_v1, particleNum, &coef2, tmp_v, L, d_v, particleNum));
    // printf("Velocity After tmp_v1 + coef2 * tmp_v\n");
    // print<<<blocks,threads>>>(d_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::evaluate() {
    int dim = L;
    function<<<blocks, threads>>>(d_particles, tmpFuncVals, dim, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T, L, particleNum, &alpha, tmpFuncVals, L, sumVec, 1, &beta, funcVals, 1));
    // printf("funcVals\n");
    // print<<<blocks, threads>>>(funcVals, particleNum);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
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
        // printf("data[%d]=%lf\n", i, data[i]);
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
        // printf("index = %d tmp = %lf tmp1 = %lf res = %lf\n", index, tmp, (tmp - (double)(index % dim + 1)), res[index]);
    }
}

__global__ void updateParticles(double * a, double * v, double dt, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        // double tmp = a[index];
        a[index] = a[index] + v[index] * dt;
        // printf("index=%d, a before=[%lf], v=%lf, dt=%lf, a after=[%lf]\n",index, tmp, v[index], dt, a[index]);
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
#endif /* B951BB41_D03D_4461_AB77_CCD976C9C276 */
