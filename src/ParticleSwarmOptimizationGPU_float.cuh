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
// using func_t = float (*) (float *);

__global__ void function(float * a, float * res, int dim, int size);
__global__ void updateParticles(float * a, float * v, float dt, int size);
__global__ void print(float * a, int size);
__global__ void Ones(float * a, int size);
__global__ void compareAndAssign(float * bestFuncVals, float * funVals, float *particles, float *pb, int pNum, int dim);
template<std::size_t L>
class ParticalSwarmOptimizationGPU{
public:
    ParticalSwarmOptimizationGPU(int NumParticles=1e5, int MaxIters=1500, float C1=2.5, float C2=0.5, float weight=0.5, float tolerence=1e-5, float delta_t=0.01, int threadNum=256);
    void run();
    float getOptimal();
    float * getSol();
    // void printParticles(){std::cout << particles << std::endl;}
    // void printV(){std::cout << v << std::endl;}
    // __device__ void function(T(*func)(float *), float * input, float * result);

private:
    float c1;
    float c2;
    float w;
    float tol;
    int maxIter;
    float dt;
    int particleNum;
    float * particles;
    float * d_particles;
    float * personal_best;
    float * personal_best_func_values;
    float * current_func_values;
    float * group_best;
    float * v;
    float * d_v;
    float * tmp_v;
    float * tmp_v1;
    float * funcVals;
    float * tmpFuncVals;
    float * bestFuncVals;
    float * sumVec;
    float * colOnes;
    float optimal;
    float *sol;
    int * sol_id;
    int size;
    const float alpha = 1.;
    const float beta = 0.;

    cublasHandle_t handle;
    int threads;
    int blocks, block_for_fun_vals;
    // float (*function)(float *);
    void _initParticles();
    void _initVelocity();
    void _initCuda();
    void _updateVelocity();
    void _updateParticles();
    void _updateCurrentFuncValues();
    float _randGen();
    void _randInit(float * data, int size, float scale);
    void findPersonalBest();
    void findGroupBest();
    void evaluate();  
    float minFunVal(int * id);  
};

// template<std::size_t L>
template<std::size_t L>
ParticalSwarmOptimizationGPU<L>::ParticalSwarmOptimizationGPU(int NumParticles, int MaxIters, float C1, float C2, float weight, float tolerence, float delta_t, int threadNum){
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
    sol = new float[L];
    sol_id = new int;
    _initParticles();
    _initVelocity();
    _initCuda();
    evaluate();
    checkCudaErrors(cudaMemcpy(bestFuncVals, funcVals, particleNum * sizeof(float), cudaMemcpyDeviceToDevice)); 
    // bestFuncVals = funcVals;
    optimal = minFunVal(sol_id);   
    printf("optimal = %lf, minId = %d pos=%d\n", optimal, *sol_id, (*sol_id) * L);
    checkCudaErrors(cudaMemcpy(group_best, d_particles + (*sol_id) * L, L * sizeof(float), cudaMemcpyDeviceToDevice)); 
    checkCudaErrors(cudaMemcpy(sol, group_best, L * sizeof(float), cudaMemcpyDeviceToHost)); 
    // printf("group best\n");
    // print<<<blocks,threads>>>(group_best, L);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::run(){
    int minId;
    float tmpOptimal;
    _updateParticles();
    // printf("Particles After\n");
    // print<<<blocks,threads>>>(d_particles, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    for(int i = 0; i < maxIter; ++i){
        evaluate();
        tmpOptimal = minFunVal(&minId);
        float diff = abs(optimal - tmpOptimal);
        if(diff < tol){
            optimal = tmpOptimal;
            checkCudaErrors(cudaMemcpy(sol, d_particles + minId * L, L * sizeof(float), cudaMemcpyDeviceToHost)); 
            break;
        }
        if(tmpOptimal < optimal){
            optimal = tmpOptimal;
            checkCudaErrors(cudaMemcpy(sol, group_best, L * sizeof(float), cudaMemcpyDeviceToHost)); 
        }
        _updateVelocity();
        _updateParticles();
        findPersonalBest();
        printf("Group Best\n");
        print<<<blocks,threads>>>(group_best, L);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        findGroupBest();
        printf("Iter%d: minFunVal=%lf x=[", i, optimal);
        for(int j = 0; j < L-1; j++){
            printf("%lf ", sol[j]);
        }
        printf("%lf]\n", sol[L-1]);
    }
}

template<std::size_t L>
float ParticalSwarmOptimizationGPU<L>::minFunVal(int * id) {
    CUBLAS_CHECK(cublasIsamin(handle, particleNum, funcVals, 1, id));
    float * res = new float;
    (*id)-=1;
    checkCudaErrors(cudaMemcpy(res, funcVals+(*id), sizeof(float), cudaMemcpyDeviceToHost));
    return *res;
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initParticles() {
    particles = new float[size];
    _randInit(particles, size, 100.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::findPersonalBest() {
    compareAndAssign<<<block_for_fun_vals, threads>>>(bestFuncVals, funcVals, d_particles, personal_best, particleNum, L);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::findGroupBest() {
    checkCudaErrors(cudaMemcpy(group_best, d_particles + (*sol_id) * L, L * sizeof(float), cudaMemcpyDeviceToDevice)); 
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initVelocity() {
    v = new float[size];
    _randInit(v, particleNum * L, 1.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initCuda() {
    //TODO: Use cudaMemcpyAsync
    checkCudaErrors(cudaMalloc((void **) &d_particles, size * sizeof(float)));
    // printf("d_particles = \n");
    // print<<<blocks, threads>>>(d_particles, size);
    checkCudaErrors(cudaMalloc((void **) &funcVals, particleNum * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &tmpFuncVals, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &sumVec, L * sizeof(float)));
    Ones<<<blocks, threads>>>(sumVec, L);
    checkCudaErrors(cudaMalloc((void **) &colOnes, particleNum * sizeof(float)));
    Ones<<<blocks, threads>>>(colOnes, particleNum);
    checkCudaErrors(cudaMalloc((void **) &bestFuncVals, particleNum * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &personal_best, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &group_best, L * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_v, size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_particles, particles, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(personal_best, particles, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &tmp_v, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &tmp_v1, size * sizeof(float)));
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
    float r1 = _randGen();
    float r2 = _randGen();
    const float coef1 = c1 * r1 /dt;
    const float coef2 = c2 * r2 /dt;
    const float a = 1.0 * coef1;
    const float b = -1.0 * coef1;
    const float c = -1.0;
    // printf("coef1=%lf coef2=%lf\n Velocity before:\n", coef1, coef2);
    // print<<<blocks,threads>>>(d_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // d_v = d_v * w
    CUBLAS_CHECK(cublasSscal(handle, size, &w, d_v, 1));
    // printf("d_v * w\n");
    // print<<<blocks,threads>>>(d_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // tmp_v = coef1 * (personal_best-d_particles)
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, particleNum, L, &a, personal_best, particleNum, &b, d_particles, particleNum, tmp_v, particleNum));
    // printf("personal best\n");
    // print<<<blocks,threads>>>(personal_best, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("d_particles\n");
    // print<<<blocks,threads>>>(d_particles, size);
    // gpuErrchk( cudaPeekAtLastError());
    // gpuErrchk( cudaDeviceSynchronize() );
    // tmp_v1 = d_v + tmp_v  now tmp_v is free to use
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, particleNum, L, &alpha, d_v, particleNum, &alpha, tmp_v, particleNum, tmp_v1, particleNum));
    // copy d_particles to tmp_v and tranpose it (col major for next step)
    // checkCudaErrors(cudaMemcpy(tmp_v, d_particles, size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, particleNum, L, &alpha, d_particles, L, &beta, d_particles, L, tmp_v, particleNum));
    // printf("tmp_v1\n");
    // print<<<blocks,threads>>>(tmp_v1, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // printf("group best\n");
    // print<<<blocks,threads>>>(group_best, L);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // tmp_v = tmp_v - group_best (rowwise)
    CUBLAS_CHECK(cublasSger(handle, particleNum, L, &c, colOnes, 1, group_best, 1, tmp_v, particleNum));
    // printf("tmp_v - group best\n");
    // print<<<blocks,threads>>>(tmp_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    //  tmp_v *=-1
    CUBLAS_CHECK(cublasSscal(handle, size, &c, tmp_v, 1));
    // printf("tmp_v *= -1\n");
    // print<<<blocks,threads>>>(tmp_v, size);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize());
    // d_v = tmp_v1 + coef2 * tmp_v
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, particleNum, L, &alpha, tmp_v1, particleNum, &coef2, tmp_v, L, d_v, particleNum));
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
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, L, particleNum, &alpha, tmpFuncVals, L, sumVec, 1, &beta, funcVals, 1));
    // printf("funcVals\n");
    // print<<<blocks, threads>>>(funcVals, particleNum);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
}

template<std::size_t L>
float ParticalSwarmOptimizationGPU<L>::_randGen() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distribution(0, 1);
    return distribution(eng);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_randInit(float * data, int size, float scale) {
    for (int i = 0; i < size; i++){
        data[i] = scale * _randGen();
        // printf("data[%d]=%lf\n", i, data[i]);
    }
}

template<std::size_t L>
float ParticalSwarmOptimizationGPU<L>::getOptimal() {
    return optimal;
}

template<std::size_t L>
float* ParticalSwarmOptimizationGPU<L>::getSol() {
    return sol;
}

__global__ void function(float * a, float * res, int dim, int size){
    // printf("Entered\n");
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    float tmp = a[index];
    if (index < size){
        res[index] = tmp * (tmp - (float)(index % dim + 1));
        // printf("index = %d tmp = %lf tmp1 = %lf res = %lf\n", index, tmp, (tmp - (float)(index % dim + 1)), res[index]);
    }
}

__global__ void updateParticles(float * a, float * v, float dt, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        // float tmp = a[index];
        a[index] = a[index] + v[index] * dt;
        // printf("index=%d, a before=[%lf], v=%lf, dt=%lf, a after=[%lf]\n",index, tmp, v[index], dt, a[index]);
    }
}

__global__ void print(float * a, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        printf("index[%d]=%lf\n", index, a[index]);
    }
}

__global__ void Ones(float * a, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        a[index] = 1.0;
    }
}

__global__ void compareAndAssign(float * bestFuncVals, float * funVals, float *particles, float *pb, int pNum, int dim){
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
