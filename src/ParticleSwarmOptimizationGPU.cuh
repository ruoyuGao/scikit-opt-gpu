#ifndef B951BB41_D03D_4461_AB77_CCD976C9C276
#define B951BB41_D03D_4461_AB77_CCD976C9C276
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <iostream>
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
#include <src/functions.cuh>

// template<typename T>
// using func_t = float (*) (float *);

// template<typename T>

// template<std::size_t L>
template<std::size_t L>
class ParticalSwarmOptimizationGPU{
public:
    ParticalSwarmOptimizationGPU(int NumParticles=1e5, int MaxIters=1500, float C1=2.5, float C2=0.5, float weight=0.5, float tolerence=1e-10, float delta_t=0.01, int threadNum=256);
    void run();
    float * getOptimal();
    float getSol();
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
    float * funcVals;
    float * tmpFuncVals;
    float * bestFuncVals;
    float * sumVec;
    float optimal;
    float *sol;
    int * sol_id;
    int size;
    const float alpha = 1.;
    const float beta = 0.;

    cublasHandle_t handle;
    int threads;
    int blocks;
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
    _initParticles();
    _initVelocity();
    _initCuda();
    evaluate();
    bestFuncVals = funcVals;
    // CUBLAS_CHECK(cublasIsamin(handle, particleNum, funcVals, 1, &sol_id));
    optimal = minFunVal(sol_id);   
    checkCudaErrors(cudaMemcpy(group_best, d_particles + (*sol_id) * L, L * sizeof(float), cudaMemcpyDeviceToDevice)); 
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::run(){
    int minId;
    float tmpOptimal;
    _updateParticles();
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
        }
    }
}

template<std::size_t L>
float ParticalSwarmOptimizationGPU<L>::minFunVal(int * id) {
    CUBLAS_CHECK(cublasIsamin(handle, particleNum, funcVals, 1, id));
    float * res = new float;
    checkCudaErrors(cudaMemcpy(res, funcVals+(*id), sizeof(float), cudaMemcpyDeviceToHost));
    return *res;
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initParticles() {
    particles = new float[size];
    _randInit(particles, particleNum * L, 100.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initVelocity() {
    v = new float[size];
    _randInit(particles, particleNum * L, 1.0);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_initCuda() {
    checkCudaErrors(cudaMalloc((void **) &d_particles, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &funcVals, particleNum * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &tmpFuncVals, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &sumVec, L * sizeof(float)));
    checkCudaErrors(cudaMemset(sumVec, 1.0, L * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &bestFuncVals, particleNum * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &personal_best, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &group_best, L * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_v, size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_particles, particles, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(personal_best, particles, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasCreate(&handle));
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::_updateParticles() {
    updateParticles<<<blocks, threads>>>(d_particles, d_v, dt, size);
}

template<std::size_t L>
void ParticalSwarmOptimizationGPU<L>::evaluate() {
    function<<<blocks, threads>>>(d_particles, tmpFuncVals, size);
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, particleNum, L, &alpha, tmpFuncVals, particleNum, sumVec, 1, &beta, funcVals, 1));
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
    }
}

// template<typename T>

#endif /* B951BB41_D03D_4461_AB77_CCD976C9C276 */
