#ifndef B951BB41_D03D_4461_AB77_CCD976C9C276
#define B951BB41_D03D_4461_AB77_CCD976C9C276
#include <src/Optimizer.h>
#include <cuda_runtime.h>
#include <src/cublas_utils.h>
#include <cublas_v2.h>
#include <src/helper_cuda.h>
#include <cuda.h>


template<typename T>
using func_t = T (*) (T*);

template<typename T>
__global__ void function(T * a, T * res, int size);

template<typename T>
__global__ void updateParticles(T * a, T * v, T* dt, int size);

template<typename T, std::size_t L>
class ParticalSwarmOptimizationGPU: public Optimizer{
public:
    ParticalSwarmOptimizationGPU(int NumParticles=1e5, int MaxIters=1500, T C1=2.5, T C2=0.5, T weight=0.5, T tolerence=1e-10, T delta_t=0.01, int threadNum=256);
    void run() override;
    T* getOptimal();
    T getSol();
    // void printParticles(){std::cout << particles << std::endl;}
    // void printV(){std::cout << v << std::endl;}
    // __device__ void function(T(*func)(T*), T* input, T* result);

private:
    T c1;
    T c2;
    T w;
    T tol;
    int maxIter;
    T dt;
    int particleNum;
    T sol;
    T* particles;
    T * d_particles;
    T* personal_best;
    T* personal_best_func_values;
    T* current_func_values;
    T* group_best;
    T* v;
    T * d_v;
    T * funcVals;
    T * bestFuncVals;
    T optimal;
    T * sol;
    int size;
    cublasHandle_t handle;
    int threads;
    int blocks;
    // std::function<T(Eigen::Vector<T,L>&)> function;
    T (*function)(T*);
    void _initParticles();
    void _initVelocity();
    void _initCuda();
    void _updateVelocity();
    void _updateParticles();
    void _updateCurrentFuncValues();
    T _randGen();
    void _randInit(T * data, int size, T scale);
    void findPersonalBest();
    void findGroupBest();
    // Eigen::Vector<T, Eigen::Dynamic> evaluate(Eigen::Matrix<T, Eigen::Dynamic, L>);
    void evaluate();
    Eigen::Vector<T, L> optimal;
};

template<typename T, std::size_t L>
ParticalSwarmOptimizationGPU<T,L>::ParticalSwarmOptimizationGPU(int NumParticles, int MaxIters, T C1, T C2, T weight, T tolerence, T delta_t, int threadNum):Optimizer(){
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
    CUBLAS_CHECK()
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T,L>::_initParticles() {
    particles = new T[size];
    _randInit(particles, particleNum * L, 100.0);
    
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T, L>::_initVelocity() {
    v = new T[size];
    _randInit(particles, particleNum * L, 1.0);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T, L>::_initCuda() {
    checkCudaErrors(cudaMalloc((void **) &d_particles, size * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **) &funcVals, particleNum * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **) &bestFuncVals, particleNum * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **) &personal_best, size * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **) &d_v, size * sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_particles, particles, size * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(personal_best, particles, size * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, v, size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cublasCreate(&handle));
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T,L>::updateParticles() {
    particles = new T[size];
    _randInit(particles, particleNum * L, 100.0);
    
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T, L>::evaluate() {
    function<<<blocks, threads>>>(d_particles, funcVals, size);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T,L>::run(){}

template<typename T, std::size_t L>
T ParticalSwarmOptimizationGPU<T, L>::_randGen() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<T> distribution(0, 1);
    return distribution(eng);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimizationGPU<T, L>::_randInit(T * data, int size, T scale) {
    for (int i = 0; i < size; i++){
        data[i] = scale * _randGen();
    }
}

template<typename T>
__global__ void function(T * a, T * res, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    T tmp = a[index];
    if (index < size){
        res[index] = tmp * (tmp - (tmp % 5) + 1);
    }
}

template<typename T>
__global__ void updateParticles(T * a, T * v, T* dt, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        a[index] = a[index] + v[index] * dt;
    }
}

#endif /* B951BB41_D03D_4461_AB77_CCD976C9C276 */
