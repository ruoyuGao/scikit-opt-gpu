#include <cuda.h>

__global__ void function(float * a, float * res, int size);
// __global__ void rowwiseSum(float * res, int size)
// template<typename T>
__global__ void updateParticles(float * a, float * v, float * dt, int size);

__global__ void function(float * a, float * res, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    float tmp = a[index];
    if (index < size){
        res[index] = tmp * (tmp - (index % 5) + 1);
    }
}

// template<typename T>
__global__ void updateParticles(float * a, float * v, float dt, int size){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < size){
        a[index] = a[index] + v[index] * dt;
    }
}
