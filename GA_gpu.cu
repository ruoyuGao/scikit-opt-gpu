#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include<math.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"curand_kernel.h"// this lib shoulb be included
#include<ctime>
#include<iostream>
#include<random>
#define CHECK(res) if(res!=cudaSuccess){printf("CUDA Error: %s\n", cudaGetErrorString(res));exit(-1);}
//-------------------generate random numbers-------//
__device__ int selection(float rand_number, int pop_size);
__device__ float2 generate(curandStatePhilox4_32_10_t *globalState, int ind);
__device__ float get_fitness_gpu(float** X_cuda, int row_index);
__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed);
//-------------This is our kernel function where the random numbers generated------//
__global__ void our_kernel(curandStatePhilox4_32_10_t *globalState, int pop_size, int n_dim, float cross_prob, float mutation_prob, float** X_cuda, float* lb, float* ub, int block_size,float* x_cuda_row);
float get_random(float low, float up);
void run();
void get_fitness(float** X, float* fitness, int pop_size, int n_dim);
void init_population(float** X, int pop_size, int n_dim);
int main()
{   
    printf("Here is GA, first parameter: iteration, second parameter: pop_size\n");
    cudaError_t res;
    int iteration = 100;
    int pop_size = 4;
    int n_dim = 3;
    float cross_prob = 0.7;
    float mutation_prob = 0.1;
    //init X
    float** X;
    float** X_new;
    float** X_cuda;
    float* X_cuda_row;
    float* X_one_dim;
    float* X_res;
    X = (float **)malloc(pop_size*sizeof(float*));
    X_new = (float **)malloc(pop_size*sizeof(float*));
    X_one_dim = (float*)malloc(sizeof(float)*n_dim*pop_size);
    X_res = (float*)malloc(sizeof(float)*n_dim*pop_size);
    
    res=cudaMalloc((void**)(&X_cuda), pop_size*sizeof(float*));CHECK(res);
    res=cudaMalloc((void**)(&X_cuda_row), pop_size*n_dim*sizeof(float));CHECK(res);
    
    for(int i=0;i<pop_size;i++){
        X[i] = (float*)malloc(sizeof(float)*n_dim);
    }
    for (int r = 0; r < pop_size; r++)
	{
		X_new[r] = X_cuda_row + r*n_dim;
	}
    init_population(X,pop_size,n_dim);
    for(int i=0;i<pop_size;i++){
        for(int j=0;j<n_dim;j++){
            X_one_dim[i*pop_size+j] = X[i][j];
        }
    }
    cudaMemcpy(X_cuda_row,X_one_dim, pop_size*n_dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(X_cuda, X_new, pop_size*sizeof(float*), cudaMemcpyHostToDevice);

    res = cudaMalloc((void**)(&X_cuda_row), pop_size*n_dim*sizeof(float));CHECK(res);
    //get_fitness(X,fitness,pop_size,n_dim);
    //init lb and ub
    float *lb, *ub;
    float *lb_cuda, *ub_cuda;
    lb = (float *)malloc(sizeof(float)*n_dim);
    ub = (float *)malloc(sizeof(float)*n_dim);
    res=cudaMalloc((void**)(&lb_cuda), pop_size*sizeof(float));CHECK(res);
    res=cudaMalloc((void**)(&ub_cuda), pop_size*sizeof(float));CHECK(res);

    //set lb and ub
    for(int i=0;i<n_dim;i++){
        lb[i] = -10;
        ub[i] = 10;
    }
    cudaMemcpy(lb_cuda, lb, pop_size*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(ub_cuda, ub, pop_size*sizeof(float*), cudaMemcpyHostToDevice);
    //set random kernel
    int N = pop_size;
    curandStatePhilox4_32_10_t* devStates;
	cudaMalloc(&devStates, N * sizeof(curandStatePhilox4_32_10_t));
	srand(time(0));
	int seed = rand();
    int block_size = 256;
	//  Initialize the states
	setup_kernel <<<1, N>>> (devStates, seed);
	our_kernel <<<1, N>>> (devStates,pop_size, n_dim, cross_prob, mutation_prob, X_cuda,lb_cuda, ub_cuda,block_size,X_cuda_row);
    
    cudaMemcpy(X_res, X_cuda_row, pop_size*n_dim*sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i=0;i<pop_size*n_dim;i++){
    //     printf("%f ", X_res[i]);
    // }
    //get fitness
    float *fitness;
    fitness = (float *)malloc(sizeof(float)*pop_size);
    get_fitness(X_new,fitness,pop_size,n_dim);
    int max_index =0;
    float max_fitness = 0;
    for(int i=0;i<pop_size;i++){
        if(fitness[i]>max_fitness){
            max_index = i;
            max_fitness = fitness[i];
        }
    }
    printf("max_fitness: %f\n",max_fitness);
    
    cudaFree(X_cuda);
    cudaFree(X_cuda_row);
    cudaFree(lb_cuda);
    cudaFree(ub_cuda);
    free(fitness);
    free(X);
    free(X_one_dim);
    free(lb);
    free(ub);
	cudaDeviceReset();
	return 0;
}

__device__ float2 generate(curandStatePhilox4_32_10_t *globalState, int ind)
{
	curandStatePhilox4_32_10_t localState = globalState[ind];
	float2 res = curand_normal2(&localState);// uniform distribution
	globalState[ind] = localState;
	return res;
}

__device__ int selection(float rand_number, int pop_size){
    int res = (int)(rand_number*pop_size)%pop_size;
    return res;
}

__device__ float get_fitness_gpu(float** X_cuda, int row_index){
    float res = X_cuda[row_index][0] + 10* sin(5*X_cuda[row_index][0]) +7* cos(4* X_cuda[row_index][0]);
    return res;
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);// initialize the state
}

//-------------This is our kernel function where the random numbers generated------//
__global__ void our_kernel(curandStatePhilox4_32_10_t *globalState, int pop_size, int n_dim, float cross_prob,float mutation_prob, float** X_cuda, float* lb, float* ub, int block_size,float* x_cuda_row)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //random_first.x is selection index, random_first.y is cross prob
    float2 random_first = generate(globalState, idx);
    //random_second.x is cross dimension, random_second.y is mutation prob
    float2 random_second = generate(globalState, idx);
    //random_third.x is mutation_index, random_third.y is the mutation result on this index
    float2 random_third = generate(globalState, idx);

    int paraent_index = selection(abs(random_first.x), pop_size);
    __syncthreads();
    //crossover
    if(idx<block_size && abs(random_first.y) < cross_prob){
        float child1, child2;
        float fitness_father = get_fitness_gpu(X_cuda,idx);
        float fitness_mother = get_fitness_gpu(X_cuda,idx);
        int cross_index = (int)(abs(random_second.x)*n_dim)%n_dim;
        child1 = X_cuda[idx][cross_index];
        child2 = X_cuda[paraent_index][cross_index];
        X_cuda[idx][cross_index] = 0.7*X_cuda[idx][cross_index]+ 0.3* X_cuda[paraent_index][cross_index];
        X_cuda[paraent_index][cross_index] = 0.3*X_cuda[idx][cross_index]+ 0.7* X_cuda[paraent_index][cross_index];
        float fitness_child1 = get_fitness_gpu(X_cuda,idx);
        float fitness_child2 = get_fitness_gpu(X_cuda,idx);
        if(fitness_child1<fitness_father){
            X_cuda[idx][cross_index]=child1;
        }
        if(fitness_child2<fitness_mother){
            X_cuda[paraent_index][cross_index] = child2;
        }
        // printf("fitness_father: %f",  fitness_father);
        // printf("fitness_child1: %f",  fitness_child1);

        //mutation
        if(abs(random_second.y)<mutation_prob){
            int mutate_index = (int)(abs(random_third.x)*n_dim)%n_dim;
            X_cuda[idx][mutate_index] = abs(random_third.y)*(ub[mutate_index]-lb[mutate_index]) + lb[mutate_index];
        }
        for(int i=0;i<n_dim;i++){
            printf("%f\n",X_cuda[idx][i]);
            printf("%f\n",x_cuda_row[idx*n_dim+i]);
        }
    }
    
    // if(abs(random_first.y)< cross_prob){
    //     printf("here success\n");
    // }
}

float get_random(float low, float up){
    float res = low + 1.0 * ( rand() % RAND_MAX ) / RAND_MAX * ( up - low );
    return res;
}

void run(){

}

void get_fitness(float** X, float* fitness, int pop_size, int n_dim){
    printf("%d\n",pop_size);
    for(int m=0;m<pop_size;m++){
        printf("%d\n",m);
        fitness[m] = X[m][0] + 10* sin(5*X[m][0]) +7* cos(4* X[m][0]);
    }
}

void init_population(float** X, int pop_size, int n_dim){
    //printf("in function:\n");
    for(int i=0;i<pop_size;i++){
        for(int j=0;j<n_dim;j++){
            X[i][j] = get_random(-3.0, 5.0);
            //printf("%f ", X[i][j]);
        }
        //printf("\n");
    } 
}