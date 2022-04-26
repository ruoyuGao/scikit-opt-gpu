#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include<math.h>
#include<curand_kernel.h>// this lib shoulb be included
#include<cuda_runtime.h>
#include"device_launch_parameters.h"
#include<random>
#define CHECK(res) if(res!=cudaSuccess){printf("CUDA Error: %s\n", cudaGetErrorString(res));exit(-1);}
//-------------------generate random numbers-------//
__device__ int selection(float rand_number, int pop_size);
__device__ float4 generate(curandStatePhilox4_32_10_t *globalState, int ind);
__device__ float get_fitness_gpu(float* X_cuda, int row_index, int n_dim);
__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed);
//-------------This is our kernel function where the random numbers generated and do GA here------//
__global__ void our_kernel(curandStatePhilox4_32_10_t *globalState, int pop_size, int n_dim, float cross_prob, float mutation_prob, float* X_cuda, float* lb, float* ub, int block_size, int offset);
float get_random(float low, float up);
void run();
void get_fitness(float* X, float* fitness, int pop_size, int n_dim);
void init_population(float** X, int pop_size, int n_dim);
int main(int argc, char *argv[])
{   
    printf("Here is GA, first parameter: iteration, second parameter: pop_size\n");
    cudaError_t res;
    printf("parameter seq: iteration pop_size cross_prob, mutate_prob");
    if(argc!= 4){
        printf("wrong number of input size\n");
    }
    int iteration = atoi(argv[1]);
    int pop_size =  atoi(argv[2]);
    float cross_prob = atof(argv[3]);
    float mutation_prob = atof(argv[4]);
    //int iteration = 100000;
    //int pop_size = 1024;
    int block_size = 128;
    int n_dim = 3;
    //float cross_prob = 0.7;
    //float mutation_prob = 0.01;
    //init X
    // float** X= NULL;
    float* X_cuda_row= NULL;
    float* X_one_dim= NULL;
    //printf("here38\n");
    // X = (float **)calloc(pop_size*sizeof(float*));
    //printf("here40\n");
    X_one_dim = (float*)calloc(n_dim*pop_size, sizeof(float));
    //printf("here42\n");
    res=cudaMalloc((void**)&X_cuda_row, pop_size*n_dim*sizeof(float));CHECK(res);
    
    //printf("here45\n");
    // for(int i=0;i<pop_size;i++){
    //     X[i] = (float*)calloc(sizeof(float)*n_dim);
    // }
    //printf("here49\n");
    // init_population(X,pop_size,n_dim);
    for(int i=0; i < pop_size * n_dim; i++){
        X_one_dim[i]=get_random(-3.0, 5.0);
    }
    // for(int i=0;i<pop_size;i++){
    //     for(int j=0;j<n_dim;j++){
    //         X_one_dim[i*pop_size+j] = X[i][j];
    //         //printf("%f\n",X_one_dim[i*pop_size+j]);
    //     }
    // }
    //get fitness
    float *fitness= NULL;
    //printf("here60\n");
    fitness = (float *)calloc(pop_size, sizeof(float));
    // for(int i=0;i<pop_size*n_dim;i++){
    //     printf("%f ", X_one_dim[i]);
    // }
    //printf("here65\n");
    res=cudaMemcpy(X_cuda_row,X_one_dim, pop_size*n_dim*sizeof(float), cudaMemcpyHostToDevice);CHECK(res);
    //cudaMemcpy(X_cuda, X_new, pop_size*sizeof(float*), cudaMemcpyHostToDevice);
    //res = cudaMalloc((void**)(&X_cuda_row), pop_size*n_dim*sizeof(float));CHECK(res);
    //get_fitness(X,fitness,pop_size,n_dim);
    //init lb and ub
    float *lb= NULL;
    float *ub= NULL;
    float *lb_cuda;
    float *ub_cuda;
    //printf("here73\n");
    lb = (float *)calloc(n_dim, sizeof(float));
    //printf("here75\n");
    ub = (float *)calloc(n_dim, sizeof(float));
    //printf("here77\n");
    res=cudaMalloc((void**)&lb_cuda, sizeof(float)*n_dim);CHECK(res);
    //printf("here79\n");
    res=cudaMalloc((void**)&ub_cuda, sizeof(float)*n_dim);CHECK(res);
    //printf("here81\n");
    //set lb and ub
    
    for(int i=0;i<n_dim;i++){
        lb[i] = -10.0;
        ub[i] = 10.0;
    }
    //printf("here88\n");
    cudaMemcpy(lb_cuda, lb, n_dim*sizeof(float), cudaMemcpyHostToDevice);
    //printf("here90\n");
    cudaMemcpy(ub_cuda, ub, n_dim*sizeof(float), cudaMemcpyHostToDevice);
    //set random kernel
    
    int N = block_size;
    curandStatePhilox4_32_10_t* devStates;
    //printf("here96\n");
    cudaMalloc(&devStates, N * sizeof(curandStatePhilox4_32_10_t));
    srand(time(0));
    int seed = rand();
    //  Initialize the states
    setup_kernel <<<1, N>>> (devStates, seed);
    for(int iter =0;iter<iteration;iter++){
        
        int block_number = pop_size/block_size;
        for(int offset=0;offset<block_number;offset++){
            //printf("%d\n",offset);
            our_kernel <<<1, N>>> (devStates,pop_size, n_dim, cross_prob, mutation_prob, X_cuda_row,lb_cuda, ub_cuda,block_size,offset);
        }
        
        // if(iter%100==0){
        //     printf("max x: %f", X_one_dim[max_index*n_dim+0]);
        //     printf("max_fitness: %f\n",max_fitness);
        // }
        
    }
    cudaDeviceSynchronize();
    cudaMemcpy(X_one_dim, X_cuda_row, pop_size*n_dim*sizeof(float), cudaMemcpyDeviceToHost);
        // for(int i=0;i<pop_size*n_dim;i++){
        //     printf("%f ", X_one_dim[i]);
        // }
    cudaDeviceSynchronize();
    get_fitness(X_one_dim,fitness,pop_size,n_dim);
    int max_index =0;
    float max_fitness = 0;
    for(int i=0;i<pop_size;i++){
        if(fitness[i]>max_fitness){
            max_index = i;
            max_fitness = fitness[i];
        }
    }
    printf("max x: %f", X_one_dim[max_index*n_dim+0]);
    printf("max_fitness: %f\n",max_fitness);
    
    
    
    //printf("here\n");
    cudaFree(X_cuda_row);
    //printf("here1\n");
    cudaFree(lb_cuda);
    //printf("here2\n");
    cudaFree(ub_cuda);
    //printf("here3\n");
    free(fitness);
    //printf("here4\n");
    // free(X);
    //printf("here5\n");
    //free(X_one_dim);
    //printf("here6\n");
    free(lb);
    //printf("here7\n");
    free(ub);
    //printf("here8\n");
	cudaDeviceReset();
    //printf("here9\n");
    return 0;
}

__device__ float4 generate(curandStatePhilox4_32_10_t *globalState, int ind)
{
	curandStatePhilox4_32_10_t localState = globalState[ind];
	float4 res = curand_uniform4(&localState);// uniform distribution
	globalState[ind] = localState;
	return res;
}

__device__ int selection(float rand_number, int pop_size){
    int res = (int)(rand_number*pop_size)%pop_size;
    return res;
}

__device__ float get_fitness_gpu(float* X_cuda, int row_index, int n_dim){
    float res = X_cuda[row_index*n_dim+0] + 10* sin(5*X_cuda[row_index*n_dim+0]) +7* cos(4* X_cuda[row_index*n_dim+0])+X_cuda[row_index*n_dim+1]+X_cuda[row_index*n_dim+2];
    return res;
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);// initialize the state
}

//-------------This is our kernel function where the random numbers generated------//
__global__ void our_kernel(curandStatePhilox4_32_10_t *globalState, int pop_size, int n_dim, float cross_prob,float mutation_prob, float* X_cuda, float* lb, float* ub, int block_size, int offset)
{
	int idx = offset*block_size + blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%f\n",lb[1]);
    //random_first.x is selection index, random_first.y is cross prob,random_first.z is cross dimension, random_first.w is mutation prob
    float4 random_first = generate(globalState, idx);
    // random_second.x is mutation_index, random_second.y is the mutation result on this index
    float4 random_second = generate(globalState, idx);

    int paraent_index = selection(abs(random_first.x), pop_size);
    __syncthreads();
    //crossover
    // printf("idx: %d\n",idx);
    // printf("par_idx: %d\n",paraent_index);
    __syncthreads();
    if(idx<pop_size && abs(random_first.y) < cross_prob){
        float child1, child2;
        float fitness_father = get_fitness_gpu(X_cuda,idx,n_dim);
        float fitness_mother = get_fitness_gpu(X_cuda,paraent_index,n_dim);
        int cross_index = (int)(abs(random_first.z)*n_dim)%n_dim;
        child1 = X_cuda[idx*n_dim+cross_index];
        child2 = X_cuda[paraent_index*n_dim+cross_index];
        X_cuda[idx*n_dim+cross_index] = 0.7*X_cuda[idx*n_dim+cross_index]+ 0.3* X_cuda[paraent_index*n_dim+cross_index];
        X_cuda[paraent_index*n_dim+cross_index] = 0.3*X_cuda[idx*n_dim+cross_index]+ 0.7* X_cuda[paraent_index*n_dim+cross_index];
        float fitness_child1 = get_fitness_gpu(X_cuda,idx,n_dim);
        float fitness_child2 = get_fitness_gpu(X_cuda,paraent_index,n_dim);
        __threadfence();
        
        if(fitness_child1<fitness_father){
            // printf("fitness_father: %f\n",  fitness_father);
            // printf("fitness_child1: %f\n",  fitness_child1);
            X_cuda[idx*n_dim+cross_index]=child1;
        }
        if(fitness_child2<fitness_mother){
            // printf("fitness_mother: %f\n",  fitness_mother);
            // printf("fitness_child2: %f\n",  fitness_child2);
            X_cuda[paraent_index*n_dim+cross_index] = child2;
        }
        __threadfence();
        

        //mutation
        __threadfence();
        if(abs(random_first.w)<mutation_prob){
            int mutate_index = (int)(abs(random_second.x)*n_dim)%n_dim;
            //printf("mutate_lb: %f\n", lb[mutate_index]);
            //printf("mutate_res1: %f\n",abs(random_second.y));
            //printf("mutate_res: %f\n",abs(random_second.y)*(ub[mutate_index]-lb[mutate_index]) + lb[mutate_index]);
            X_cuda[idx*n_dim+mutate_index] = abs(random_second.y)*(ub[mutate_index]-lb[mutate_index]) + lb[mutate_index];
        }
        __threadfence();
        //printf("here\n");
        // for(int i=0;i<n_dim;i++){
        //     //printf("%f\n",X_cuda[idx][i]);
        //     printf("%f\n",X_cuda[idx*n_dim+i]);
        // }
    }
    
    // if(abs(random_first.y)< cross_prob){
    //     printf("here success\n");
    // }
}

float get_random(float low, float up){
    float res = low + 1.0 * ( rand() % RAND_MAX ) / RAND_MAX * ( up - low );
    return res;
}

void get_fitness(float* X, float* fitness, int pop_size, int n_dim){
    //printf("%d\n",pop_size);
    for(int m=0;m<pop_size;m++){
        //printf("%d\n",m);
        fitness[m] = X[m*n_dim+0] + 10* sin(5*X[m*n_dim+0]) +7* cos(4* X[m*n_dim+0])+X[m*n_dim+1]+X[m*n_dim+2];
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