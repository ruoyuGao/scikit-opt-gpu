#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include<math.h>
#include<curand_kernel.h>// this lib shoulb be included
#include<cuda_runtime.h>
#include"device_launch_parameters.h"
#include<random>
#include<src/GAgpu.cuh>
#define CHECK(res) if(res!=cudaSuccess){printf("CUDA Error: %s\n", cudaGetErrorString(res));exit(-1);}
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

