#ifndef B951BB41_D03D_4461_AB77_CCD976C9C276
#define B951BB41_D03D_4461_AB77_CCD976C9C276
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

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

#endif /* B951BB41_D03D_4461_AB77_CCD976C9C276 */