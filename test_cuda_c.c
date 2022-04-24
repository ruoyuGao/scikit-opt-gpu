#include <stdio.h>
#include <stdlib.h>
#include <time.h>
float get_random(float low, float up){
    float res = low + 1.0 * ( rand() % RAND_MAX ) / RAND_MAX * ( up - low );
    return res;
}

void init_population(float** X, int pop_size, int n_dim){
    printf("in function:\n");
    for(int i=0;i<pop_size;i++){
        for(int j=0;j<n_dim;j++){
            X[i][j] = get_random(-3.0, 5.0);
            printf("%f ", X[i][j]);
        }
        printf("\n");
    } 
}
int main(int argc, char *argv[]){
    printf("Here is GA, first parameter: iteration, second parameter: pop_size\n");
    // int iteration = atoi(argv[1]);
    // int pop_size = atoi(argv[2]);
    // int n_dim = atoi(argv[3]);
    int iteration = 100;
    int pop_size = 10;
    int n_dim = 3;

    float** X;
    X = (float **)malloc(pop_size*sizeof(float*));
    for(int i=0;i<pop_size;i++){
        X[i] = (float*)malloc(sizeof(float)*n_dim);
    }
    init_population(X,pop_size,n_dim);
    printf("out function\n");
    for(int i=0;i<pop_size;i++){
        for(int j=0;j<n_dim;j++){
            //X[i][j] = get_random(-3.0, 5.0);
            printf("%f ", X[i][j]);
        }
        printf("\n");
    } 

}