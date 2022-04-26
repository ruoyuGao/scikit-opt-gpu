#include<iostream>
#include<functional>
#include<vector>
#include<random>
#include<cmath>
#include<list>
#include <algorithm> 
#include"src/ga_seq.h"
using namespace std;

float pow_test(vector<float> tt){
    return tt[0] + 10* sin(5*tt[0])+ 7*cos(4*tt[0])+ tt[1] +tt[2];
}
int main(int argc, char *argv[]){

    printf("parameter seq: iteration pop_size cross_prob, mutate_prob");
    if(argc!= 4){
        printf("wrong number of input size\n");
    }
    int iteration = atoi(argv[1]);
    int pop_size =  atoi(argv[2]);
    float cross_prob = atof(argv[3]);
    float mutate_prob = atof(argv[4]);
    vector<int> test_rand(10);
    vector<float>lb = {-10.0,-10,-10};
    vector<float>ub = {10.0,10 ,10 };
    function<float(vector<float>)> test_func = pow_test;
    GA test(test_func,pop_size,3,lb,ub,cross_prob, mutate_prob,iteration);
    test.run();
}