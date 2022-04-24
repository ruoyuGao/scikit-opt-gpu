#include<iostream>
#include <iostream>
#include<functional>
#include<vector>
#include<random>
#include<cmath>
#include<list>
#include "Eigen"
#include"Dense"

using namespace std;
using namespace Eigen;
class GA{
    public:
        function<float(vector<float>)> func;
        vector<float> lb; //lower bound for each var
        vector<float> ub; // upper bound for each var
        bool is_encoded; //use encode method or not
        bool is_roulette_wheel_selection;
        int pop_size; //population size
        int n_dim; // dimension for input
        int iteration;// iteration for GA
        float cross_prob; //probility to do crossover
        float mutate_prob; //probility to mutate
        vector<vector<float>> X; //input of the fitness function
        vector<float> Y; //output of the original function
        vector<float> fitness; //fiteness function of each input
        vector<int> selection();
        vector<float> best_x;
        float best_fitness;
        void init_population();
        void crossover(vector<int> parent_index);
        void get_fitness();
        float get_fitness(vector<float> indival);
        float get_random(float low, float up);
        int get_rand(int up);
        void mutation();
        void run();
        GA(function<float(vector<float>)> func,int pop_size, int n_dim,vector<float>& in_lb, vector<float>&in_ub,
        float cross_prob, float mutate_prob, int iter);
};

GA::GA(function<float(vector<float>)> func, int pop_size, int n_dim,vector<float>& in_lb, vector<float>&in_ub, float cross_prob, float mutate_prob, int iter):lb(n_dim),ub(n_dim),fitness(pop_size),X(pop_size, vector<float>(n_dim)){
    this->pop_size = pop_size;
    this->is_roulette_wheel_selection = false;
    this->n_dim = n_dim;
    this->lb = in_lb;
    this->ub = in_ub;
    this->func = func;
    this->cross_prob =cross_prob;
    this->mutate_prob = mutate_prob;
    this->iteration = iter;
}

void GA::get_fitness(){
    for(int i=0;i<pop_size;i++){
        fitness[i] = this->func(X[i]);
    }
}

float GA::get_fitness(vector<float> indival){
    return this->func(indival);
}

float GA:: get_random(float low, float up){
    std::random_device rd;     //Get a random seed from the OS entropy device, or whatever
    std::mt19937_64 eng(rd());
    uniform_real_distribution<float>distr(low,up);
    return distr(eng);
}

int GA::get_rand(int up){
    std::random_device rd;     //Get a random seed from the OS entropy device, or whatever
    std::mt19937_64 eng(rd());
    uniform_int_distribution<int> distr(1,up);
    return distr(eng);
}
void GA::init_population(){
    for(int i=0;i<this->n_dim;i++){
        for(int j=0;j<this->pop_size;j++){
            //cout<<pop_size<<endl;
            X[j][i] = get_random(lb[i],ub[i]);
        }
    }
}
vector<int> GA::selection(){
    vector<int> index_res(pop_size);
    if(!is_roulette_wheel_selection){
        //default_random_engine generator;
        std::random_device rd;     //Get a random seed from the OS entropy device, or whatever
        std::mt19937_64 eng(rd());
        uniform_int_distribution<int> distr(0,pop_size-1);
        for(int i=0;i<pop_size;i++){
            index_res[i]=distr(eng);
        }
    }else{
        //need to implement
    }
    return index_res;
}
void GA::crossover(vector<int> parent_index){
    for(int i=0;i<pop_size;i++){
        float prob = get_random(0.0,1.0);
        if(prob< this->cross_prob){
            vector<float> child1(n_dim);
            vector<float> child2(n_dim);
        
            int cross_dim = get_rand(n_dim);
            for(int j=0;j<cross_dim;j++){
                child1[j] = 0.7*X[i][j] + 0.3*X[parent_index[i]][j];
                child2[j] = 0.3*X[i][j] + 0.7*X[parent_index[i]][j];
            }
            float fitness1 = get_fitness(child1);
            float fitness2 = get_fitness(child2);
            if(fitness1>fitness[i]){
                X[i] = child1;
                fitness[i] = fitness1;
            }
            if(fitness2>fitness[parent_index[i]]){
                X[parent_index[i]] = child2;
                fitness[parent_index[i]] = fitness2;
            }
        }
    }
}
void GA::mutation(){
    for(int i=0;i<pop_size;i++){
        int prob = get_random(0.0,1.0);
        if(prob<this->mutate_prob){
            std::random_device rd;     //Get a random seed from the OS entropy device, or whatever
            std::mt19937_64 eng(rd());
            uniform_int_distribution<int> distr(0,n_dim-1);
            int mutate_index = distr(eng);
            X[i][mutate_index] = get_random(lb[mutate_index],ub[mutate_index]);
            fitness[i] = get_fitness(X[i]);
        }
    }
}
void GA::run(){
    init_population();
    get_fitness();
    for(int i=0;i<iteration;i++){
        vector<int> select_index(pop_size);
        select_index = selection();
        crossover(select_index);
        mutation();
        get_fitness();
        int max_fitness_index = max_element(fitness.begin(),fitness.end()) - fitness.begin();
        float max_fitness = fitness[max_fitness_index];
        if(best_fitness<max_fitness){
            best_x = X[max_fitness_index];
            best_fitness = max_fitness;
        }
        cout<<"iteration: "<<i<<" "<<"best fitness: "<< best_fitness<<endl;
    }
}
float pow_test(vector<float> tt){
    return tt[0] + 10* sin(5*tt[0])+ 7*cos(4*tt[0]);
}
int main(){
    vector<int> test_rand(10);
    vector<float>lb = {-10.0,2.2,2.1};
    vector<float>ub = {10.0,7.8,9.9};
    function<float(vector<float>)> test_func = pow_test;
    GA test(test_func,20,3,lb,ub,0.75, 0.1,500);
    test.run();
}