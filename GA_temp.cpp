#include <iostream>
#include<functional>
#include<vector>
#include<list>
#include "Eigen"
#include"Dense"

using namespace std;
using namespace Eigen;

class test{
    public:
        ArrayXf h;
        test(int a, int b){
            h = ArrayXf::Constant(a,b);
        }
};
class GeneticAlgorithmBase{
    public:
        //self.func = func_tramsformer(func)
        std::function<int(vector<int>)> func;
        int size_pop; // size_pop must be even integer
        int max_iter;
        int prob_mut; // probability of mutation
        int n_dim;
        int early_stop;

        bool has_constraint;
        vector<int> constraint_eq; //a list of equal functions with ceq[i]==0
        vector<int> constraint_ueq; // a list of unequal constraint functions with c[i]<= 0;

        vector<vector<int>> Chrom; 
        vector<vector<int>> X; // shape = (size, n_dim)
        vector<int> Y_raw; // shape = (size_pop, ) value is f(x)
        vector<int> Y;  // shape = (soze_pop, ) value os f(x)+ penalty for constraint
        vector<int> FitV; // shape = (size_pop, )

       //waiting for determin 
       //self.generation_best_X = []
       //self.generation_best_Y = []
       //self.all_history_Y = []
       //self.all_history_FitV = []
       vector<vector<int>> generation_best_X;
       vector<int> generation_best_Y;
       vector<int> all_history_Y;
       vector<int> all_history_FitV;

       vector<int> best_x;
       int best_y;

       virtual vector<vector<int>> chrom2x(vector<vector<int>> Chrom)=0;
       void x2y(){
           //Y_raw = func(X); // all of them are array
           //calcaulate Y_raw first
           for(int i=0;i<X.size();i++){
               Y_raw[i] = func(X[i]); 
           }
           //transfer Y_raw to Y
           if(!has_constraint){
               Y = Y_raw;
           }else{
               //waiting for update
           }

       }
        virtual void ranking() = 0;
        virtual void mutation() = 0; //return self.Chrom
        virtual void selection() = 0;
        virtual void crossover() = 0;

        void run(int max_iter_in=0){
            max_iter = max_iter_in ==0 ? max_iter: max_iter_in;
            list<int> best;
            for(int i=0;i<max_iter;i++){
                X = chrom2x(Chrom);
                x2y();
                ranking();
                selection();
                crossover();
                mutation();
                if(early_stop!=0){
                    best.push_back(*min_element(generation_best_Y.begin(),generation_best_Y.end()));
                    if(best.size()>=early_stop){
                        int min_best =  *min_element(best.begin(),best.end());
                        int best_count =0;
                        for(auto number:best){
                            if(number==min_best)best_count++;
                        }
                        if(best_count==best.size())break;
                    }else{
                        best.pop_front();
                    }
                }
            }
            
            int global_best_index = min_element(generation_best_Y.begin(),generation_best_Y.end())- generation_best_Y.begin();
            best_x = generation_best_X[global_best_index];
            best_y = generation_best_Y[global_best_index];
        }


};

class GA: public GeneticAlgorithmBase{
    public:
        ArrayXf lb; // the lower bound of every variables of func
        ArrayXf ub; // the upper bound of every variables of func
        ArrayXf precision;
        ArrayXf Lind;
        GA(int size_pop = 50, int max_iter = 200, int lb_number=-1, int n_dim=1,
        int ub_number = 1, float precision = 1e-7, int early_stop = 0): GeneticAlgorithmBase(){
            this->size_pop =  size_pop;
            this->max_iter = max_iter;
            this->n_dim = n_dim;
            lb = ArrayXf::Constant(n_dim,lb_number);
            ub = ArrayXf::Constant(n_dim, lb_number);
            this->precision = ArrayXf::Constant(n_dim, precision);
            ArrayXf Coefficient = ArrayXf::Ones(n_dim).log2();
            ArrayXf Lind_raw = Coefficient*(ub-lb)/(this->precision);
        }
};
int main()
{
  cout<<"hello"<<endl;
  test a(12,3);
  cout<<a.h(0)<<endl;
}