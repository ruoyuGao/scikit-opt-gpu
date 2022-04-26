#ifndef BD4AF906_F0D3_4C2A_ABD4_6F49830B9F98
#define BD4AF906_F0D3_4C2A_ABD4_6F49830B9F98

#include "Optimizer.h"

template<typename T, std::size_t L>
class SimulatedAnnealing: public Optimizer{
public:
    SimulatedAnnealing(std::function<T(Eigen::Vector<T, L>&)> func, int initialTrails=1e5, int MaxIters=1500,int innerIter=300, T T_max=100., T T_min = 1e-5, T tolerance=1e-10);
    void run() override;
    Eigen::Vector<T, L> getSol();
    T getOptimal();
private:
    T t, tMax, tMin;
    int maxIter, innerMaxIter, iter, stay;
    T tol;
    T optimal;
    int xNum;
    T groupBestFunVal;
    Eigen::Vector<T, L> sol;
    Eigen::Vector<T, Eigen::Dynamic> y;
    Eigen::Vector<T, Eigen::Dynamic> y_new;
    Eigen::Vector<T, L> group_best;
    Eigen::Matrix<T, Eigen::Dynamic, L> x;
    Eigen::Matrix<T, Eigen::Dynamic, L> x_new;
    T _randGen();
    std::function<T(Eigen::Vector<T,L>&)> function;
    Eigen::Vector<T, Eigen::Dynamic> evaluate(Eigen::Matrix<T, Eigen::Dynamic, L>);
    void evaluate();
    void _updateTemp();
    // void _updateX();
    void _updateXNew();
    void findGroupBest();
};

template<typename T, std::size_t L>
SimulatedAnnealing<T,L>::SimulatedAnnealing(std::function<T(Eigen::Vector<T, L>&)> func, int initialTrails, int MaxIters, int innerIter, T T_max, T T_min, T tolerance){
    function = func;
    tol = tolerance;
    maxIter = MaxIters;
    innerMaxIter= innerIter;
    iter = 0;
    stay = 0;
    xNum = initialTrails;
    x = 100. * Eigen::Matrix<T, Eigen::Dynamic, L>::Random(xNum, L);
    x_new = x;
    tMax = T_max;
    tMin = T_min;
    t = T_max;
    evaluate();
    findGroupBest();
    sol = group_best;
    optimal = groupBestFunVal;
}

template<typename T, std::size_t L>
void SimulatedAnnealing<T,L>::run(){
    T tmpOptimal=optimal;
    for(int ii=0; ii < maxIter; ii++){
        for(int i=0; i < innerMaxIter; i++){
            _updateXNew();
            y_new = evaluate(x_new);
            Eigen::Vector<T, Eigen::Dynamic> diff = y_new - y;
            Eigen::Vector<T, Eigen::Dynamic> diff_over_t = -1. * diff / t;
            diff_over_t = diff_over_t.array().exp();
            for(int d=0; d < xNum; d++){
                T r = _randGen();
                if(diff(d) < 0 || diff_over_t(d)>r){
                    x.row(d) = x_new.row(d);
                    y(d) = y_new(d);
                    if(y(d) < optimal) {
                        tmpOptimal = optimal;
                        optimal = y(d);
                        sol = x.row(d);
                    }
                }
            }
        }
        T delta_f = abs(tmpOptimal - optimal);
        printf("Iter%d: Optimal=%lf Temp=%lf ∆f=%lf x=[", ii, optimal, t, delta_f);
        for(int j = 0; j < L-1; j++){
                printf("%lf ", sol[j]);
            }
        printf("%lf]\n", sol[L-1]);
        iter ++;
        _updateTemp();
        if( delta_f < tol ) stay++;
        else stay = 0;

        if (t < tMin) break;
        if (stay > 150) break;
    }

}

template<typename T, std::size_t L>
void SimulatedAnnealing<T,L>::_updateTemp(){
    t = tMax / log(iter + 1);
}

template<typename T, std::size_t L>
void SimulatedAnnealing<T,L>::_updateXNew(){
    Eigen::Matrix<T, Eigen::Dynamic, L> u = Eigen::Matrix<T, Eigen::Dynamic, L>::Random(xNum, L);
    Eigen::Matrix<T, Eigen::Dynamic, L> sign = Eigen::Matrix<T, Eigen::Dynamic, L>::Random(xNum, L);
    Eigen::Matrix<T, Eigen::Dynamic, L> ones = Eigen::Matrix<T, Eigen::Dynamic, L>::Ones(xNum, L);
    // printf("tmp initialized.\n");
    T factor = ((T)1.0 + (T)1.0/t);
    for(int i = 0; i < u.rows(); i++){
        for(int j = 0; j < u.cols(); j++){
            if(u(i, j)>0) sign(i,j) = 1.;
            else if(u(i, j)==0) sign(i, j) = 0.;
            else sign(i, j) = -1.; 
            u(i, j) = pow(factor, abs(u(i, j)));
        }
    }
    // printf("check add.\n");
    u = u - ones;
    x_new = x + 20. * t * (sign.array() * u.array()).matrix();
    // printf("check after.\n");
}

template<typename T, std::size_t L>
void SimulatedAnnealing<T, L>::findGroupBest() {
    Eigen::Index minId;
    groupBestFunVal = y.minCoeff(&minId);
    group_best = x.row(minId);
}

template<typename T, std::size_t L>
void SimulatedAnnealing<T, L>::evaluate(){
    y = evaluate(x);
}

template<typename T, std::size_t L>
Eigen::Vector<T, Eigen::Dynamic> SimulatedAnnealing<T, L>::evaluate(Eigen::Matrix<T, Eigen::Dynamic, L> x){
    Eigen::Vector<T, Eigen::Dynamic> val(xNum);
    for(std::size_t i = 0; i < x.rows(); ++i){
        Eigen::Vector<T, L> xi = x.row(i);
        val(i) = function(xi);
    }
    return val;
}

template<typename T, std::size_t L>
T SimulatedAnnealing<T, L>::_randGen() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<T> distribution(0, 1);
    return distribution(eng);
}

template<typename T, std::size_t L>
T SimulatedAnnealing<T, L>::getOptimal() {
    return optimal;
}

template<typename T, std::size_t L>
Eigen::Vector<T, L> SimulatedAnnealing<T, L>::getSol(){
    return sol;
}

#endif /* BD4AF906_F0D3_4C2A_ABD4_6F49830B9F98 */
