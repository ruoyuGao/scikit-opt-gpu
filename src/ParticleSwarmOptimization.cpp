#include "ParticleSwarmOptimization.h"


template<class T>
ParticalSwarmOptimization<T>::ParticalSwarmOptimization(std::function<T(Eigen::VectorXd)> func, T c1, T c2, T w, T tol, int MaxIters, T delta_t):Optimizer(){
    function = func;
    c1 = c1;
    c2 = c2;
    w = w;
    tol = tol;
    maxIter = MaxIters;
    dt = delta_t;
}

template<class T>
Eigen::MatrixXd ParticalSwarmOptimization<T>::run(){
    
}
