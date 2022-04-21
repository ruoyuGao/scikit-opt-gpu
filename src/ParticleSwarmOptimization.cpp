#include "ParticleSwarmOptimization.h"


template<typename T, std::size_t L>
ParticalSwarmOptimization<T,L>::ParticalSwarmOptimization(std::function<T(Eigen::Vector<T, L>&)> func, T c1, T c2, T w, T tol, int MaxIters, T delta_t, int NumParticles):Optimizer(){
    function = func;
    c1 = c1;
    c2 = c2;
    w = w;
    tol = tol;
    maxIter = MaxIters;
    dt = delta_t;
    particleNum = NumParticles;
    _initParticles();
    _initVelocity();
    personal_best = particles;
    personal_best_func_values = evaluate();
    Eigen::Index minId;
    personal_best_func_values.minCoeff(&minId);
    group_best.rowwise() = personal_best.row(minId);
    sol = personal_best_func_values.minCoeff();
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T,L>::run(){
    Eigen::Index minId;
    T tmpSol;
    for (int i = 0; i < maxIter; ++i) {
        _updateCurrentFuncValues();
        tmpSol = current_func_values.minCoeff(minId);
        if(std::abs(sol - tmpSol) < tol){
            sol = tmpSol;
            optimal = particles.row(minId);
            break;
        }
        if(tmpSol<sol) {
            sol = tmpSol;
        }
        _updateVelocity();
        _updateParticles();
    }
    sol = tmpSol;
    optimal = particles.row(minId);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T,L>::_initParticles() {
    particles = 100. * Eigen::Matrix<T, Eigen::Dynamic, L>::Random(particleNum, L);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_initVelocity() {
    v = Eigen::Matrix<T, Eigen::Dynamic, L>::Random();
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_updateVelocity() {
    T r1 = _randGen();
    T r2 = _randGen();
    v = w * v + c1 * r1 * (personal_best-particles)/dt+c2 * r2 *(group_best - particles)/dt;
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_updateParticles() {
    particles = particles + v * dt;
}

template<typename T, std::size_t L>
T ParticalSwarmOptimization<T, L>::_randGen() {
    std::default_random_engine generator;
    std::uniform_int_distribution<T> distribution(1, 6);
    auto dice = std::bind(distribution, generator);
    T roll = dice();
    return roll;
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::findPersonalBest() {
    for(std::size_t i = 0; i < particles.rows(); ++i){
        if(current_func_values(i) < personal_best_func_values(i)){
            personal_best.row(i) = particles.row(i);
        }
    }
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::findGroupBest() {
    Eigen::Index minId;
    current_func_values.minCoeff(&minId);
    group_best.rowwise() = particles.row(minId);
}

template<typename T, std::size_t L>
Eigen::Vector<T, -1> ParticalSwarmOptimization<T, L>::evaluate(Eigen::Matrix<T, Eigen::Dynamic, L> x) {
    Eigen::Vector<T, Eigen::Dynamic> val;
    for(std::size_t i = 0; i < x.rows(); ++i){
        val(i) = function(x.row(i));
    }
    return val;
}

template<typename T, std::size_t L>
Eigen::Vector<T, -1> ParticalSwarmOptimization<T, L>::evaluate() {
    return evaluate(particles);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_updateCurrentFuncValues() {
    current_func_values = evaluate();
}

template<typename T, std::size_t L>
T ParticalSwarmOptimization<T, L>::getSol() {
    return sol;
}

template<typename T, std::size_t L>
Eigen::Vector<T, L> ParticalSwarmOptimization<T, L>::getOptimal() {
    return optimal;
}
