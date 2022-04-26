#ifndef FA0CCF46_5367_4080_8DBC_A031C2232C93
#define FA0CCF46_5367_4080_8DBC_A031C2232C93

#include "Optimizer.h"

template<typename T, std::size_t L>
class ParticalSwarmOptimization: public Optimizer{
public:
    ParticalSwarmOptimization(std::function<T(Eigen::Vector<T, L>&)> func, int NumParticles=1e5, int MaxIters=1500, int Verbose=1, T C1=2.5, T C2=0.5, T weight=0.5, T tolerence=1e-10, T delta_t=1.0);
    void run() override;
    Eigen::Vector<T, L> getOptimal();
    T getSol();
    void printParticles(){std::cout << particles << std::endl;}
    void printV(){std::cout << v << std::endl;}
private:
    T c1;
    T c2;
    T w;
    T tol;
    T dt;
    T sol;
    int maxIter;
    int verbose;
    int particleNum;
    Eigen::Matrix<T, Eigen::Dynamic, L> particles;
    Eigen::Matrix<T, Eigen::Dynamic, L> personal_best;
    Eigen::Vector<T, Eigen::Dynamic>    personal_best_func_values;
    Eigen::Vector<T, Eigen::Dynamic>    current_func_values;
    Eigen::Vector<T, Eigen::Dynamic>    v_norm;
    Eigen::Matrix<T, Eigen::Dynamic, L> group_best;
    Eigen::Matrix<T, Eigen::Dynamic, L> v;
    std::function<T(Eigen::Vector<T,L>&)> function;
    void _initParticles();
    void _initVelocity();
    void _updateVelocity();
    void _updateParticles();
    void _updateCurrentFuncValues();
    T _randGen();
    void findPersonalBest();
    void findGroupBest();
    Eigen::Vector<T, Eigen::Dynamic> evaluate(Eigen::Matrix<T, Eigen::Dynamic, L>);
    Eigen::Vector<T, Eigen::Dynamic> evaluate();
    Eigen::Vector<T, L> optimal;
};

template<typename T, std::size_t L>
ParticalSwarmOptimization<T,L>::ParticalSwarmOptimization(std::function<T(Eigen::Vector<T, L>&)> func, int NumParticles, int MaxIters, int Verbose, T C1, T C2, T weight, T tolerence, T delta_t):Optimizer(){
    function = func;
    c1 = C1;
    c2 = C2;
    w = weight;
    tol = tolerence;
    maxIter = MaxIters;
    dt = delta_t;
    verbose = Verbose;
    particleNum = NumParticles;
    _initParticles();
    _initVelocity();
    personal_best = particles;
    personal_best_func_values = evaluate();
    Eigen::Index minId;
    personal_best_func_values.minCoeff(&minId);
    group_best = particles;
    group_best.rowwise() = personal_best.row(minId);
    sol = personal_best_func_values.minCoeff();
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T,L>::run(){
    Eigen::Index minId;
    T tmpSol;
    _updateParticles();
    for (int i = 0; i < maxIter; ++i) {
        _updateCurrentFuncValues();
        tmpSol = current_func_values.minCoeff(&minId);
        T diff = std::abs(sol - tmpSol);
        if(diff < tol){
            sol = tmpSol;
            optimal = particles.row(minId);
            break;
        }
        if(tmpSol<sol){
            sol = tmpSol;
            optimal = particles.row(minId);
        }
        v_norm = v.rowwise().norm();
        Eigen::Index minVId, maxVId;
        T minV, maxV;
        minV = v_norm.minCoeff(&minVId);
        maxV = v_norm.maxCoeff(&maxVId);
        _updateVelocity();
        _updateParticles();
        findPersonalBest();
        findGroupBest();
        if(verbose){
            std::cout << "Iter" << i << ": minFuncVal=" << sol <<" ∆f=" <<diff << " x= [" << optimal.transpose()<<"]" << std::endl;
        }
    }
    // sol = tmpSol;
    // optimal = particles.row(minId);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T,L>::_initParticles() {
    particles = 100. * Eigen::Matrix<T, Eigen::Dynamic, L>::Random(particleNum, L);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_initVelocity() {
    v = Eigen::Matrix<T, Eigen::Dynamic, L>::Random(particleNum, L);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_updateVelocity() {
    T r1 = _randGen();
    T r2 = _randGen();
    v = w * v + c1 * r1 * (personal_best-particles)/dt + c2 * r2 *(group_best - particles)/dt;
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::_updateParticles() {
    particles = particles + v * dt;
}

template<typename T, std::size_t L>
T ParticalSwarmOptimization<T, L>::_randGen() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<T> distribution(0, 1);
    return distribution(eng);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T, L>::findPersonalBest() {
    for(std::size_t i = 0; i < current_func_values.size(); ++i){
        if(current_func_values(i) < personal_best_func_values(i)){
            personal_best.row(i) = particles.row(i);
            personal_best_func_values(i) = current_func_values(i);
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
    Eigen::Vector<T, Eigen::Dynamic> val(particleNum);
    for(std::size_t i = 0; i < x.rows(); ++i){
        Eigen::Vector<T, L> xi = x.row(i);
        val(i) = function(xi);
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


#endif /* FA0CCF46_5367_4080_8DBC_A031C2232C93 */
