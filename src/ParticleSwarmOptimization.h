#ifndef FA0CCF46_5367_4080_8DBC_A031C2232C93
#define FA0CCF46_5367_4080_8DBC_A031C2232C93

#include "Optimizer.h"

template<typename T, std::size_t L>
class ParticalSwarmOptimization: public Optimizer{
public:
    ParticalSwarmOptimization(std::function<T(Eigen::Vector<T, L>&)> func, int NumParticles=1e5, int MaxIters=1500, T C1=2.5, T C2=0.5, T weight=0.5, T tolerence=1e-10, T delta_t=0.01);
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
    int maxIter;
    T dt;
    int particleNum;
    T sol;
    Eigen::Matrix<T, Eigen::Dynamic, L> particles;
    Eigen::Matrix<T, Eigen::Dynamic, L> old_particles;
    Eigen::Matrix<T, Eigen::Dynamic, L> personal_best;
    Eigen::Vector<T, Eigen::Dynamic>    personal_best_func_values;
    Eigen::Vector<T, Eigen::Dynamic>    current_func_values;
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
ParticalSwarmOptimization<T,L>::ParticalSwarmOptimization(std::function<T(Eigen::Vector<T, L>&)> func, int NumParticles, int MaxIters, T C1, T C2, T weight, T tolerence, T delta_t):Optimizer(){
    function = func;
    c1 = C1;
    c2 = C2;
    w = weight;
    tol = tolerence;
    maxIter = MaxIters;
    dt = delta_t;
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
        // std::cout << "current_function_values = " <<current_func_values.transpose()<< std::endl;
        tmpSol = current_func_values.minCoeff(&minId);
        T diff = std::abs(sol - tmpSol);
        if(diff < tol){
            sol = tmpSol;
            optimal = particles.row(minId);
            break;
        }
        if(tmpSol<sol) {
            sol = tmpSol;
            optimal = particles.row(minId);
        }
        _updateVelocity();
        // std::cout << "v = " << std::endl;
        // std::cout << v << std::endl;
        _updateParticles();
        // std::cout << "particles = " << std::endl;
        // std::cout << particles << std::endl;
        findPersonalBest();
        // std::cout << "personal best = " << std::endl;
        // std::cout << personal_best << std::endl;
        findGroupBest();
        // std::cout << "group best = " << std::endl;
        // std::cout << group_best << std::endl;
        std::cout << "Iter" << i << ": minFuncVal=" << sol << " x= [" << optimal.transpose()<<"]" <<" delta_f=" <<diff<< " ||V||=" <<v.squaredNorm() << std::endl;
    }
    sol = tmpSol;
    optimal = particles.row(minId);
}

template<typename T, std::size_t L>
void ParticalSwarmOptimization<T,L>::_initParticles() {
    particles = 100. * Eigen::Matrix<T, Eigen::Dynamic, L>::Random(particleNum, L);
    old_particles = particles;
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
    // auto diff = particles - old_particles;
    // old_particles = particles;
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
