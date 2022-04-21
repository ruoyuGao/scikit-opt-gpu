#ifndef FA0CCF46_5367_4080_8DBC_A031C2232C93
#define FA0CCF46_5367_4080_8DBC_A031C2232C93

#include "Optimizer.h"


template<class T, std::size_t L>
class ParticalSwarmOptimization: public Optimizer{
public:
    explicit ParticalSwarmOptimization(std::function<T(Eigen::Vector<T, L>&)> func, T c1=2.5, T c2=0.5, T w=0.5, T tol=1e-5, int MaxIters=150, T delta_t=1, int NumParticles=1e5);
    void run() override;
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
    Eigen::Matrix<T, Eigen::Dynamic, L> personal_best;
    Eigen::Vector<T, Eigen::Dynamic>    personal_best_func_values;
    Eigen::Vector<T, Eigen::Dynamic>    current_func_values;
    Eigen::Matrix<T, Eigen::Dynamic, L> group_best;
    Eigen::Matrix<T, Eigen::Dynamic, L> v;
    std::function<T(Eigen::Vector<T,L>)> function;
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
    T getSol();
    Eigen::Vector<T, L> optimal;
    Eigen::Vector<T, L> getOptimal();
};


#endif /* FA0CCF46_5367_4080_8DBC_A031C2232C93 */
