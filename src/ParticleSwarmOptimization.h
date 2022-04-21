#ifndef FA0CCF46_5367_4080_8DBC_A031C2232C93
#define FA0CCF46_5367_4080_8DBC_A031C2232C93

#include "Optimizer.h"

template<class T>
class ParticalSwarmOptimization: public Optimizer{
public:
    ParticalSwarmOptimization(std::function<T(Eigen::VectorXd x)> func, T c1=2.5, T c2=0.5, T w=0.5, T tol=1e-5, int MaxIters=150, T delta_t=1);
    Eigen::MatrixXd run() override;
private:
    T c1;
    T c2;
    T w;
    T tol;
    int maxIter;
    T dt;
    std::function<T(Eigen::VectorXd)> function;
};

#endif /* FA0CCF46_5367_4080_8DBC_A031C2232C93 */
