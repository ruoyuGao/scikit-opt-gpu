#include <iostream>
#include <src/ParticleSwarmOptimization.h>
#include <Eigen/Dense>

double fun(Eigen::Vector<double, 3> &x){
    return x.dot(x);
}

int main(int, char**) {
    ParticalSwarmOptimization<double, 3> PSO(&fun);
    PSO.run();
    return 0;
}
