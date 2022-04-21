#include <iostream>
#include <Eigen/Dense>
#include <src/ParticleSwarmOptimization.h>

double fun(Eigen::Vector<double, 3> &x){
    return x.dot(x);
}

int main(int, char**) {
    ParticalSwarmOptimization<double, 3> PSO(&fun);
    PSO.run();
    return 0;
}