#include <iostream>
#include <src/ParticleSwarmOptimization.h>
#include <Eigen/Dense>
#include <src/test.h>


double fun(Eigen::Vector<double, 3> &x){
    return x.dot(x);
}

double fun1(Eigen::Vector<double, 5>& x){
    Eigen::Vector<double, 5> v;
    v<<1,2,3,4,5;
    return x.dot(x - v);
}

int main(int, char**) {
    // Eigen::Vector<double, 3> v;
    // v<<1.0, 2.0, 3.0;
    // std::cout << fun(v) << std::endl;
    ParticalSwarmOptimization<double, 5> PSO(&fun1, 10000, 10000);
    PSO.run();
    std::cout << "best sol = " << PSO.getSol() << " optima = [" << PSO.getOptimal().transpose() << "]" << std::endl;
    return 0;
}
