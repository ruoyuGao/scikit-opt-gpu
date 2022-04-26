#include <iostream>
#include <src/ParticleSwarmOptimization.h>
#include <Eigen/Dense>


double fun(Eigen::Vector<double, 3> &x){
    return x.dot(x);
}

double fun1(Eigen::Vector<double, 2>& x){
    Eigen::Vector<double, 2> v;
    v<<1,2;
    return x.dot(x - v);
}

int main(int argc, char* argv[]) {
    int particleNum = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int verbose = atoi(argv[3]);
    ParticalSwarmOptimization<double, 2> PSO(&fun1, particleNum, iters, verbose);
    PSO.run();
    std::cout << "Optimal = " << PSO.getSol() << " x = [" << PSO.getOptimal().transpose() << "]" << std::endl;
    return 0;
}
