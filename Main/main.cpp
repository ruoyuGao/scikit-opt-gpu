#include <iostream>
#include <src/ParticleSwarmOptimization.h>
#include <Eigen/Dense>
#include <src/test.h>


double fun(Eigen::Vector<double, 3> &x){
    return x.dot(x);
}

int main(int, char**) {
    Balabala b;
    b.GenRandom();
    b.PrintMatrix();
    ParticalSwarmOptimization<double, 3> PSO(&fun, 10, 10);
    PSO.printParticles();
    PSO.printV();
    PSO.run();
    std::cout << "best sol = " << PSO.getSol() << " optima = [" << PSO.getOptimal().transpose() << "]" << std::endl;
    return 0;
}
