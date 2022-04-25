#include <iostream>
#include <src/ParticleSwarmOptimizationGPU.cuh>


int main(int, char**) {
    // Eigen::Vector<double, 3> v;
    // v<<1.0, 2.0, 3.0;
    // std::cout << fun(v) << std::endl;
    ParticalSwarmOptimizationGPU<5> PSO(10000, 10000);
    // PSO.run();
    // std::cout << "best sol = " << PSO.getSol() << " optima = [" << PSO.getOptimal().transpose() << "]" << std::endl;
    return 0;
}
