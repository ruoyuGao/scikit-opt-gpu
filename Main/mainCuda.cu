#include <src/ParticleSwarmOptimizationGPU.cuh>


int main(int, char**) {
    // Eigen::Vector<double, 3> v;
    // v<<1.0, 2.0, 3.0;
    // std::cout << fun(v) << std::endl;
    const std::size_t dim = 5;
    ParticalSwarmOptimizationGPU<dim> PSO(10, 10);
    PSO.run();
    float * sol = PSO.getSol();
    // std::cout << "best sol = " << PSO.getSol() << " optima = [" << PSO.getOptimal().transpose() << "]" << std::endl;
    printf("Optimal = %lf Sol = [", PSO.getOptimal());
    for (size_t i = 0; i < dim - 1; i++)
    {
        printf("%ld, ", sol[i]);
    }
    printf("%ld]\n", sol[dim-1]);
    return 0;
}
