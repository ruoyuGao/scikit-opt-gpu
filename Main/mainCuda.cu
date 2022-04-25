#include <src/ParticleSwarmOptimizationGPU.cuh>


int main(int, char**) {
    const std::size_t dim = 2;
    ParticalSwarmOptimizationGPU<dim> PSO(1000000, 10000);
    PSO.run();
    double * sol = new double[dim];
    sol = PSO.getSol();
    printf("Optimal = %lf Sol = [", PSO.getOptimal());
    for (size_t i = 0; i < dim - 1; i++)
    {
        printf("%lf, ", sol[i]);
    }
    printf("%lf]\n", sol[dim-1]);

    return 0;
}
