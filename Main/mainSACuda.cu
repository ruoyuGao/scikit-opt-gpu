#include <src/SimulatedAnnealingGPU.cuh>


int main(int argc, char* argv[]) {
    const std::size_t dim = 2;
    int particleNum = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int inneriters = atoi(argv[3]);
    SimulatedAnnealingGPU<dim> SA(particleNum, iters, inneriters);
    SA.run();
    double * sol = new double[dim];
    sol = SA.getSol();
    printf("Optimal = %lf Sol = [", SA.getOptimal());
    for (size_t i = 0; i < dim - 1; i++)
    {
        printf("%lf, ", sol[i]);
    }
    printf("%lf]\n", sol[dim-1]);

    return 0;
}
