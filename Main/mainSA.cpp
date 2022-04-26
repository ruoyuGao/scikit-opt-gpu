#include <iostream>
#include <src/SimulatedAnnealing.h>
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
    int inneriters = atoi(argv[3]);
    int verbose = atoi(argv[4]);
    SimulatedAnnealing<double, 2> SA(&fun1, particleNum, iters, inneriters, verbose);
    SA.run();
    std::cout << "Optimal = " << SA.getOptimal() << " x = [" << SA.getSol().transpose() << "]" << std::endl;
    return 0;
}
