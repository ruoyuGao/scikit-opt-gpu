//
// Created by Shengnan Zhang on 4/21/22.
//

#ifndef LIBSCIKITGPU_ACA_TSP_H
#define LIBSCIKITGPU_ACA_TSP_H

#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include <set>
#include <vector>
#include <random>

using namespace std;
using namespace Eigen;

// example distance calculation function.
/*
double cal_total_distance(RowVectorXd &routine, MatrixXd &distance_matrix) {
    int num_points = routine.size();
    double sum = 0;
    for (int i = 0; i < num_points; i++) {
        int n1 = i % num_points;
        int n2 = (i + 1) % num_points;
        sum += distance_matrix((int)routine(n1), (int)routine(n2));
    }
    return sum;
}
*/

class ACA_TSP {
public:
    ACA_TSP(function<double(RowVectorXd , MatrixXd )> func,
            int n_dim,
            int size_pop,
            int max_iter,
            MatrixXd &distance_matrix,
            double alpha,
            double beta,
            double rho);

    pair<RowVectorXd, RowVectorXd> run(int max_iteration);

private:
    MatrixXd distance_matrix;
    function<double(RowVectorXd , MatrixXd )> func; // the function to calculate total distance
    int n_dim; //the number of cities
    int size_pop; //the number of ants
    int max_iter; //the limit of iterations
    double alpha, beta, rho; // parameter for pheromone/distance/evaporation factor
    MatrixXd prob_matrix_distance; // change the distance to its reciprocal
    MatrixXd Tau; // pheromone matrix for each road, update each iteration.
    MatrixXd Table; // the routine of an ant by current iteration
    RowVectorXd y; // the total distance an ant has gone through by current iteration.
    MatrixXd generation_best_X, generation_best_Y; // record the best of each iteration

    random_device rd;
};


#endif //LIBSCIKITGPU_ACA_TSP_H
