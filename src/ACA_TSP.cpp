//
// Created by Shengnan Zhang on 4/21/22.
//

#include "ACA_TSP.h"

ACA_TSP::ACA_TSP(function<double(RowVectorXd,MatrixXd)> func,
                 int n_dim,
                 int size_pop,
                 int max_iter,
                 MatrixXd &distance_matrix,
                 int alpha,
                 int beta,
                 int rho) {
    this->distance_matrix = distance_matrix;
    this->func = func;
    this->n_dim = n_dim;
    this->size_pop = size_pop;
    this->max_iter = max_iter;
    this->alpha = alpha;
    this->beta = beta;
    this->rho = rho;

    this->prob_matrix_distance = 1 / (distance_matrix + 1e-10 * MatrixXd::Identity(n_dim, n_dim)).array();
    this->Tau = MatrixXd::Ones(n_dim, n_dim);
    this->Table = MatrixXd::Zero(size_pop, n_dim);
    this->y = RowVectorXd(1, size_pop);

    this->generation_best_X = MatrixXd(0, n_dim);
    this->generation_best_Y = MatrixXd(0, n_dim);
}

pair<RowVectorXd, RowVectorXd> ACA_TSP::run(int max_iteration) {
    // final return results
    RowVectorXd best_x_res, best_y_res;

    int iteration = (max_iteration == -1) ? this->max_iter : max_iteration;

    // for every iteration
    for (int i = 0; i < iteration; i++) {
        MatrixXd prob_matrix = this->Tau.array().pow(this->alpha) * this->prob_matrix_distance.array().pow(this->beta);
        // for every ant
        for (int j = 0; j < this->size_pop; j++) {
            this->Table(j, 0) = 0; // suppose all the ants start from point 0
            // for every position
            for (int k = 0; k < this->n_dim - 1; k++) {
                RowVectorXd taboo = this->Table.block(j, 0, 1, k+1);

                set<int> taboo_set;
                for (int indx = 0; indx <= k; indx++) {
                    taboo_set.insert(taboo(indx));
                }

                vector<int> allow_list;
                for (int loc = 0; loc < this->n_dim; loc++) {
                    if (taboo_set.find(loc) == taboo_set.end()) {
                        allow_list.push_back(loc);
                    }
                }

                // get the probability of all the allowed next points
                RowVectorXd prob(1, allow_list.size());
                for (int indx = 0; indx < allow_list.size(); indx++) {
                    prob(indx) = prob_matrix((int)this->Table(j, k), allow_list[indx]);
                }

                // normalized the probability
                prob = prob / prob.sum();

                // change the probability vector to list
                // vector<double> prob_list;
                // prob_list.resize(prob.size());
                // VectorXd::Map(&prob_list[0], prob.size()) = prob;

                // Create the distribution with the probability
                std::default_random_engine gen {this->rd()};
                discrete_distribution<int> dist(prob.begin(), prob.end());

                // get the next point
                int next_point = allow_list[dist(gen)];
                this->Table(j, k + 1) = next_point;
            }

        }

        // calculate the total_distance for each ant
        this->y = RowVectorXd(1, this->size_pop);
        for (int indx = 0; indx < this->size_pop; indx++) {
            y(indx) = this->func(this->Table.row(indx), this->distance_matrix);
        }

        // store the best routine for each iteration.
        RowVectorXd::Index minIndx;
        double minDis = y.minCoeff(&minIndx);
        RowVectorXd best_x = this->Table.row(minIndx);
        RowVectorXd best_y = y(minIndx) * RowVectorXd::Ones(1, this->n_dim);

        this->generation_best_X.conservativeResize(this->generation_best_X.rows() + 1,
                                                   this->generation_best_X.cols());
        this->generation_best_X.row(this->generation_best_X.rows() - 1) = best_x;
        this->generation_best_Y.conservativeResize(this->generation_best_Y.rows() + 1,
                                                   this->generation_best_Y.cols());
        this->generation_best_Y.row(this->generation_best_Y.rows() - 1) = best_y;

        // Calculate the change of pheromone for all path for all ants.
        MatrixXd delta_tau(this->n_dim, this->n_dim);
        for (int j = 0; j < this->size_pop; j++) {
            for (int k = 0; k < this->n_dim - 1; k++) {
                // ant go from n1 to n2
                int n1 = this->Table(j, k);
                int n2 = this->Table(j, k+1);
                delta_tau(n1, n2) = delta_tau(n1, n2) + 1 / y(j);
            }
            // for the final road return to the start point.
            int n1 = this->Table(j, this->n_dim - 1);
            int n2 = this->Table(j, 0);
            delta_tau(n1, n2) = delta_tau(n1, n2) + 1 / y(j);
        }

        // the evaporation and addition of pheromone
        this->Tau = (1 - this->rho) * this->Tau + delta_tau;
    }

    int best_generation;
    int temp;
    this->generation_best_Y.minCoeff(&best_generation, &temp);
    best_x_res = this->generation_best_X.row(best_generation);
    best_y_res = this->generation_best_Y.row(best_generation);

    pair<RowVectorXd, RowVectorXd> resPair;
    resPair.first = best_x_res;
    resPair.second = best_y_res;

    return resPair;
}
