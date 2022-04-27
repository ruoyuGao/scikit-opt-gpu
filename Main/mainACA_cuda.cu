#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include <set>
#include <vector>
#include <random>
#include <cuda.h>
#include <fstream>

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

/********************************************************************************************/
/*** method declare ***/
__global__ void pheromone_delta_kernel(double * delta_tau_device, double * table_device, double * y, int num_ant, int num_pos);
__global__ void pheromone_change_kernel(double rho, double * Tau_device, double * delta_tau_device, int num_pos);


/********************************************************************************************/


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

ACA_TSP::ACA_TSP(function<double(RowVectorXd , MatrixXd )> func,
                 int n_dim,
                 int size_pop,
                 int max_iter,
                 MatrixXd &distance_matrix,
                 double alpha,
                 double beta,
                 double rho) {
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

    /*** malloc memory in gpu ***/
    unsigned int tau_table_size = 0;
    unsigned int table_size = 0;
    unsigned int y_size = sizeof(double) * size_pop;
    tau_table_size = sizeof(double) * (this->n_dim * this->n_dim);
    table_size = sizeof(double) * (this->size_pop * this->n_dim);

    double * delta_tau_device;
    double * Tau_device;
    double * table_device;
    double * y_device ;

    cudaMalloc((void**) &delta_tau_device, tau_table_size);
    cudaMalloc((void**) &Tau_device, tau_table_size);
    cudaMalloc((void**) &table_device, table_size);
    cudaMalloc((void**) &y_device, y_size);

    /*** ****** ****** ** *** ***/

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
            VectorXd this_routine = this->Table.row(indx);
            y(indx) = this->func(this_routine, this->distance_matrix);
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

        int block_x = 8;
        int block_y = block_x;
        int grid_x = (this->n_dim % block_x == 0)? this->n_dim / block_x : this->n_dim / block_x + 1;
        int grid_y = (this->size_pop % block_y == 0)? this->size_pop / block_y : this->size_pop / block_y + 1;

        dim3 dimGrid1(grid_x, grid_y);
        dim3 dimBlock1(block_x, block_y);

        double * Tau_host = this->Tau.data();
        double * table_host = this->Table.data();
        double * y_host = this->y.data();


        cudaMemset(delta_tau_device, 0, tau_table_size);
        cudaMemcpy(Tau_device, Tau_host, tau_table_size, cudaMemcpyHostToDevice);
        cudaMemcpy(table_device, table_host, table_size, cudaMemcpyHostToDevice);
        cudaMemcpy(y_device, y_host, y_size, cudaMemcpyHostToDevice);

        pheromone_delta_kernel<<<dimGrid1, dimBlock1>>>(delta_tau_device, table_device, y_device, size_pop, n_dim);

        dim3 dimGrid2(grid_x, grid_x);
        dim3 dimBlock2(block_x, block_y);

        pheromone_change_kernel<<<dimGrid2, dimBlock2>>>(this->rho, Tau_device, delta_tau_device, this->n_dim);

        cudaMemcpy(Tau_host, Tau_device, tau_table_size, cudaMemcpyDeviceToHost);



        /* sequential version

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

        */
    }

    int best_generation;
    int temp;
    this->generation_best_Y.minCoeff(&best_generation, &temp);
    best_x_res = this->generation_best_X.row(best_generation);
    best_y_res = this->generation_best_Y.row(best_generation);

    pair<RowVectorXd, RowVectorXd> resPair;
    resPair.first = best_x_res;
    resPair.second = best_y_res;

    cudaFree(delta_tau_device);
    cudaFree(Tau_device);
    cudaFree(table_device);
    cudaFree(y_device);

    return resPair;
}

double cal_total_distance(RowVectorXd routine, MatrixXd distance_matrix) {
    int num_points = routine.size();
    double sum = 0;
    for (int i = 0; i < num_points; i++) {
        int n1 = i % num_points;
        int n2 = (i + 1) % num_points;
        sum += distance_matrix((int)routine(n1), (int)routine(n2));
    }
    return sum;
}

int main(int argc, char* argv[]) {

    int num_points = 0;
    int num_ants = 0;
    int max_iter = 0;
    char* textfile;

    double alpha = 1;
    double beta = 2;
    double rho = 0.1;

    if (argc != 4) {
        fprintf(stderr, "usage: ./exe num_ants max_iterations cities_textfile\n");
        exit(1);
    }

    num_ants = atoi(argv[1]);
    max_iter = atoi(argv[2]);
    textfile = argv[3];

    // reading numbers from file
    ifstream fp;
    fp.open(textfile);

    vector<double> pos_coord;

    string thisLine;
    while (getline(fp, thisLine)) {
        int count;
        double x;
        double y;
        sscanf(thisLine.c_str(), "%d %lf %lf", &count, &x, &y);
        pos_coord.push_back(x);
        pos_coord.push_back(y);
    }

    // change the postions into eigen Matrix format
    num_points = pos_coord.size()/2;

    MatrixXd points_coordinate(num_points, 2);

    for (int i = 0; i < pos_coord.size() - 1; i+=2) {
        points_coordinate(i/2, 0) = pos_coord[i];
        points_coordinate(i/2, 1) = pos_coord[i+1];
    }

    // parameters.
    cout << num_ants << " ants" << endl;
    cout << num_points << " cities" << endl;
    cout << max_iter << " iterations" << endl;

    fp.close();


    //MatrixXd points_coordinate = MatrixXd::Random(num_points, 2);
    //points_coordinate = 2 * MatrixXd::Random(num_points, 2);
    MatrixXd distance_matrix(num_points, num_points);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            distance_matrix(i, j) = sqrt(pow((points_coordinate(i, 0) - points_coordinate(j, 0)), 2)
                                         + pow((points_coordinate(i, 1) - points_coordinate(j, 1)), 2));
        }
    }


    ACA_TSP aca(cal_total_distance, num_points, num_ants, max_iter, distance_matrix, alpha, beta, rho);
    pair<RowVectorXd, RowVectorXd> res;
    res = aca.run(-1);

    double sum = 0;
    for (int i = 0; i < num_points; i++) {
        int n1 = res.first(i);
        int n2 = res.first((i+1) % num_points);
        sum += distance_matrix(n1, n2);
    }

    cout << "route: " << endl;
    cout << res.first << endl;
    cout << "total distance: " << sum << endl;
    return 0;
}

/************************************************************************************************/
/*** kernels ***/
__global__ void pheromone_delta_kernel(double * delta_tau_device, double * table_device, double * y, int num_ant, int num_pos) {

    int ant = blockIdx.y * blockDim.y + threadIdx.y;
    int pos_indx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ant < num_ant && pos_indx < num_pos) {
        int n1 = table_device[ant + pos_indx * num_ant];
        int n2;
        if (pos_indx == num_pos - 1) {
            n2 = table_device[ant + 0 * num_ant];
        } else {
            n2 = table_device[ant + (pos_indx + 1) * num_ant];
        }

        double before = delta_tau_device[n1 + n2 * num_pos];

        delta_tau_device[n1 + num_pos * n2] = before + 1 / y[ant];
    }
}

__global__ void pheromone_change_kernel(double rho, double * Tau_device, double * delta_tau_device, int num_pos) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < num_pos && x < num_pos) {
        double before = Tau_device[y + x * num_pos];
        Tau_device[y + x * num_pos] = (1 - rho) * before + delta_tau_device[y + x * num_pos];
    }
}