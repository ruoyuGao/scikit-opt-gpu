#include <iostream>
#include <Eigen/Dense>
#include "src/ACA_TSP.h"
#include <stdio.h>

#include <fstream>

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
