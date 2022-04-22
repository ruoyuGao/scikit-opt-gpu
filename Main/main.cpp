#include <iostream>
#include <Eigen/Dense>
#include "src/ACA_TSP.h"

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

int main(int, char**) {

    int num_points = 5;
    //MatrixXd points_coordinate = MatrixXd::Random(num_points, 2);
    MatrixXd points_coordinate(num_points, 2);
    points_coordinate << 0,1,1,1,0,0,1,0,2,0.5;
    MatrixXd distance_matrix(num_points, num_points);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            distance_matrix(i, j) = sqrt(pow((points_coordinate(i, 0) - points_coordinate(j, 0)), 2)
                                         + pow((points_coordinate(i, 1) - points_coordinate(j, 1)), 2));
        }
    }

    ACA_TSP aca(cal_total_distance, num_points, 50, 200, distance_matrix, 1, 2, 0.1);
    pair<RowVectorXd, RowVectorXd> res;
    res = aca.run(-1);

    cout << res.first << endl;
    return 0;
}
