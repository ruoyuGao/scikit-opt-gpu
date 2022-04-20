#ifndef TEST_H
#define TEST_H

#include <Eigen/Dense>
#include <iostream>

class Balabala{
public:
    Balabala();
    void GenRandom();
    void PrintMatrix();
private:
    Eigen::MatrixXd m;
};

#endif