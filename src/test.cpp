#include "test.h"

Balabala::Balabala() {
    len=0;
}

void Balabala::GenRandom() {
    m = Eigen::MatrixXd::Random(3,3);
}

void Balabala::PrintMatrix() {
    std::cout << m << std::endl;
}
