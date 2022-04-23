#ifndef B8FB8CB3_8116_46C7_AB2E_62C35898B0C2
#define B8FB8CB3_8116_46C7_AB2E_62C35898B0C2

#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <ctime>
#include <cstdlib>
#include <random>
#include <cmath>

class Optimizer{
public:
    Optimizer()=default;
    virtual void run()=0;
};

#endif /* B8FB8CB3_8116_46C7_AB2E_62C35898B0C2 */