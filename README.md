# [scikit-opt-gpu](https://github.com/guofei9987/scikit-opt)

[![codecov](https://codecov.io/gh/guofei9987/scikit-opt/branch/master/graph/badge.svg)](https://codecov.io/gh/guofei9987/scikit-opt)
[![License](https://img.shields.io/pypi/l/scikit-opt.svg)](https://github.com/guofei9987/scikit-opt/blob/master/LICENSE)
![Platform](https://img.shields.io/badge/platform-windows%20|%20linux%20|%20macos-green.svg)
## Usage
1. Load gcc 9.2 and git 2.6 from CIMS cluster
```
module load gcc-9.2
module load git-2.6.3
module load cmake-3
module load cuda-11.4
```
2. Clone this repo and build it
```
git clone git@github.com:ruoyuGao/scikit-opt-gpu.git
cd scikit-opt-gpu
mkdir build
cd build
```
You can use any cmake flags to build the project as long as it compiles. However, If you are on CIMS cluster please use the following flags or gcc-9.2 can't be used. CMAKE will automatically go to /usr/local/gcc whose version is 4.8.5.
```
cmake -DCMAKE_C_COMPILER=/usr/local/stow/gcc-9.2/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/stow/gcc-9.2/bin/g++ ..
make -j4
```
