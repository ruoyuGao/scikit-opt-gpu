# scikit-opt-gpu

## Usage
1. Load gcc 9.2 and git 2.6 from cims cluster
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
cmake -DCMAKE_C_COMPILER=/usr/local/stow/gcc-9.2/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/stow/gcc-9.2/bin/g++ ..
make -j4
```