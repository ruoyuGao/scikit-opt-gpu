# [scikit-opt-gpu](https://github.com/ruoyuGao/scikit-opt-gpu) 
[![License](https://img.shields.io/pypi/l/scikit-opt.svg)](https://github.com/ruoyuGao/scikit-opt-gpu/blob/ruoyu_edit/LICENSE)
![Platform](https://img.shields.io/badge/platform-windows%20|%20linux%20|%20macos-green.svg)
[![GitHub version](https://badge.fury.io/gh/ruoyuGao%2Fscikit-opt-gpu.svg)](https://badge.fury.io/gh/ruoyuGao%2Fscikit-opt-gpu)
![Cuda version](https://img.shields.io/badge/cuda-%3E%3D11.4-blue)
![Cmake version](https://img.shields.io/badge/cmake-%3E%3D3-blue)
![gcc version](https://img.shields.io/badge/gcc-%3E%3D9.2-blue)
![py version](https://img.shields.io/badge/python-%3E%3D3.7-blue)
![loc](https://img.shields.io/tokei/lines/github/ruoyuGao/scikit-opt-gpu?color=red)
![commit](https://img.shields.io/github/commit-activity/y/ruoyuGao/scikit-opt-gpu?color=red)
![lang](https://img.shields.io/github/languages/top/ruoyuGao/scikit-opt-gpu)
---

## Links
test_file(ACA) → [link](https://drive.google.com/file/d/1rAmWo6g4f0wc2BRm5pEV6hzi79XltjE7/view?usp=sharing)
## Usage
1. Load gcc 9.2 and git 2.6 from CIMS cluster
```
module load gcc-9.2
module load git-2.6.3
module load cmake-3
module load cuda-11.4
```
If you want to test the python version of this lib, please refer to the url above and do 
```
module load python-3.7
pip install scikit-opt
```
2. Clone this repo and build it
```
git clone https://github.com/ruoyuGao/scikit-opt-gpu.git
cd scikit-opt-gpu
mkdir build
cd build
```
You can use any cmake flags to build the project as long as it compiles. However, If you are on CIMS cluster please use the following flags or gcc-9.2 can't be used. CMAKE will automatically go to /usr/local/gcc whose version is 4.8.5.
```
cmake -DCMAKE_C_COMPILER=/usr/local/stow/gcc-9.2/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/stow/gcc-9.2/bin/g++ ..
make -j4
```

3. Run executable file
```
# GA sequential version
./mainGaSeq iteration pop_size cross_prob mutate_prob
# GA cuda version
./GA_gpu iteration pop_size cross_prob mutate_prob
# PSO sequntial version
./main particleNum maxIteration verbose
# PSO cuda version
./mainCuda particleNum maxIteration verbose
# SA sequential version
./mainSA num_of_initalSolutions maxOuterIteration maxInnerIteration verbose
# SA cuda version
./mainSACuda num_of_initalSolutions maxOuterIteration maxInnerIteration verbose
# ACA sequential version
./mainACA_seq num_ants max_iters filename
# ACA cuda version
./mainACA_cuda num_ants max_iters filename
```


