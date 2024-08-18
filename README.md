# Variational Mode Graph Convolutional Network (VMGCN) 
LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf)
Github repo: [Link](https://github.com/liuxu77/LargeST)
you can download all data from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest).
Variational Mode Decomposition [Link](https://ieeexplore.ieee.org/document/6655981) method.
vmdpy [Link](https://github.com/vrcarva/vmdpy). paper [Link](https://www.sciencedirect.com/science/article/pii/S1746809420302299?via%3Dihub)
## Preprocessing 

## Main Code
The main architecture consists of two main compoenents: decomposition of spatiotemporal data and deep neural network:

### 1. Variational Mode Decomposition (VMD)

### 2. Neural Network 

#### Training 
```
python main/experiments/vmgcn/main.py --device cuda:0 --model_name astgcn --dataset SD --years 2019 --bs 48 --input_dim 16 --filename his.npz
```

```
python main/experiments/3d_vmgcn/main.py --device cuda:0 --model_name astgcn --dataset SD --years 2019 --bs 48 --input_dim 16 --filename his.npz
```
#### Evaluation

```
python main/experiments/3d_vmgcn/main.py --device cuda:0 --model_name astgcn --dataset SD --years 2019 --bs 48 --input_dim 16 --mode test --filename his.npz
```
