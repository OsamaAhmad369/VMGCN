# Variational Mode Graph Convolutional Network (VMGCN) 
This repository is based on traffic forecasting on the LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf) that consists of three regions Greater Bay Area (GBA), Greater Los Angeles (GLA), and San Diego (SD). Our main code has been modified from the Github repo: [Link](https://github.com/liuxu77/LargeST). 

You can download all data from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest).

The Variational Mode Decomposition method is proposed by Konstantin Dragomiretskiy and Dominique Zosso  [Link](https://ieeexplore.ieee.org/document/6655981) to determine the modes of the signals. Vmdpy tool in Numpy [Link](https://github.com/vrcarva/vmdpy) is used in our code. The paper link for this tool is [Link](https://www.sciencedirect.com/science/article/pii/S1746809420302299?via%3Dihub).

Note:

Please cite all reference papers, if you find these papers useful in your research.

## Preprocessing 
To preprocessing the raw data, please refer to github repo [Link](https://github.com/liuxu77/LargeST). We set the sampling time for 15 minutes during preprocessing.
## Main Code
The main architecture consists of two main compoenents: decomposition of spatiotemporal data and deep neural network:

### 1. Variational Mode Decomposition (VMD)
The data for decomposition is arranged in the order of (time, nodes, features). The features in our work are concatenated such as counts, time of the day, day of the week. The output features of this decomposition will be (counts, time of the day, day of the week, modes). 
```
python main/preprocessing/vmd.py --dataset SD --years 2019 --f his.npz --alpha 2000 --K 13 --tol 1e-7 --DC 0 --init 1 --tau 0 --sigma 0.1
```
### 2. Neural Network 
Currently, there are total 2 supported codes available in this repository, VMGCN (vmgcn) and VMGCN with 3D attention (3d_vmgcn). We use the flow data from 2019 in our training and evaluation in our paper.  The backbone of the deep neural network is based on [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch).

#### Training 


Execute the Python file in the terminal

To run the VMGCN,
```
python main/experiments/vmgcn/main.py --device cuda:0 --model_name astgcn --dataset SD --years 2019 --bs 48 --input_dim 16 --filename his_vmd.npz 
```

To run the VMGCN with channel attention,
```
python main/experiments/3d_vmgcn/main.py --device cuda:0 --model_name astgcn --dataset SD --years 2019 --bs 48 --input_dim 16 --filename his_vmd.npz
```
#### Evaluation
We evaluate MAE, RMSE, and MAPE error metrics on prediction horizons 1 to 12. 

```
python main/experiments/3d_vmgcn/main.py --device cuda:0 --model_name astgcn --dataset SD --years 2019 --bs 48 --input_dim 16 --mode test --filename his_vmd.npz
```
