# Adaptive Polyak Step-Size for Momentum Accelerated Stochastic Gradient Descent

This repository contains the code for the **SGDM-APS** algorithm proposed in the paper **"Adaptive Polyak Step-Size for Momentum Accelerated Stochastic Gradient Descent with General Convergence Guarantee"**, published in *IEEE Transactions on Signal Processing (TSP)*.

## Overview
This project builds upon the implementation from [SPS repository](https://github.com/IssamLaradji/sps). The code is implemented in Python and requires PyTorch.

## Requirements
To run the code, make sure you have the following installed:
- Python (>=3.7)
- PyTorch (>=1.8)

Install PyTorch by following the instructions on [pytorch.org](https://pytorch.org/get-started/).

## Usage
To run the main script, use the following command:
```bash
python main.py --beta [beta] --c [c] --omega [omega] --bs [bs] --max_epoch [max_epoch] --task [task]
```
beta: momentum parameter, default 0.9
c: SGDM-APS hyperparameter, default 0.2
bs: batch size
max_epoch: training epochs
task: selected from 'ijcnn', 'mushrooms', 'rcv1', 'w8a', 'matrix_1', 'matrix_4', 'matrix_10', 'cifar10', 'cifar100', 'cifar10_densenet', 'cifar100_densenet'

### Example Parameters
Here are some examples:
**ijcnn (or mushrooms, rcv1, w8a):**
   ```bash
   python main.py --beta 0.9 --c 0.2 --bs 100 --max_epoch 35 --task ijcnn
   ```
**matrix:**
   ```bash
   python main.py --beta 0.9 --c 0.2 --bs 100 --max_epoch 50 --task matrix_4
   ```
**cifar10 (or cifar100):**
   ```bash
   python main.py --beta 0.9 --c 0.2 --bs 128 --max_epoch 200 --task cifar10
   ```

## Citation
If you use this code in your research, feel free to cite our paper:
```bibtex
@ARTICLE{10836899,
  author={Zhang, Jiawei and Jin, Cheng and Gu, Yuantao},
  journal={IEEE Transactions on Signal Processing}, 
  title={Adaptive Polyak Step-Size for Momentum Accelerated Stochastic Gradient Descent with General Convergence Guarantee}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Convergence;Optimization;Signal processing algorithms;Lower bound;Interpolation;Tuning;Manuals;Convex functions;Upper bound;Training;Optimization;SGDM;momentum;Polyak stepsize;adaptive step size},
  doi={10.1109/TSP.2025.3528217}
}
```

## Acknowledgements
This code is built upon the work in the [SPS repository](https://github.com/IssamLaradji/sps). We thank the authors for making their code available.

---
For any questions or issues, feel free to open an issue in this repository or contact the authors of the paper.
