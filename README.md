# Spike-based BMI Decoding
This repository contains Python implementation of our proposed method for improving the decoding performance of spike-based BMIs which was published in IEEE Access (2022) [[1](https://ieeexplore.ieee.org/abstract/document/9733879)]. This paper proposes Bayesian adaptive kernel smoother (BAKS) [[2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206794)] as a firing rate estimation algorithm coupled with deep learning based quasi-recurrent neural network (QRNN) as a decoding algorithm. QRNN was developed by James Bradbury et al. from [Salesforce Research](https://github.com/salesforce/pytorch-qrnn) and implemented in Pytorch (see [here](https://arxiv.org/abs/1611.01576) for the paper). In this repository, the QRNN decoder is implemented in TensorFlow/Keras using a codebase from [here](https://github.com/DingKe/nn_playground/blob/master/qrnn/qrnn.py).

## Dataset
This repository uses neural data from publicly available dataset deposited on [Zenodo](https://zenodo.org/record/3854034) by Sabes lab. Details of the
recording setup and experimental task are described [here](https://iopscience.iop.org/article/10.1088/1741-2552/aa9e95).

## Requirements
This repository uses Python (v3.8.8) and the following packages:
- TensorFlow (v2.3.0)
- Scikit-learn (v0.24.1)
- Optuna (v2.10.0)
- Scipy (v1.6.2)
- Matplotlib (v3.3.4)

## Tutorial
Step-by-step tutorial is given in [tutorial.ipynb](https://github.com/nurahmadi/spike_bmi/blob/main/tutorial.ipynb).

## Citation
If you use this repository in a scientific publication, we would appreciate if you cite our paper. Here is the BibTeX citation format of our paper:

```
@article{ahmadi2022improved,
  title={Improved spike-based brain-machine interface using Bayesian adaptive kernel smoother and deep learning},
  author={Ahmadi, Nur and Adiono, Trio and Purwarianti, Ayu and Constandinou, Timothy G and Bouganis, Christos-Savvas},
  journal={IEEE Access},
  volume={10},
  pages={29341--29356},
  year={2022},
  publisher={IEEE}
}
```

## References
1. Ahmadi, N., Adiono, T., Purwarianti, A., Constandinou, T. G., & Bouganis, C. S. (2022). Improved spike-based brain-machine interface using Bayesian adaptive kernel smoother and deep learning. IEEE Access, 10, 29341-29356.
2. Ahmadi, N., Constandinou, T. G., & Bouganis, C. S. (2018). Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS). Plos one, 13(11), e0206794.
