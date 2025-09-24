# EfficientNet1D_FRB

This repository contains a code used for detecting fast radio bursts (FRB) on the RATAN-600 radio telescope using 1D convolutional neural network ([Astronomy & Computing, 2026, 101002](https://doi.org/10.1016/j.ascom.2025.101002); the accepted manuscript is also available on [arXiv:2509.11215](https://arxiv.org/abs/2509.11215)). The parts are the following.

* EfficientNet1D-XS: a 1D conlolutional neural network architecture based on the  EfficientNet family of models (Tan & Le, [2019](https://proceedings.mlr.press/v97/tan19a.html), [2021](https://proceedings.mlr.press/v139/tan21a.html)). Although used here for a particular purpose, the network can be used to classify arbitrary multichannel time series.
* A code to generate synthetic FRB events. Partially based on [injectfrb](https://github.com/liamconnor/injectfrb).

## 1. EfficientNet1D-XS

A 1D convolutional neural network for classification of multichannel time series. The network is coded as a PyTorch class (torch.nn.Module) and can be trained using the [standard PyTorch training workflow](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). 

Alternatively, the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) library may be used for the training. A general layout of the training procedure is provided in a Jupiter notebook: [training_example.ipynb](training_example.ipynb).

## 2. Generation of synthetic FRB events

The Jupiter notebook [synthetic_frb.ipynb](synthetic_frb.ipynb) gives an example for generation of a synthetic FRB event.

## 3. Dependencies

Package dependencies are listed in [requirements.txt](requirements.txt). The versions do not have to be the same as specified in the file, the code will most probably work with any of the recent versions.

## 4. Files

* [data](data): mock dataset (4-channel time series in binary format)
* [modules/event.py](modules/event.py): FRB event generation code
* [modules/networks.py](modules/networks.py): neural network architecture
* [modules/funcs.py](modules/funcs.py): service functions
* [synthetic_frb.ipynb](synthetic_frb.ipynb): an example of synthetic FRB generation
* [training_example.ipynb](training_example.ipynb): general layout of the training with PyTorch Lightning
* [requirements.txt](requirements.txt) package dependencies
