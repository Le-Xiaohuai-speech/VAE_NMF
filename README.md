# VAE_NMF
A semi-supervised single channel speech enhancement method using Variational Autoencoder and Non-negative Matrix Factorization

## Introduction
This repository remains the same strustures as [A VARIANCE MODELING FRAMEWORK BASED ON VARIATIONAL AUTOENCODERS FOR SPEECH ENHANCEMENT](https://gitlab.inria.fr/sileglai/mlsp-2018).
We replace MCMC-EM algorithm with backpropagation-EM algorithm for performing semi-supervised speech enhancement.

We use a variance modeling framework to model the speech and noise：</br>
![avatar](./research_page-1-768x480.png)</br>
VAE for speech modeling<br>
![avatar](./nmf.png)</br>
NMF for noise modeling</br>
## Running
