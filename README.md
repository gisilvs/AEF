# Closing the gap: Exact maximum likelihood training of generative autoencoders using invertible layers 

This repository contains the code used to run the experiments detailed in our paper [Closing the gap: Exact maximum likelihood training of generative autoencoders using invertible layers](https://arxiv.org/abs/2205.09546v1).

## Notebook
In `notebook.ipynb` we give a short tutorial on how to initialize and train an AEF model.

## Running experiments

---
To run experiments we provide a command line interface with the file `main_cli.py`, or `wandb_cli.py` which uses wandb to save experiment details. To train an AEF with a center mask on the MNIST dataset with a latent dimensionality of 2, run: 

    ./main_cli.py --model aef-center --dataset mnist --latent-dims 2 

Many more parameters can be specified such as number of training iterations, learning rate, and batch size. For more details, please consult `main_cli.py`.