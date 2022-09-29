# Closing the gap: Exact maximum likelihood training of generative autoencoders using invertible layers 


This repository contains the official implementation of our paper "[Closing the gap: Exact maximum likelihood training of generative autoencoders using invertible layers]".

![Image from abstract](figs/readme_samples_celeba64.png)

In this work, we provide an exact likelihood alternative to the variational training of generative autoencoders. This is achieved while leaving complete freedom in the choice of encoder, decoder and prior architectures, making our approach a drop-in replacement for the training of existing VAEs and VAE-style models. We show that the approach results in strikingly higher performance than architecturally equivalent VAEs in term of log-likelihood, sample quality and denoising performance. 

## Notebook

In `notebook.ipynb` we give a short tutorial on how to initialize and train an AEF model, and compare samples generated by an AEF to samples generated by a VAE with an equivalent architecture.

## Running experiments

To run experiments we provide a command line interface with the file `main_cli.py`, or `wandb_cli.py` which uses wandb to save experiment details. To train an AEF with a center mask on the MNIST dataset with a latent dimensionality of 2, run: 

    ./main_cli.py --model aef-center --dataset mnist --latent-dims 2 

To reproduce the experiments on CelebA-HQ resized to 64x64 with a latent dimensionality of 128, run:

    ./main_cli.py --model aef-linear --architecture big --posterior-flow maf --prior-flow maf --dataset celebahq64 --latent-dims 128 --iterations 1000000 --lr 1e-4 --batch-size 16 --early-stopping 100000 --data-dir [celebahq64-folder]
    ./main_cli.py --model vae --architecture big --posterior-flow iaf --prior-flow maf --dataset celebahq64 --latent-dims 128 --iterations 1000000 --lr 1e-4 --batch-size 16 --early-stopping 100000 --data-dir [celebahq64-folder]

 For more details, please consult `main_cli.py`.

## Data
For the CelebA-HQ experiments we used the 'data128x128.zip' file from [here](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P). It can be resized using

    data/process_celebahq.py --data-dir "download_folder/data128x128" --output-folder "celebahq64" --dimension 64 

for a size of 64x64.
## Samples

<table width="100%">
  <tr>
  <th>MNIST</th>
  <th>FashionMNIST</th>
  </tr>
  <tr>
  <td width="50%"><img src="https://raw.githubusercontent.com/gisilvs/AEF/a04985747ad6c60573d9556aba4926403d10a079/figs/nae-external_mnist_run_2_latent_size_32_decoder_independent_1-1.png"></td>
  <td width="50%"><img src="https://raw.githubusercontent.com/gisilvs/AEF/a04985747ad6c60573d9556aba4926403d10a079/figs/nae-external_fashionmnist_run_2_latent_size_32_decoder_independent_0-1.png"></td>
  </tr>
  <tr>
  <th>KMNIST</th>
  <th>CelebA-HQ</th>
  </tr>
  <tr>
  <td width="50%"><img src="https://raw.githubusercontent.com/gisilvs/AEF/update_notebook_readme/figs/nae-external_kmnist_run_4_latent_size_32_decoder_independent_0-1.png"></td>
  <td width="50%"><img src="https://raw.githubusercontent.com/gisilvs/AEF/update_notebook_readme/figs/celeba_samples_0.8_run_1_1-1.png"></td>
  </tr>
</table>

## Acknowledgements
This implementation uses parts of the code from the following Github repositories: [nflows](https://github.com/bayesiains/nflows), [rectangular-flows](https://github.com/layer6ai-labs/rectangular-flows), [pytorch-fid](https://github.com/mseitzer/pytorch-fid) as described in our code.