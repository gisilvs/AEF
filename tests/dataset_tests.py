import torch

from datasets import get_train_val_dataloaders, get_list_of_datasets, get_test_dataloader
from models.autoencoder import IndependentVarianceDecoderSmall, ConvolutionalEncoderSmall
from models.vae import VAE


def test_processing_all_datasets():
    dataset_names = get_list_of_datasets()

    for dataset_name in dataset_names:
        train_dataloader, val_dataloader, img_dims, alpha = get_train_val_dataloaders(dataset_name)
        test_dataloader = get_test_dataloader(dataset_name)

        encoder = ConvolutionalEncoderSmall(32, img_dims, 2)
        decoder = IndependentVarianceDecoderSmall(32, img_dims, 2)
        model = VAE(encoder, decoder)

        train_batch, _ = next(iter(train_dataloader))
        _ = model.loss_function(train_batch)

        val_batch, _ = next(iter(val_dataloader))
        _ = model.loss_function(val_batch)

        test_batch, _ = next(iter(test_dataloader))
        _ = model.loss_function(test_batch)
    return True

def test_equality_validation_batches(n_to_check: int = 5):
    dataset_names = get_list_of_datasets()

    for dataset_name in dataset_names:
        _, val_dataloader, _, _ = get_train_val_dataloaders(dataset_name)
        _, val_dataloader_2, _, _ = get_train_val_dataloaders(dataset_name)
        val_dataloader_iterator_2 = iter(val_dataloader_2)
        n_checked = 0
        for val_batch, _ in val_dataloader:
            val_batch_2, _ = next(val_dataloader_iterator_2)
            if not torch.allclose(val_batch, val_batch_2, atol=1e-6):
                return False
            n_checked += 1
            if n_checked >= n_to_check:
                break
    return True


print(f"test_processing_all_datasets: {'Success' if test_processing_all_datasets() else 'Fail'}")
print(f"test_equality_validation_batches: {'Success' if test_equality_validation_batches() else 'Fail'}")