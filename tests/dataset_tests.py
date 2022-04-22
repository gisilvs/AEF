from models.autoencoder_base import AutoEncoder
from datasets import get_train_val_dataloaders, get_list_of_datasets, get_test_dataloader
from models.vae import VAE


def test_processing_all_datasets():
    dataset_names = get_list_of_datasets()

    for dataset_name in dataset_names:
        train_dataloader, val_dataloader, img_dims, alpha = get_train_val_dataloaders(dataset_name)
        test_dataloader = get_test_dataloader(dataset_name)
        model = VAE(64, 2, img_dims)

        train_batch, _ = next(iter(train_dataloader))
        _ = model.loss_function(train_batch)

        val_batch, _ = next(iter(val_dataloader))
        _ = model.loss_function(val_batch)

        test_batch, _ = next(iter(test_dataloader))
        _ = model.loss_function(test_batch)
    return True

print(f"test_processing_all_datasets: {'Success' if test_processing_all_datasets() else 'Fail'}")