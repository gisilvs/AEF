import torch

def dequantize(batch):
    noise = torch.rand(*batch.shape)
    batch = (batch * 255. + noise) / 256.
    return batch