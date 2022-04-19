from typing import Mapping, Union, Optional, Callable
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average
    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager
        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager


def save_in_dataframe(df_log, labels, mus, stddevs, epoch):
    df = pd.DataFrame()

    df['index'] = np.arange(len(mus[:, 0])) * epoch
    df['image_ind'] = np.arange(len(mus[:, 0]))
    df['class'] = labels.data.numpy().astype(str)
    df['mu_x'] = mus[:, 0]
    df['mu_y'] = mus[:, 1]
    df['std_x'] = stddevs[:, 0]
    df['std_y'] = stddevs[:, 1]
    df['epoch'] = np.ones(len(mus[:, 0])) * epoch

    df_log = pd.concat([df_log, df])

    return df_log


def run_on_testbatch(df_log, vae, epoch, x, y, device=None):
    with torch.no_grad():
        if device is not None:
            x = x.to(device)
        x, mus, stddevs = vae(x)
        x = x.to('cpu')
        mus = mus.to('cpu').data.numpy()
        stddevs = stddevs.to('cpu').mul(0.5).exp_().data.numpy()

    return save_in_dataframe(df_log, y, mus, stddevs, epoch)

def plot_loss(train_loss, val_loss=None):
    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, color='tab:blue')
    if val_loss is not None:
        plt.plot(np.arange(len(val_loss)), val_loss, color='tab:orange')
    plt.legend(["Training loss"] if val_loss is None else ["Training loss", "Validation loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Negative log likelihood")
    plt.show()

def refresh_bar(bar, desc):
    bar.set_description(desc)
    bar.refresh()

def dequantize(batch): # TODO: move somewhere else
    noise = torch.rand(*batch.shape)
    batch = (batch * 255. + noise) / 256.
    return batch