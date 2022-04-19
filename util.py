import numpy as np
import matplotlib.pyplot as plt

def get_avg_loss_over_iterations(iteration_losses: np.array, window_size: int, cur_iteration: int):
    low_window = max(0, cur_iteration - window_size)
    high_window = cur_iteration + 1
    return np.mean(iteration_losses[low_window:high_window])

def plot_loss_over_iterations(iterations_losses: np.array, val_losses: np.array = None,
                              val_iterations: np.array = None, window_size: int = 10):
    def moving_average(a, n=window_size):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = (ret[n:] - ret[:-n])/n
        ret[:n] = ret[:n] / np.arange(1, n+1)
        return ret


    plt.figure()

    plt.plot(np.arange(len(iterations_losses)), moving_average(iterations_losses), color='tab:blue')
    if val_losses is not None:
        plt.plot(val_iterations, val_losses, color='tab:orange')
    plt.legend(["Training loss"] if val_losses is None else ["Training loss", "Validation loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Negative log likelihood")
    plt.show()
