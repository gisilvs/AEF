import numpy as np

def get_avg_loss_over_iterations(iteration_losses: np.array, window_size: int, cur_iteration: int):
    low_window = max(0, cur_iteration - window_size)
    high_window = cur_iteration + 1
    return np.mean(iteration_losses[low_window:high_window])