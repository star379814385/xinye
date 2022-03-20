import numpy as np
import torch
import os


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, mode="max", verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            mode (str):     "max" or "min"
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_best (bool):
            cover_best (bool):
        """
        self.patience = patience
        assert mode in ["max", "min"]
        self.mode = mode
        self.verbose = verbose
        self.counter = None
        self.best_score = None
        self.early_stop = False
        self.best_metrics = np.Inf if self.mode == "min" else -np.Inf
        self.delta = delta
        self.pre_save_path = None

    def step(self, metrics):
        score = metrics if self.mode == "max" else -metrics
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter:{}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("EarlyStopping!!!")
        else:
            self.best_score = score
            self.counter = 0

