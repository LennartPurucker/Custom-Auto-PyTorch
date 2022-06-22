"""
Implementation from autogluon early_stopping
"""
from abc import ABC


class AbstractEarlyStopper(ABC):
    """
    Abstract class for early stopping
    """
    def update(self, current_epoch, is_best=False) -> bool:
        raise NotImplementedError

    def early_stop(self, current_epoch, is_best=False) -> bool:
        raise NotImplementedError


class SimpleEarlyStopper(AbstractEarlyStopper):
    """
    Implements early stopping with fixed patience
    Args:
    patience : int, default 10
        If no improvement occurs in `patience` epochs or greater, self.early_stop will return True.
    """
    def __init__(self, patience=10):
        self.patience = patience
        self.best_epoch = 0

    def update(self, current_epoch, is_best=False):
        if is_best:
            self.best_epoch = current_epoch
        return self.early_stop(current_epoch, is_best=is_best)

    def early_stop(self, current_epoch, is_best=False):
        if is_best:
            return False
        return current_epoch - self.best_epoch >= self.patience


def get_early_stopping_rounds(num_rows_train, min_patience=20, max_patience=300, min_rows=10000):

    modifier = 1 if num_rows_train <= min_rows else min_rows / num_rows_train
    simple_early_stopping_rounds = max(
        round(modifier * max_patience),
        min_patience,
    )
    return simple_early_stopping_rounds
