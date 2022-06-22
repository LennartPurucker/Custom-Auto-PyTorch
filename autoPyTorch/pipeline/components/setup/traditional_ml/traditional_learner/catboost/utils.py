from typing import Any, Dict
import logging
import time
import psutil
from enum import Enum


class AutoPyTorchToCatboostMetrics(Enum):
    mean_absolute_error = "MAE"
    root_mean_squared_error = "RMSE"
    mean_squared_log_error = "MSLE"
    r2 = "R2"
    accuracy = "Accuracy"
    balanced_accuracy = "BalancedAccuracy"
    f1 = "F1"
    roc_auc = "AUC"
    precision = "Precision"
    recall = "Recall"
    log_loss = "Logloss"


class MemoryCheckCallback:
    """
    Callback to ensure memory usage is safe, otherwise early stops the model to avoid OOM errors.

    This callback is CatBoost specific.

    Args:

        period : int, default = 10
            Number of iterations between checking memory status. Higher values are less precise but use less compute.
        verbose : bool, default = False
            Whether to log information on memory status even if memory usage is low.
    """
    def __init__(self, period: int = 10, verbose=False):
        self.period = period
        self.mem_status = psutil.Process()
        self.init_mem_rss = self.mem_status.memory_info().rss
        self.init_mem_avail = psutil.virtual_memory().available
        self.verbose = verbose

        self._cur_period = 1

    def after_iteration(self, info):
        iteration = info.iteration
        if iteration % self._cur_period == 0:
            not_enough_memory = self.memory_check(iteration)
            if not_enough_memory:
                return False
        return True

    def memory_check(self, iter) -> bool:
        """Checks if memory usage is unsafe. If so, then returns True to signal the model to stop training early."""
        available_bytes = psutil.virtual_memory().available
        cur_rss = self.mem_status.memory_info().rss

        if cur_rss < self.init_mem_rss:
            self.init_mem_rss = cur_rss
        estimated_model_size_mb = (cur_rss - self.init_mem_rss) >> 20
        available_mb = available_bytes >> 20
        model_size_memory_ratio = estimated_model_size_mb / available_mb

        early_stop = False
        if model_size_memory_ratio > 1.0:
            early_stop = True

        if available_mb < 512:  # Less than 500 MB
            early_stop = True

        if early_stop:
            return True
        elif self.verbose or (model_size_memory_ratio > 0.25):

            if model_size_memory_ratio > 0.5:
                self._cur_period = 1  # Increase rate of memory check if model gets large enough to cause OOM potentially
            elif iter > self.period:
                self._cur_period = self.period

        return False


class EarlyStoppingCallback:
    """
    Early stopping callback.

    This callback is CatBoost specific.

    Args:
        stopping_rounds : int or tuple
            If int, The possible number of rounds without the trend occurrence.
            If tuple, contains early stopping class as first element and class init kwargs as second element.
        eval_metric : str
            The eval_metric to use for early stopping. Must also be specified in the CatBoost model params.
        compare_key : str, default = 'validation'
            The data to use for scoring. It is recommended to keep as default.
    """
    def __init__(self, stopping_rounds, eval_metric, compare_key='validation'):
        if isinstance(stopping_rounds, int):
            from autoPyTorch.utils.early_stopping import SimpleEarlyStopper
            self.es = SimpleEarlyStopper(patience=stopping_rounds)
        else:
            self.es = stopping_rounds[0](**stopping_rounds[1])
        self.best_score = None
        self.compare_key = compare_key

        if isinstance(eval_metric, str):
            from catboost._catboost import is_maximizable_metric
            is_max_optimal = is_maximizable_metric(eval_metric)
            eval_metric_name = eval_metric
        else:
            is_max_optimal = eval_metric.is_max_optimal()

            eval_metric_name = eval_metric.__class__.__name__

        self.eval_metric_name = eval_metric_name
        self.is_max_optimal = is_max_optimal

    def after_iteration(self, info):
        is_best_iter = False
        cur_score = info.metrics[self.compare_key][self.eval_metric_name][-1]
        if not self.is_max_optimal:
            cur_score *= -1
        if self.best_score is None:
            self.best_score = cur_score
        elif cur_score > self.best_score:
            is_best_iter = True
            self.best_score = cur_score

        should_stop = self.es.update(current_epoch=info.iteration, is_best=is_best_iter)
        return not should_stop


def get_params(output_type: int) -> Dict[str, Any]:

	return {
	"iterations" : 10000,
	"learning_rate" : 0.1
}