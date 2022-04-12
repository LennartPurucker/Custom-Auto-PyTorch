from multiprocessing.queues import Queue
import os
import re
import time
from timeit import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    MULTICLASSMULTIOUTPUT,
)
from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes, RepeatedCrossValTypes
from autoPyTorch.ensemble.stacking_ensemble_builder import calculate_nomalised_margin_loss
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.ensemble.stacking_ensemble import StackingEnsemble
from autoPyTorch.evaluation.utils import VotingRegressorWrapper
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr, subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = ['StackingEvaluator', 'eval_function']


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


class StackingEvaluator(AbstractEvaluator):
    """
    This class builds a pipeline using the provided configuration.
    A pipeline implementing the provided configuration is fitted
    using the datamanager object retrieved from disc, via the backend.
    After the pipeline is fitted, it is save to disc and the performance estimate
    is communicated to the main process via a Queue.

    Attributes:
        backend (Backend):
            An object to interface with the disk storage. In particular, allows to
            access the train and test datasets
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will return the evaluation result via a multiprocessing queue
        metric (autoPyTorchMetric):
            A scorer object that is able to evaluate how good a pipeline was fit. It
            is a wrapper on top of the actual score method (a wrapper on top of scikit
            lean accuracy for example) that formats the predictions accordingly.
        budget: (float):
            The amount of epochs/time a configuration is allowed to run.
        budget_type  (str):
            The budget type, which can be epochs or time
        pipeline_config (Optional[Dict[str, Any]]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
        configuration (Union[int, str, Configuration]):
            Determines the pipeline to be constructed. A dummy estimator is created for
            integer configurations, a traditional machine learning pipeline is created
            for string based configuration, and NAS is performed when a configuration
            object is passed.
        seed (int):
            A integer that allows for reproducibility of results
        output_y_hat_optimization (bool):
            Whether this worker should output the target predictions, so that they are
            stored on disk. Fundamentally, the resampling strategy might shuffle the
            Y_train targets, so we store the split in order to re-use them for ensemble
            selection.
        num_run (Optional[int]):
            An identifier of the current configuration being fit. This number is unique per
            configuration.
        include (Optional[Dict[str, Any]]):
            An optional dictionary to include components of the pipeline steps.
        exclude (Optional[Dict[str, Any]]):
            An optional dictionary to exclude components of the pipeline steps.
        disable_file_output (Union[bool, List[str]]):
            By default, the model, it's predictions and other metadata is stored on disk
            for each finished configuration. This argument allows the user to skip
            saving certain file type, for example the model, from being written to disk.
        init_params (Optional[Dict[str, Any]]):
            Optional argument that is passed to each pipeline step. It is the equivalent of
            kwargs for the pipeline steps.
        logger_port (Optional[int]):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel. If None is provided, a traditional
            logger is used.
        all_supported_metrics  (bool):
            Whether all supported metric should be calculated for every configuration.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            An object used to fine tune the hyperparameter search space of the pipeline
    """
    def __init__(self, backend: Backend, queue: Queue,
                 metric: autoPyTorchMetric,
                 budget: float,
                 configuration: Union[int, str, Configuration],
                 budget_type: str = None,
                 pipeline_config: Optional[Dict[str, Any]] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Union[bool, List] = False,
                 init_params: Optional[Dict[str, Any]] = None,
                 logger_port: Optional[int] = None,
                 all_supported_metrics: bool = True,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 use_ensemble_opt_loss=False) -> None:
        super().__init__(
            backend=backend,
            queue=queue,
            configuration=configuration,
            metric=metric,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
            budget=budget,
            budget_type=budget_type,
            logger_port=logger_port,
            all_supported_metrics=all_supported_metrics,
            pipeline_config=pipeline_config,
            search_space_updates=search_space_updates,
            use_ensemble_opt_loss=use_ensemble_opt_loss
        )

        self.splits = self.datamanager.splits
        self.num_repeats = len(self.splits)
        self.num_folds = len(self.splits[0])
        if self.splits is None:
            raise AttributeError("Must have called create_splits on {}".format(self.datamanager.__class__.__name__))

        self.logger.debug("use_ensemble_loss :{}".format(self.use_ensemble_opt_loss))

    def finish_up(self, loss: Dict[str, float], train_loss: Dict[str, float],
                  valid_pred: Optional[np.ndarray],
                  test_pred: Optional[np.ndarray],
                  pipeline_opt_pred: np.ndarray,
                  ensemble_opt_pred: np.ndarray,
                  additional_run_info: Optional[Dict],
                  file_output: bool, status: StatusType,                  
                  ) -> Optional[Tuple[float, float, int, Dict]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the necessary files
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                pipeline_opt_pred, valid_pred, test_pred
            )
        else:
            loss_ = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred
        )

        pipeline_loss, _ = self.calculate_auxiliary_losses(
            pipeline_opt_pred, None
        )
        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        cost = loss["ensemble_opt_loss"] if self.use_ensemble_opt_loss else loss[self.metric.name]

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss
        if pipeline_loss is not None:
            additional_run_info['pipeline_loss'] = pipeline_loss
        additional_run_info['opt_loss'] = loss
        rval_dict = {'loss': cost,
                     'additional_run_info': additional_run_info,
                     'status': status}

        self.queue.put(rval_dict)
        return None

    def get_sorted_preds(self, preds: List[List[np.ndarray]], repeat_id: int) -> np.ndarray:
        predictions = np.concatenate([pred for pred in preds if pred is not None])
        indices = np.concatenate([test_indices for _, test_indices in self.splits[repeat_id]])
        zipped_lists = zip(indices, predictions)

        sorted_zipped_lists = sorted(zipped_lists)
        predictions = [pred for _, pred in sorted_zipped_lists]
        return predictions

    def get_sorted_train_preds(self, preds: List[List[np.ndarray]], repeat_id: int):
        predictions = np.concatenate([pred for pred in preds if pred is not None])
        indices = np.concatenate([train_indices for train_indices, _ in self.splits[repeat_id]])

        unique_indices = set(indices)
        sorted_predictions = np.zeros((len(unique_indices), self.datamanager.num_classes))

        for i in unique_indices:
            positions = np.where(indices == i)
            tmp = list()
            for position in positions:
                tmp.append(predictions[position])
            mean_tmp = np.squeeze(np.mean(tmp, axis=1))
            for j, mean in enumerate(mean_tmp):
                sorted_predictions[i][j] = mean
        return sorted_predictions

    def get_sorted_train_targets(self, preds: List[List[np.ndarray]], repeat_id: int):
        predictions = np.concatenate([pred for pred in preds if pred is not None])
        indices = np.concatenate([train_indices for train_indices, _ in self.splits[repeat_id]])

        unique_indices = set(indices)
        sorted_predictions = np.zeros(len(unique_indices))

        for i in unique_indices:
            positions = np.where(indices == i)
            tmp = list()
            for position in positions:
                tmp.append(predictions[position])
            mean_tmp = np.squeeze(np.mean(tmp, axis=1))
            sorted_predictions[i] = mean_tmp
        return sorted_predictions

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Dict]:

        # Abort in case of shape misalignment
        if self.Y_optimization.shape[0] != Y_optimization_pred.shape[0]:
            return (
                1.0,
                {
                    'error':
                        "Targets %s and prediction %s don't have "
                        "the same length. Probably training didn't "
                        "finish" % (self.Y_optimization.shape, Y_optimization_pred.shape)
                },
            )

        # Abort if predictions contain NaNs
        for y, s in [
            # Y_train_pred deleted here. Fix unittest accordingly.
            [Y_optimization_pred, 'optimization'],
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test'],
        ]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {
                        'error':
                            'Model predictions for %s set contains NaNs.' % s
                    },
                )

        # Abort if we don't want to output anything.
        if hasattr(self, 'disable_file_output'):
            if self.disable_file_output:
                return None, {}
            else:
                self.disabled_file_outputs = []

        # This file can be written independently of the others down bellow
        if 'y_optimization' not in self.disabled_file_outputs:
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        if hasattr(self, 'pipelines') and self.pipelines is not None and isinstance(self.datamanager.resampling_strategy, RepeatedCrossValTypes):
            if self.pipelines[0] is not None and len(self.pipelines) > 0:
                if 'pipelines' not in self.disabled_file_outputs:
                    if self.task_type in CLASSIFICATION_TASKS:
                        pipelines = VotingClassifier(estimators=None, voting='soft', )
                    else:
                        pipelines = VotingRegressorWrapper(estimators=None)
                    pipelines.estimators_ = [pipeline for repeat_pipelines in self.pipelines for pipeline in repeat_pipelines]
                else:
                    pipelines = None
            else:
                pipelines = None
        else:
            pipelines = None

        if hasattr(self, 'pipeline') and self.pipeline is not None and isinstance(self.datamanager.resampling_strategy, HoldoutValTypes):
            if 'pipeline' not in self.disabled_file_outputs:
                pipeline = self.pipeline
            else:
                pipeline = None
        else:
            pipeline = None

        self.logger.debug("Saving model {}_{}_{} to disk".format(self.seed, self.num_run, self.budget))
        self.backend.save_numrun_to_dir(
            seed=int(self.seed),
            idx=int(self.num_run),
            budget=float(self.budget),
            model=self.pipelines[-1][-1],
            cv_model=pipelines,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in
                                       self.disabled_file_outputs else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in
                                self.disabled_file_outputs else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in
                               self.disabled_file_outputs else None
            ),
        )

        return None, {}

    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout"""
        assert self.splits is not None, "Can't fit pipeline in {} is datamanager.splits is None" \
            .format(self.__class__.__name__)

        Y_train_pred: List[List[Optional[np.ndarray]]] = [None] * self.num_repeats
        Y_pipeline_optimization_pred: List[List[Optional[np.ndarray]]] = [None] * self.num_repeats
        Y_valid_pred: List[List[Optional[np.ndarray]]] = [None] * self.num_repeats
        Y_test_pred: List[List[Optional[np.ndarray]]] = [None] * self.num_repeats
        # Y_train_targets: List[Optional[np.ndarray]] = [None] * self.num_folds
        # Y_targets: List[Optional[np.ndarray]] = [None] * self.num_folds


        self.pipelines = [[self._get_pipeline() for _ in range(self.num_folds)] for _ in range(self.num_repeats)]

        additional_run_info = {}


        for repeat_id, folds in enumerate(self.splits):
            y_train_pred_folds = [None] * self.num_folds
            y_pipeline_optimization_pred_folds = [None] * self.num_folds
            y_valid_pred_folds = [None] * self.num_folds
            y_test_pred_folds = [None] * self.num_folds
            # y_train_targets: List[Optional[np.ndarray]] = [None] * self.num_folds
            # y_targets: List[Optional[np.ndarray]] = [None] * self.num_folds

            for i, (train_split, test_split) in enumerate(folds):

                self.logger.info(f"Starting fit for repeat: {repeat_id} and fold: {i}")
                pipeline = self.pipelines[repeat_id][i]
                (
                    y_train_pred,
                    y_pipeline_opt_pred,
                    y_valid_pred,
                    y_test_pred,
                ) = self._fit_and_predict(pipeline, i, repeat_id,
                                        train_indices=train_split,
                                        test_indices=test_split)
                y_train_pred_folds[i] = y_train_pred
                y_pipeline_optimization_pred_folds[i] = y_pipeline_opt_pred
                if y_valid_pred is not None:
                    y_valid_pred_folds[i] = y_valid_pred
                if y_test_pred is not None:
                    y_test_pred_folds[i] = y_test_pred

                # y_train_targets[i] = self.y_train[train_split]
                # y_targets[i] = self.y_train[test_split]
                
                additional_run_info.update(pipeline.get_additional_run_info() if hasattr(
                    pipeline, 'get_additional_run_info') and pipeline.get_additional_run_info() is not None else {})

            Y_train_pred[repeat_id] = self.get_sorted_train_preds(y_train_pred_folds, repeat_id)
            Y_pipeline_optimization_pred[repeat_id] = self.get_sorted_preds(y_pipeline_optimization_pred_folds, repeat_id)
            if self.X_valid is not None:
                Y_valid_pred[repeat_id] = np.array([y_valid_pred_folds[i] for i in range(self.num_folds) if y_valid_pred_folds[i] is not None])
                # Average the predictions of several pipelines
                if len(Y_valid_pred[repeat_id].shape) == 3:
                    Y_valid_pred[repeat_id] = np.nanmean(Y_valid_pred[repeat_id], axis=0)
            else:
                Y_valid_pred = None

            if self.X_test is not None:
                Y_test_pred[repeat_id] = np.array([y_test_pred_folds[i] for i in range(self.num_folds) if y_test_pred_folds[i] is not None])
                # Average the predictions of several pipelines of the folds
                if len(Y_test_pred[repeat_id].shape) == 3:
                    Y_test_pred[repeat_id] = np.nanmean(Y_test_pred[repeat_id], axis=0)
            else:
                Y_test_pred = None

        # # as targets do change within repeats
        # Y_targets = self.y_train.copy() # self.get_sorted_preds(y_targets, -1)
        # Y_train_targets = self.y_train.copy() # self.get_sorted_train_targets(y_train_targets, -1)

        # Average prediction values accross repeats
        Y_train_pred = np.mean(Y_train_pred, axis=0)
        Y_pipeline_optimization_pred = np.mean(Y_pipeline_optimization_pred, axis=0)
        Y_valid_pred = np.mean(Y_valid_pred, axis=0) if Y_valid_pred is not None else None
        Y_test_pred = np.mean(Y_test_pred, axis=0) if Y_test_pred is not None else None

        ensemble_dir = self.backend.get_ensemble_dir()
        if os.path.exists(ensemble_dir) and len(os.listdir(ensemble_dir)) >= 1:
            old_ensemble = self.backend.load_ensemble(self.seed)
            assert isinstance(old_ensemble, StackingEnsemble)
            Y_ensemble_optimization_pred = old_ensemble.predict_with_current_pipeline(Y_pipeline_optimization_pred)
            Y_ensemble_preds = old_ensemble.get_ensemble_predictions_with_current_pipeline(Y_pipeline_optimization_pred)
        else:
            Y_ensemble_optimization_pred = Y_pipeline_optimization_pred.copy()
            Y_ensemble_preds = [Y_pipeline_optimization_pred]

        self.Y_optimization = self.y_train # np.array(Y_targets)
        self.Y_actual_train = self.y_train # np.array(Y_train_targets)

        self.pipeline = self._get_pipeline()

        train_loss = self._loss(self.Y_actual_train, Y_train_pred)
        opt_loss = self._loss(self.Y_optimization, Y_ensemble_optimization_pred)

        status = StatusType.SUCCESS
        self.logger.debug("In train evaluator fit_predict_and_loss, num_run: {} loss:{}".format(
            self.num_run,
            opt_loss
        ))
        self.finish_up(
            loss=opt_loss,
            train_loss=train_loss,
            ensemble_opt_pred=Y_ensemble_optimization_pred,
            valid_pred=Y_valid_pred,
            test_pred=Y_test_pred,
            additional_run_info=additional_run_info,
            file_output=True,
            status=status,
            pipeline_opt_pred=Y_pipeline_optimization_pred
        )

    def _fit_and_predict(
        self,
        pipeline: BaseEstimator,
        fold: int,
        repeat_id: int,
        train_indices: Union[np.ndarray, List],
        test_indices: Union[np.ndarray, List],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:

        # See autoPyTorch/pipeline/components/base_component.py::autoPyTorchComponent for more details
        # about fit_dictionary
        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': fold,
             'repeat_id': repeat_id,
             'num_run': self.num_run,
             **self.fit_dictionary}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, pipeline, X, y)
        self.logger.info("Model fitted, now predicting")
        (
            Y_train_pred, Y_pipeline_opt_pred, Y_valid_pred, Y_test_pred
        ) = self._predict(
            pipeline,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        self.pipeline = pipeline

        return Y_train_pred, Y_pipeline_opt_pred, Y_valid_pred, Y_test_pred

    def _predict(
        self,
        pipeline: BaseEstimator,
        test_indices: Union[np.ndarray, List],
        train_indices: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        train_pred = self.predict_function(subsampler(self.X_train, train_indices), pipeline,
                                           self.y_train[train_indices])

        pipeline_opt_pred = self.predict_function(subsampler(self.X_train, test_indices), pipeline,
                                         self.y_train[train_indices])

        # self.logger.debug(f"for model {self.seed}_{self.num_run}_{self.budget} ensemble_predictions are {ensemble_opt_pred}")
        if self.X_valid is not None:
            valid_pred = self.predict_function(self.X_valid, pipeline,
                                               self.y_valid)
        else:
            valid_pred = None

        if self.X_test is not None:
            test_pred = self.predict_function(self.X_test, pipeline,
                                              self.y_train[train_indices])
        else:
            test_pred = None

        return train_pred, pipeline_opt_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def eval_function(
    backend: Backend,
    queue: Queue,
    metric: autoPyTorchMetric,
    budget: float,
    config: Optional[Configuration],
    seed: int,
    num_run: int,
    include: Optional[Dict[str, Any]],
    exclude: Optional[Dict[str, Any]],
    disable_file_output: Union[bool, List],
    output_y_hat_optimization: bool,
    pipeline_config: Optional[Dict[str, Any]] = None,
    budget_type: str = None,
    init_params: Optional[Dict[str, Any]] = None,
    logger_port: Optional[int] = None,
    all_supported_metrics: bool = True,
    search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
    use_ensemble_opt_loss=False,
    instance: str = None,
) -> None:
    """
    This closure allows the communication between the ExecuteTaFuncWithQueue and the
    pipeline trainer (TrainEvaluator).

    Fundamentally, smac calls the ExecuteTaFuncWithQueue.run() method, which internally
    builds a TrainEvaluator. The TrainEvaluator builds a pipeline, stores the output files
    to disc via the backend, and puts the performance result of the run in the queue.


    Attributes:
        backend (Backend):
            An object to interface with the disk storage. In particular, allows to
            access the train and test datasets
        queue (Queue):
            Each worker available will instantiate an evaluator, and after completion,
            it will return the evaluation result via a multiprocessing queue
        metric (autoPyTorchMetric):
            A scorer object that is able to evaluate how good a pipeline was fit. It
            is a wrapper on top of the actual score method (a wrapper on top of scikit
            lean accuracy for example) that formats the predictions accordingly.
        budget: (float):
            The amount of epochs/time a configuration is allowed to run.
        budget_type  (str):
            The budget type, which can be epochs or time
        pipeline_config (Optional[Dict[str, Any]]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
        config (Union[int, str, Configuration]):
            Determines the pipeline to be constructed.
        seed (int):
            A integer that allows for reproducibility of results
        output_y_hat_optimization (bool):
            Whether this worker should output the target predictions, so that they are
            stored on disk. Fundamentally, the resampling strategy might shuffle the
            Y_train targets, so we store the split in order to re-use them for ensemble
            selection.
        num_run (Optional[int]):
            An identifier of the current configuration being fit. This number is unique per
            configuration.
        include (Optional[Dict[str, Any]]):
            An optional dictionary to include components of the pipeline steps.
        exclude (Optional[Dict[str, Any]]):
            An optional dictionary to exclude components of the pipeline steps.
        disable_file_output (Union[bool, List[str]]):
            By default, the model, it's predictions and other metadata is stored on disk
            for each finished configuration. This argument allows the user to skip
            saving certain file type, for example the model, from being written to disk.
        init_params (Optional[Dict[str, Any]]):
            Optional argument that is passed to each pipeline step. It is the equivalent of
            kwargs for the pipeline steps.
        logger_port (Optional[int]):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel. If None is provided, a traditional
            logger is used.
        instance (str):
            An instance on which to evaluate the current pipeline. By default we work
            with a single instance, being the provided X_train, y_train of a single dataset.
            This instance is a compatibility argument for SMAC, that is capable of working
            with multiple datasets at the same time.
    """
    evaluator = StackingEvaluator(
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        logger_port=logger_port,
        all_supported_metrics=all_supported_metrics,
        pipeline_config=pipeline_config,
        search_space_updates=search_space_updates,
        use_ensemble_opt_loss=use_ensemble_opt_loss
    )
    evaluator.fit_predict_and_loss()
