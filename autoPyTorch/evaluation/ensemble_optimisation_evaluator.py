from math import floor
from multiprocessing.queues import Queue
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration
from autoPyTorch.evaluation.repeated_crossval_evaluator import RepeatedCrossValEvaluator

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
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble_builder import calculate_nomalised_margin_loss
from autoPyTorch.ensemble.iterative_hpo_stacking_ensemble import IterativeHPOStackingEnsemble
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble import EnsembleOptimisationStackingEnsemble
from autoPyTorch.evaluation.utils import VotingRegressorWrapper, check_pipeline_is_fitted
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import dict_repr, subsampler
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates

__all__ = ['EnsembleOptimisationEvaluator', 'eval_ensemble_optimise_function']


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


class EnsembleOptimisationEvaluator(RepeatedCrossValEvaluator):
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
        self.logger.debug("use_ensemble_loss :{}".format(self.use_ensemble_opt_loss))
        self.old_ensemble: Optional[Union[EnsembleOptimisationStackingEnsemble, IterativeHPOStackingEnsemble]] = None
        ensemble_dir = self.backend.get_ensemble_dir()
        if os.path.exists(ensemble_dir) and len(os.listdir(ensemble_dir)) >= 1:
            self.old_ensemble = self.backend.load_ensemble(self.seed)
            assert isinstance(self.old_ensemble, (EnsembleOptimisationStackingEnsemble, IterativeHPOStackingEnsemble))

        self.logger.debug(f"for num run: {num_run}, X_train.shape: {self.X_train.shape} and X_test.shape: {self.X_test.shape}")

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
        if pipeline_loss is not None:
            additional_run_info['pipeline_loss'] = pipeline_loss
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss
        additional_run_info['configuration'] = self.configuration if not isinstance(self.configuration, Configuration) else self.configuration.get_dictionary()
        additional_run_info['budget'] = self.budget

        additional_run_info['opt_loss'] = loss
        rval_dict = {'loss': cost,
                     'additional_run_info': additional_run_info,
                     'status': status}

        self.queue.put(rval_dict)
        return None


    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout"""
        assert self.splits is not None, "Can't fit pipeline in {} is datamanager.splits is None" \
            .format(self.__class__.__name__)

        Y_train_pred, Y_pipeline_optimization_pred, Y_valid_pred, Y_test_pred, additional_run_info = self._run_fit_predict_repeats()

        if self.old_ensemble is not None:
            Y_ensemble_optimization_pred = self.old_ensemble.predict_with_current_pipeline(Y_pipeline_optimization_pred)
            Y_ensemble_preds = self.old_ensemble.get_ensemble_predictions_with_current_pipeline(Y_pipeline_optimization_pred)
            self.logger.debug(f"old ensemble has {self.old_ensemble.stacked_ensemble_identifiers}")
        else:
            Y_ensemble_optimization_pred = Y_pipeline_optimization_pred.copy()
            Y_ensemble_preds = [Y_pipeline_optimization_pred]


        train_loss = None # self._loss(self.Y_actual_train, Y_train_pred)
        opt_loss = self._loss(self.Y_optimization, Y_ensemble_optimization_pred)

        opt_loss ['ensemble_opt_loss'] = calculate_nomalised_margin_loss(Y_ensemble_preds, self.Y_optimization)
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


# create closure for evaluating an algorithm
def eval_ensemble_optimise_function(
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
    evaluator = EnsembleOptimisationEvaluator(
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
        use_ensemble_opt_loss=use_ensemble_opt_loss,
    )
    evaluator.fit_predict_and_loss()
