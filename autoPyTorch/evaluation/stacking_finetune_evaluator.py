from math import floor
from multiprocessing.queues import Queue
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration
from autoPyTorch.utils.common import read_predictions
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.ensemble.stacking_finetune_ensemble import StackingFineTuneEnsemble
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

__all__ = ['EnsembleOptimisationEvaluator', 'eval_stacking_finetune_function']


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


class StackingFineTuneEvaluator(RepeatedCrossValEvaluator):
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
                 use_ensemble_opt_loss=False,
                 cur_stacking_layer: int = 0,
                 mode='train',
                 previous_model_identifier: Optional[Tuple[int, int, float]] = None,
                 hpo_dataset_path: Optional[str] = None,
                 lower_layer_model_identifiers: Optional[List[Tuple[int, int, float]]] = None
    ) -> None:

        self.hpo_dataset_path = hpo_dataset_path
        self.cur_stacking_layer = cur_stacking_layer
        self.lower_layer_model_identifiers = lower_layer_model_identifiers

        self.mode = mode
        if self.mode == 'hpo':
            # previous model indentifier allows us to restore the weights of the
            # random configuration model trained in the training phase of fine tune
            # stacking.
            if previous_model_identifier is None:
                raise ValueError(f"Expected previous_model_identifier when the mode is hpo")
            model_weights_path = os.path.join(backend.internals_directory, "pretrained_weights", f"{previous_model_identifier[0]}_{previous_model_identifier[1]}_{float(previous_model_identifier[2])}")
            # backend.get_numrun_directory(*previous_model_identifier)
        else:
            model_weights_path = os.path.join(backend.internals_directory, "pretrained_weights", f"{seed}_{num_run}_{float(budget)}")
            if not os.path.exists(model_weights_path):
                os.makedirs(model_weights_path)

        self.model_weights_path = model_weights_path

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
        self.old_ensemble: Optional[StackingFineTuneEnsemble] = None
        # ensemble_dir = self.backend.get_ensemble_dir()
        # if os.path.exists(ensemble_dir) and len(os.listdir(ensemble_dir)) >= 1:
        #     self.old_ensemble = self.backend.load_ensemble(self.seed)
        #     assert isinstance(self.old_ensemble, StackingFineTuneEnsemble)

        self.logger.debug(f"for num run: {num_run}, X_train.shape: {self.X_train.shape} and X_test.shape: {self.X_test.shape}")


    def _init_fit_dictionary(
        self,
        logger_port: int,
        pipeline_config: Dict[str, Any],
        metrics_dict: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super()._init_fit_dictionary(logger_port=logger_port, pipeline_config=pipeline_config, metrics_dict=metrics_dict)
        self.fit_dictionary.update({'mode': self.mode, 'model_weights_path': self.model_weights_path})

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Dict]:
        output = super().file_output(Y_optimization_pred=Y_optimization_pred, Y_valid_pred=Y_valid_pred, Y_test_pred=Y_test_pred)
        return output

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

        hpo_preds = {}
        if self.mode == 'train':
            if self.hpo_dataset_path is None:
                raise ValueError(f"Expected hpo_dataset_path to not be None if mode: {self.mode}")
            train_hpo_dataset: TabularDataset = pickle.load(open(self.hpo_dataset_path, 'rb'))
            X_train = train_hpo_dataset.train_tensors[0].copy()
            X_test = train_hpo_dataset.test_tensors[0].copy()
            y_train = train_hpo_dataset.train_tensors[0]
            self.logger.debug(f"Shape before apending: {X_train.shape}")
            if self.cur_stacking_layer != 0:
                run_history_pred_path = None # os.path.join(self.backend.internals_directory, 'evaluator_hpo_read_preds.pkl')
                ensemble_predictions = read_predictions(self.backend, self.seed, 0, 32, run_history_pred_path=run_history_pred_path, data_set='hpo_ensemble')
                test_predictions = read_predictions(self.backend, self.seed, 0, 32, run_history_pred_path=run_history_pred_path, data_set='hpo_test')
                hpo_ensemble_predictions = []
                hpo_test_predictions = []
                for identifier in self.lower_layer_model_identifiers:
                    hpo_ensemble_predictions.append(ensemble_predictions[identifier])
                    hpo_test_predictions.append(test_predictions[identifier])
                X_train = np.concatenate([X_train , *hpo_ensemble_predictions], axis=1)
                X_test = np.concatenate([X_test, *hpo_test_predictions], axis=1)
                self.logger.debug(f"Shape after apending: {X_train.shape}, len hpo_ensemble_predictions : {len(hpo_ensemble_predictions)}")

            if self.task_type in CLASSIFICATION_TASKS:
                pipelines = VotingClassifier(estimators=None, voting='soft', )
            pipelines.estimators_ = [pipeline for repeat_pipelines in self.pipelines for pipeline in repeat_pipelines if check_pipeline_is_fitted(pipeline, self.configuration)]

            hpo_preds['hpo_ensemble'] = pipelines.predict_proba(X_train)
            hpo_preds['hpo_ensemble'] = self._ensure_prediction_array_sizes(hpo_preds['hpo_ensemble'], y_train)
            hpo_preds['hpo_test'] = pipelines.predict_proba(X_test)
            hpo_preds['hpo_test'] = self._ensure_prediction_array_sizes(hpo_preds['hpo_test'], y_train)
            

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
        if hpo_preds.get('hpo_ensemble', None) is not None:
            identifier = (self.seed, self.num_run, float(self.budget))
            num_run_dir = self.backend.get_numrun_directory(*identifier)
            if not os.path.exists(num_run_dir):
                os.makedirs(num_run_dir)
            for subset in ['hpo_ensemble', 'hpo_test']:
                file_path = os.path.join(num_run_dir, self.backend.get_prediction_filename(subset, *identifier))
                with open(file_path, "wb") as fh:
                    pickle.dump(hpo_preds[subset].astype(np.float32), fh, -1)

    def _fit_predict_one_fold(
        self,
        additional_run_info,
        total_repeats,
        repeat_id,
        y_optimization_pred_folds,
        y_valid_pred_folds,
        y_test_pred_folds,
        i,
        train_split,
        test_split
    ):
        fold_model_weights_path = os.path.join(self.model_weights_path, f"repeat_{repeat_id}", f"split_{i}")
        if not os.path.exists(fold_model_weights_path):
            os.makedirs(fold_model_weights_path)
        return super()._fit_predict_one_fold(
            additional_run_info=additional_run_info,
            total_repeats=total_repeats,
            repeat_id=repeat_id,
            y_optimization_pred_folds=y_optimization_pred_folds,
            y_valid_pred_folds=y_valid_pred_folds,
            y_test_pred_folds=y_test_pred_folds,
            i=i,
            train_split=train_split,
            test_split=test_split
        )

    def _predict_with_stacked_ensemble(self, X, Y_pipeline_optimization_pred):

        pass



# create closure for evaluating an algorithm
def eval_stacking_finetune_function(
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
    mode='train',
    previous_model_identifier: Optional[Tuple[int, int, float]] = None,
    cur_stacking_layer: int = 0,
    hpo_dataset_path: Optional[str] = None,
    lower_layer_model_identifiers: Optional[List[Tuple[int, int, float]]] = None,
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
    evaluator = StackingFineTuneEvaluator(
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
        mode=mode,
        previous_model_identifier=previous_model_identifier,
        cur_stacking_layer=cur_stacking_layer,
        hpo_dataset_path=hpo_dataset_path,
        lower_layer_model_identifiers=lower_layer_model_identifiers,
    )
    evaluator.fit_predict_and_loss()
