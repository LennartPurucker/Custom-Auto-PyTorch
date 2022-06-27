import copy
import json
import logging.handlers
import math
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from turtle import pos
import typing
import unittest.mock
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import dask
import dask.distributed

import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import DataOrigin, RunHistory, RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType

from autoPyTorch import metrics
from autoPyTorch.api.utils import get_autogluon_default_nn_config, get_config_from_run_history
from autoPyTorch.automl_common.common.utils.backend import Backend, create
from autoPyTorch.constants import (
    REGRESSION_TASKS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
)
from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.data.utils import DatasetCompressionSpec
from autoPyTorch.datasets.base_dataset import BaseDataset, BaseDatasetPropertiesType
from autoPyTorch.datasets.resampling_strategy import (
    CrossValTypes,
    HoldoutValTypes,
    NoResamplingStrategyTypes,
    ResamplingStrategies,
    RepeatedCrossValTypes
)
from autoPyTorch.datasets.utils import get_appended_dataset
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.repeat_models_stacking_ensemble import RepeatModelsStackingEnsemble
from autoPyTorch.ensemble.ensemble_builder_manager import EnsembleBuilderManager
from autoPyTorch.ensemble.singlebest_ensemble import SingleBest
from autoPyTorch.ensemble.autogluon_stacking_ensemble import AutogluonStackingEnsemble
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble import EnsembleOptimisationStackingEnsemble
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble import EnsembleSelectionPerLayerStackingEnsemble
from autoPyTorch.ensemble.utils import BaseLayerEnsembleSelectionTypes, StackingEnsembleSelectionTypes, is_stacking
from autoPyTorch.evaluation.abstract_evaluator import fit_and_suppress_warnings
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.evaluation.utils import DisableFileOutputParameters
from autoPyTorch.optimizer.run_history_callback import RunHistoryUpdaterManager
from autoPyTorch.optimizer.smbo import AutoMLSMBO
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import get_available_traditional_learners
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics
from autoPyTorch.utils.common import FitRequirement, ENSEMBLE_ITERATION_MULTIPLIER, TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT, dict_repr, replace_string_bool_to_bool, validate_config
from autoPyTorch.utils.parallel_model_runner import run_models_on_dataset
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import (
    PicklableClientLogger,
    get_named_client_logger,
    setup_logger,
    start_log_server,
)
from autoPyTorch.utils.parallel import preload_modules
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from autoPyTorch.utils.results_manager import MetricResults, ResultsManager, SearchResults
from autoPyTorch.utils.results_visualizer import ColorLabelSettings, PlotSettingParams, ResultsVisualizer
from autoPyTorch.utils.single_thread_client import SingleThreadedClient
from autoPyTorch.utils.stopwatch import StopWatch


def _pipeline_predict(pipeline: BasePipeline,
                      X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int,
                      logger: PicklableClientLogger,
                      task: int) -> np.ndarray:
    @typing.no_type_check
    def send_warnings_to_log(
            message, category, filename, lineno, file=None, line=None):
        logger.debug('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))
        return

    X_ = X.copy()
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        if task in REGRESSION_TASKS:
            # Voting regressor does not support batch size
            prediction = pipeline.predict(X_)
        else:
            # Voting classifier predict proba does not support batch size
            prediction = pipeline.predict_proba(X_)
            # Check that all probability values lie between 0 and 1.
            if not ((prediction >= 0).all() and (prediction <= 1).all()):
                np.set_printoptions(threshold=sys.maxsize)
                raise ValueError("For {}, prediction probability not within [0, 1]: {}/{}!".format(
                    pipeline,
                    prediction,
                    np.sum(prediction, axis=1)
                ))

    if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
            X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
        logger.warning(
            "Prediction shape for model %s is %s while X_.shape is %s",
            pipeline, str(prediction.shape), str(X_.shape)
        )
    return prediction


class BaseTask(ABC):
    """
    Base class for the tasks that serve as API to the pipelines.

    Args:
        seed (int: default=1):
            Seed to be used for reproducibility.
        n_jobs (int: default=1):
            Number of consecutive processes to spawn.
        n_threads (int: default=1):
            Number of threads to use for each process.
        logging_config (Optional[Dict]):
            Specifies configuration for logging, if None, it is loaded from the logging.yaml
        ensemble_size (int: default=50):
            Number of models added to the ensemble built by
            Ensemble selection from libraries of models.
            Models are drawn with replacement.
        ensemble_nbest (int: default=50):
            Only consider the ensemble_nbest models to build the ensemble
        max_models_on_disc (int: default=50):
            Maximum number of models saved to disc. It also controls the size of
            the ensemble as any additional models will be deleted.
            Must be greater than or equal to 1.
        temporary_directory (str):
            Folder to store configuration output and log file
        output_directory (str):
            Folder to store predictions for optional test set
        delete_tmp_folder_after_terminate (bool):
            Determines whether to delete the temporary directory,
            when finished
        include_components (Optional[Dict[str, Any]]):
            Dictionary containing components to include. Key is the node
            name and Value is an Iterable of the names of the components
            to include. Only these components will be present in the
            search space.
        exclude_components (Optional[Dict[str, Any]]):
            Dictionary containing components to exclude. Key is the node
            name and Value is an Iterable of the names of the components
            to exclude. All except these components will be present in
            the search space.
        resampling_strategy resampling_strategy (RESAMPLING_STRATEGIES),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
        resampling_strategy_args (Optional[Dict[str, Any]]): arguments
            required for the chosen resampling strategy. If None, uses
            the default values provided in DEFAULT_RESAMPLING_PARAMETERS
            in ```datasets/resampling_strategy.py```.
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            Search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline
    """

    def __init__(
        self,
        seed: int = 1,
        n_jobs: int = 1,
        n_threads: int = 1,
        logging_config: Optional[Dict] = None,
        ensemble_size: int = 5,
        ensemble_nbest: int = 50,
        base_ensemble_method: BaseLayerEnsembleSelectionTypes = BaseLayerEnsembleSelectionTypes.ensemble_selection,
        stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None,
        use_ensemble_opt_loss: bool = False,
        num_stacking_layers: int = 1,
        max_models_on_disc: int = 50,
        temporary_directory: Optional[str] = None,
        output_directory: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        delete_output_folder_after_terminate: bool = True,
        include_components: Optional[Dict[str, Any]] = None,
        exclude_components: Optional[Dict[str, Any]] = None,
        backend: Optional[Backend] = None,
        resampling_strategy: ResamplingStrategies = HoldoutValTypes.holdout_validation,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
        task_type: Optional[str] = None
    ) -> None:

        if isinstance(resampling_strategy, NoResamplingStrategyTypes) and ensemble_size != 0:
            raise ValueError("`NoResamplingStrategy` cannot be used for ensemble construction")

        self.seed = seed
        self.n_jobs = n_jobs
        self.n_threads = n_threads
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.base_ensemble_method = base_ensemble_method
        self.stacking_ensemble_method = stacking_ensemble_method 
        self.num_stacking_layers = num_stacking_layers
        self.use_ensemble_opt_loss = use_ensemble_opt_loss

        self.max_models_on_disc = max_models_on_disc
        self.logging_config: Optional[Dict] = logging_config
        self.include_components: Optional[Dict] = include_components
        self.exclude_components: Optional[Dict] = exclude_components
        self._temporary_directory = temporary_directory
        self._output_directory = output_directory
        if backend is not None:
            self._backend = backend
        else:
            self._backend = create(
                prefix='autoPyTorch',
                temporary_directory=self._temporary_directory,
                output_directory=self._output_directory,
                delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
                delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            )
        self.task_type = task_type or ""
        self._stopwatch = StopWatch()

        self.pipeline_options = replace_string_bool_to_bool(json.load(open(
            os.path.join(os.path.dirname(__file__), '../configs/default_pipeline_options.json'))))

        self.search_space: Optional[ConfigurationSpace] = None
        self._dataset_requirements: Optional[List[FitRequirement]] = None
        self._metric: Optional[autoPyTorchMetric] = None
        self._scoring_functions: Optional[List[autoPyTorchMetric]] = None
        self._logger: Optional[PicklableClientLogger] = None
        self.dataset_name: Optional[str] = None
        self.cv_models_: Dict = {}
        self.precision: Optional[int] = None
        self.opt_metric: Optional[str] = None
        self.dataset: Optional[BaseDataset] = None
        self.ensemble_ = None
        self._results_manager = ResultsManager()
        self.feat_types = None

        # By default try to use the TCP logging port or get a new port
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        # Store the resampling strategy from the dataset, to load models as needed
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.stop_logging_server: Optional[multiprocessing.synchronize.Event] = None

        # Single core, local runs should use fork
        # to prevent the __main__ requirements in
        # examples. Nevertheless, multi-process runs
        # have spawn as requirement to reduce the
        # possibility of a deadlock
        self._dask_client: Optional[dask.distributed.Client] = None
        self._multiprocessing_context = 'forkserver'
        if self.n_jobs == 1:
            self._multiprocessing_context = 'fork'

        self.input_validator: Optional[BaseInputValidator] = None

        self.search_space_updates = search_space_updates
        if search_space_updates is not None:
            if not isinstance(self.search_space_updates,
                              HyperparameterSearchSpaceUpdates):
                raise ValueError("Expected search space updates to be of instance"
                                 " HyperparameterSearchSpaceUpdates got {}".format(type(self.search_space_updates)))

    @abstractmethod
    def build_pipeline(
        self,
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        include_components: Optional[Dict[str, Any]] = None,
        exclude_components: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ) -> BasePipeline:
        """
        Build pipeline according to current task
        and for the passed dataset properties

        Args:
            dataset_properties (Dict[str, Any]):
                Characteristics of the dataset to guide the pipeline
                choices of components
            include_components (Optional[Dict[str, Any]]):
                Dictionary containing components to include. Key is the node
                name and Value is an Iterable of the names of the components
                to include. Only these components will be present in the
                search space.
            exclude_components (Optional[Dict[str, Any]]):
                Dictionary containing components to exclude. Key is the node
                name and Value is an Iterable of the names of the components
                to exclude. All except these components will be present in
                the search space.
            search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
                Search space updates that can be used to modify the search
                space of particular components or choice modules of the pipeline

        Returns:
            BasePipeline

        """
        raise NotImplementedError("Function called on BaseTask, this can only be called by "
                                  "specific task which is a child of the BaseTask")

    @abstractmethod
    def _get_dataset_input_validator(
        self,
        X_train: Union[List, pd.DataFrame, np.ndarray],
        y_train: Union[List, pd.DataFrame, np.ndarray],
        X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        resampling_strategy: Optional[ResamplingStrategies] = None,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        dataset_compression: Optional[DatasetCompressionSpec] = None,
        **kwargs: Any
    ) -> Tuple[BaseDataset, BaseInputValidator]:
        """
        Returns an object of a child class of `BaseDataset` and
        an object of a child class of `BaseInputValidator` according
        to the current task.

        Args:
            X_train (Union[List, pd.DataFrame, np.ndarray]):
                Training feature set.
            y_train (Union[List, pd.DataFrame, np.ndarray]):
                Training target set.
            X_test (Optional[Union[List, pd.DataFrame, np.ndarray]]):
                Testing feature set
            y_test (Optional[Union[List, pd.DataFrame, np.ndarray]]):
                Testing target set
            resampling_strategy (Optional[RESAMPLING_STRATEGIES]):
                Strategy to split the training data. if None, uses
                HoldoutValTypes.holdout_validation.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                arguments required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            dataset_name (Optional[str]):
                name of the dataset, used as experiment name.
            dataset_compression (Optional[DatasetCompressionSpec]):
                specifications for dataset compression. For more info check
                documentation for `BaseTask.get_dataset`.

        Returns:
            BaseDataset:
                the dataset object
            BaseInputValidator:
                fitted input validator
        """
        raise NotImplementedError

    def get_dataset(
        self,
        X_train: Union[List, pd.DataFrame, np.ndarray],
        y_train: Union[List, pd.DataFrame, np.ndarray],
        X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        resampling_strategy: Optional[ResamplingStrategies] = None,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        dataset_compression: Optional[DatasetCompressionSpec] = None,
        **kwargs: Any
    ) -> BaseDataset:
        """
        Returns an object of a child class of `BaseDataset` according to the current task.

        Args:
            X_train (Union[List, pd.DataFrame, np.ndarray]):
                Training feature set.
            y_train (Union[List, pd.DataFrame, np.ndarray]):
                Training target set.
            X_test (Optional[Union[List, pd.DataFrame, np.ndarray]]):
                Testing feature set
            y_test (Optional[Union[List, pd.DataFrame, np.ndarray]]):
                Testing target set
            resampling_strategy (Optional[RESAMPLING_STRATEGIES]):
                Strategy to split the training data. if None, uses
                HoldoutValTypes.holdout_validation.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                arguments required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            dataset_name (Optional[str]):
                name of the dataset, used as experiment name.
            dataset_compression (Optional[DatasetCompressionSpec]):
                We compress datasets so that they fit into some predefined amount of memory.
                **NOTE**

                You can also pass your own configuration with the same keys and choosing
                from the available ``"methods"``.
                The available options are described here:
                **memory_allocation**
                    Absolute memory in MB, e.g. 10MB is ``"memory_allocation": 10``.
                    The memory used by the dataset is checked after each reduction method is
                    performed. If the dataset fits into the allocated memory, any further methods
                    listed in ``"methods"`` will not be performed.
                    It can be either float or int.

                **methods**
                    We currently provide the following methods for reducing the dataset size.
                    These can be provided in a list and are performed in the order as given.
                    *   ``"precision"`` -
                        We reduce floating point precision as follows:
                            *   ``np.float128 -> np.float64``
                            *   ``np.float96 -> np.float64``
                            *   ``np.float64 -> np.float32``
                            *   pandas dataframes are reduced using the downcast option of `pd.to_numeric`
                                to the lowest possible precision.
                    *   ``subsample`` -
                        We subsample data such that it **fits directly into
                        the memory allocation** ``memory_allocation * memory_limit``.
                        Therefore, this should likely be the last method listed in
                        ``"methods"``.
                        Subsampling takes into account classification labels and stratifies
                        accordingly. We guarantee that at least one occurrence of each
                        label is included in the sampled set.

        Returns:
            BaseDataset:
                the dataset object
        """
        dataset, _ = self._get_dataset_input_validator(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
            dataset_name=dataset_name,
            dataset_compression=dataset_compression,
            **kwargs)

        return dataset

    @property
    def run_history(self) -> RunHistory:
        return self._results_manager.run_history

    @property
    def ensemble_performance_history(self) -> List[Dict[str, Any]]:
        return self._results_manager.ensemble_performance_history

    @property
    def trajectory(self) -> Optional[List]:
        return self._results_manager.trajectory

    def set_pipeline_config(self, **pipeline_config_kwargs: Any) -> None:
        """
        Check whether arguments are valid and
        then sets them to the current pipeline
        configuration.

        Args:
            **pipeline_config_kwargs: Valid config options include "num_run",
            "device", "budget_type", "epochs", "runtime", "torch_num_threads",
            "early_stopping", "use_tensorboard_logger",
            "metrics_during_training"

        Returns:
            None
        """
        unknown_keys = []
        for option in pipeline_config_kwargs.keys():
            if option in self.pipeline_options.keys():
                pass
            else:
                unknown_keys.append(option)

        if len(unknown_keys) > 0:
            raise ValueError("Invalid configuration arguments given {},"
                             " expected arguments to be in {}".
                             format(unknown_keys, self.pipeline_options.keys()))

        self.pipeline_options.update(pipeline_config_kwargs)

    def get_pipeline_options(self) -> dict:
        """
        Returns the current pipeline configuration.
        """
        return self.pipeline_options

    def get_search_space(self, dataset: BaseDataset = None) -> ConfigurationSpace:
        """
        Returns the current search space as ConfigurationSpace object.
        """
        if self.search_space is not None:
            return self.search_space
        elif dataset is not None:
            return self._get_search_space(
                dataset,
                include_components=self.include_components,
                exclude_components=self.exclude_components,
                search_space_updates=self.search_space_updates)
        raise ValueError("No search space initialised and no dataset passed. "
                         "Can't create default search space without the dataset")

    @staticmethod
    def _get_search_space(
        dataset: BaseDataset,
        include_components,
        exclude_components,
        search_space_updates,
    ) -> ConfigurationSpace:
        dataset_requirements = get_dataset_requirements(
                info=dataset.get_required_dataset_info(),
                include=include_components,
                exclude=exclude_components,
                search_space_updates=search_space_updates)
        return get_configuration_space(info=dataset.get_dataset_properties(dataset_requirements),
                                           include=include_components,
                                           exclude=exclude_components,
                                           search_space_updates=search_space_updates)

    def _get_logger(self, name: str) -> PicklableClientLogger:
        """
        Instantiates the logger used throughout the experiment

        Args:
            name (str):
                Name of the log file, usually the dataset name

        Returns:
            PicklableClientLogger
        """
        logger_name = 'AutoPyTorch:%s:%d' % (name, self.seed)

        # Setup the configuration for the logger
        # This is gonna be honored by the server
        # Which is created below
        setup_logger(
            filename='%s.log' % str(logger_name),
            logging_config=self.logging_config,
            output_dir=self._backend.temporary_directory,
        )

        # As AutoPyTorch works with distributed process,
        # we implement a logger server that can receive tcp
        # pickled messages. They are unpickled and processed locally
        # under the above logging configuration setting
        # We need to specify the logger_name so that received records
        # are treated under the logger_name ROOT logger setting
        context = multiprocessing.get_context(self._multiprocessing_context)
        preload_modules(context)
        self.stop_logging_server = context.Event()
        port = context.Value('l')  # be safe by using a long
        port.value = -1

        # "BaseContext" has no attribute "Process" motivates to ignore the attr check
        self.logging_server = context.Process(  # type: ignore [attr-defined]
            target=start_log_server,
            kwargs=dict(
                host='localhost',
                logname=logger_name,
                event=self.stop_logging_server,
                port=port,
                filename='%s.log' % str(logger_name),
                logging_config=self.logging_config,
                output_dir=self._backend.temporary_directory,
            ),
        )

        self.logging_server.start()

        while True:
            with port.get_lock():
                if port.value == -1:
                    time.sleep(0.01)
                else:
                    break

        self._logger_port = int(port.value)

        return get_named_client_logger(
            name=logger_name,
            host='localhost',
            port=self._logger_port,
        )

    def _clean_logger(self) -> None:
        """
        cleans the logging server created

        Returns:
            None
        """
        if not hasattr(self, 'stop_logging_server') or self.stop_logging_server is None:
            return

        # Clean up the logger
        if self.logging_server.is_alive():
            self.stop_logging_server.set()

            # We try to join the process, after we sent
            # the terminate event. Then we try a join to
            # nicely join the event. In case something
            # bad happens with nicely trying to kill the
            # process, we execute a terminate to kill the
            # process.
            self.logging_server.join(timeout=5)
            self.logging_server.terminate()
            del self.stop_logging_server
            self._logger = None

    def _create_dask_client(self) -> None:
        """
        creates the dask client that is used to parallelize
        the training of pipelines

        Returns:
            None
        """
        self._is_dask_client_internally_created = True
        dask.config.set({'distributed.worker.daemon': False})
        self._dask_client = dask.distributed.Client(
            dask.distributed.LocalCluster(
                n_workers=self.n_jobs,
                processes=True,
                threads_per_worker=self.n_threads,
                # We use the temporal directory to save the
                # dask workers, because deleting workers
                # more time than deleting backend directories
                # This prevent an error saying that the worker
                # file was deleted, so the client could not close
                # the worker properly
                local_directory=tempfile.gettempdir(),
                # Memory is handled by the pynisher, not by the dask worker/nanny
                memory_limit=0,
            ),
            # Heartbeat every 10s
            heartbeat_interval=10000,
        )

    def _close_dask_client(self) -> None:
        """
        Closes the created dask client

        Returns:
            None
        """
        if (
            hasattr(self, '_is_dask_client_internally_created')
            and self._is_dask_client_internally_created
            and self._dask_client
        ):
            self._dask_client.shutdown()
            self._dask_client.close()
            del self._dask_client
            self._dask_client = None
            self._is_dask_client_internally_created = False
            del self._is_dask_client_internally_created

    def _load_models(self) -> bool:

        """
        Loads the models saved in the temporary directory
        during the smac run and the final ensemble created

        Returns:
            None
        """
        if self.resampling_strategy is None:
            raise ValueError("Resampling strategy is needed to determine what models to load")
        self.ensemble_ = self._backend.load_ensemble(self.seed)

        # If no ensemble is loaded, try to get the best performing model
        if not self.ensemble_:
            self.ensemble_ = self._load_best_individual_model()

        if self.ensemble_:
            identifiers = self.ensemble_.get_selected_model_identifiers()
            # nonnull_identifiers = [i for i in identifiers if i is not None]
            # self.models_ = self._backend.load_models_by_identifiers(nonnull_identifiers)
            # if isinstance(self.resampling_strategy, CrossValTypes):
            #     self.cv_models_ = self._backend.load_cv_models_by_identifiers(nonnull_identifiers)

            self._logger.debug(f"stacked ensemble identifiers are :{identifiers}")
            if is_stacking(self.base_ensemble_method, self.stacking_ensemble_method):
                models = []
                cv_models = []
                for identifier in identifiers:
                    nonnull_identifiers = [i for i in identifier if i is not None]
                    models.append(self._backend.load_models_by_identifiers(nonnull_identifiers))
                    cv_models.append(self._backend.load_cv_models_by_identifiers(nonnull_identifiers))
                # self._logger.debug(f"stacked ensemble models are :{models}")
                self.models_ = models
                self.cv_models_ = cv_models

            else:
                self.models_ = self._backend.load_models_by_identifiers(identifiers)
                if isinstance(self.resampling_strategy, (CrossValTypes, RepeatedCrossValTypes)):
                    self.cv_models_ = self._backend.load_cv_models_by_identifiers(identifiers)
            if isinstance(self.resampling_strategy, (CrossValTypes, RepeatedCrossValTypes)):
                if len(self.cv_models_) == 0:
                    raise ValueError('No models fitted!')

        elif 'pipeline' not in self._disable_file_output:
            model_names = self._backend.list_all_models(self.seed)

            if len(model_names) == 0:
                raise ValueError('No models fitted!')

            self.models_ = {}

        else:
            self.models_ = {}

        return True

    def _cleanup(self) -> None:
        """
        Closes the different servers created during api search.
        Returns:
                None
        """
        if hasattr(self, '_logger') and self._logger is not None:
            self._logger.info("Closing the dask infrastructure")
            self._close_dask_client()
            self._logger.info("Finished closing the dask infrastructure")

            # Clean up the logger
            self._logger.info("Starting to clean up the logger")
            self._clean_logger()
        else:
            self._close_dask_client()

    def _load_best_individual_model(self) -> SingleBest:
        """
        In case of failure during ensemble building,
        this method returns the single best model found
        by AutoML.
        This is a robust mechanism to be able to predict,
        even though no ensemble was found by ensemble builder.

        Returns:
            SingleBest:
                Ensemble made with incumbent pipeline
        """

        if self._metric is None:
            raise ValueError("Providing a metric to AutoPytorch is required to fit a model. "
                             "A default metric could not be inferred. Please check the log "
                             "for error messages."
                             )

        # SingleBest contains the best model found by AutoML
        ensemble = SingleBest(
            metric=self._metric,
            seed=self.seed,
            run_history=self.run_history,
            backend=self._backend,
        )
        if self._logger is None:
            warnings.warn(
                "No valid ensemble was created. Please check the log"
                "file for errors. Default to the best individual estimator:{}".format(
                    ensemble.identifiers_
                )
            )
        else:
            self._logger.exception(
                "No valid ensemble was created. Please check the log"
                "file for errors. Default to the best individual estimator:{}".format(
                    ensemble.identifiers_
                )
            )

        return ensemble

    def _do_dummy_prediction(self) -> None:

        assert self._metric is not None
        assert self._logger is not None

        # For dummy estimator, we always expect the num_run to be 1
        num_run = 1

        self._logger.info("Starting to create dummy predictions.")

        memory_limit = self._memory_limit
        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))

        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = self._time_for_task
        # This stats object is a hack - maybe the SMAC stats object should
        # already be generated here!
        stats = Stats(scenario_mock)
        stats.start_timing()
        ta = ExecuteTaFuncWithQueue(
            pynisher_context=self._multiprocessing_context,
            backend=self._backend,
            seed=self.seed,
            metric=self._metric,
            multi_objectives=["cost"],
            logger_port=self._logger_port,
            cost_for_crash=get_cost_of_crash(self._metric),
            abort_on_first_run_crash=False,
            initial_num_run=num_run,
            stats=stats,
            memory_limit=memory_limit,
            disable_file_output=self._disable_file_output,
            all_supported_metrics=self._all_supported_metrics,
            base_ensemble_method=self.base_ensemble_method,
            pipeline_config=self.pipeline_options
        )

        status, _, _, additional_info = ta.run(num_run, cutoff=self._time_for_task)
        if status == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy predictions.")
        else:
            if additional_info.get('exitcode') == -6:
                err_msg = "Dummy prediction failed with run state {},\n" \
                          "because the provided memory limits were too tight.\n" \
                          "Please increase the 'ml_memory_limit' and try again.\n" \
                          "If you still get the problem, please open an issue and\n" \
                          "paste the additional info.\n" \
                          "Additional info:\n{}.".format(str(status), dict_repr(additional_info))
                self._logger.error(err_msg)
                # Fail if dummy prediction fails.
                raise ValueError(err_msg)

            else:
                err_msg = "Dummy prediction failed with run state {} and additional info:\n{}.".format(
                    str(status), dict_repr(additional_info)
                )
                self._logger.error(err_msg)
                # Fail if dummy prediction fails.
                raise ValueError(err_msg)

    def _do_traditional_prediction(self, time_left: int, func_eval_time_limit_secs: int) -> None:
        """
        Fits traditional machine learning algorithms to the provided dataset, while
        complying with time resource allocation.

        This method currently only supports classification.

        Args:
            time_left: (int)
                Hard limit on how many machine learning algorithms can be fit. Depending on how
                fast a traditional machine learning algorithm trains, it will allow multiple
                models to be fitted.
            func_eval_time_limit_secs: (int)
                Maximum training time each algorithm is allowed to take, during training
        """

        # Mypy Checkings -- Traditional prediction is only called for search
        # where the following objects are created
        assert self._metric is not None
        assert self._logger is not None
        assert self._dask_client is not None

        self._logger.info("Starting to create traditional classifier predictions.")

        available_classifiers = get_available_traditional_learners(dataset_properties=self._get_dataset_properties(self.dataset))
        model_configs = [(key, self.pipeline_options[self.pipeline_options['budget_type']]) for key in available_classifiers.keys()]
        
        run_history, _ = run_models_on_dataset(
            time_left=time_left,
            func_eval_time_limit_secs=func_eval_time_limit_secs,
            model_configs=model_configs,
            logger=self._logger,
            logger_port=self._logger_port,
            metric=self._metric,
            dask_client=self._dask_client,
            backend=self._backend,
            memory_limit=self._memory_limit,
            disable_file_output=self._disable_file_output,
            all_supported_metrics=self._all_supported_metrics,
            base_ensemble_method=self.base_ensemble_method,
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates,
            pipeline_options=self.pipeline_options,
            seed=self.seed,
            multiprocessing_context=self._multiprocessing_context,
            n_jobs=self.n_jobs,
            current_search_space=self.search_space,
            smac_initial_run=self._backend.get_next_num_run()
        )

        self._logger.debug("Run history traditional: {}".format(run_history))
        # add run history of traditional to api run history
        self.run_history.update(run_history, DataOrigin.EXTERNAL_SAME_INSTANCES)
        run_history.save_json(os.path.join(self._backend.internals_directory, 'traditional_run_history.json'),
                              save_external=True)
        return

    def run_traditional_ml(
        self,
        current_task_name: str,
        runtime_limit: int,
        func_eval_time_limit_secs: int
    ) -> None:
        """
        This function can be used to run the suite of traditional machine
        learning models during the current task (for e.g, ensemble fit, search)

        Args:
            current_task_name (str): name of the current task,
            runtime_limit (int): time limit for fitting traditional models,
            func_eval_time_limit_secs (int): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit.
        """
        assert self._logger is not None  # for mypy compliancy
        traditional_task_name = 'runTraditional'
        self._stopwatch.start_task(traditional_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(current_task_name)
        time_for_traditional = int(runtime_limit - elapsed_time)
        self._do_traditional_prediction(
            func_eval_time_limit_secs=func_eval_time_limit_secs,
            time_left=time_for_traditional,
        )
        self._stopwatch.stop_task(traditional_task_name)

    def _fit_models_on_dataset(
        self,
        model_configs: List[Tuple[Union[Configuration, str], Union[int, float]]],
        func_eval_time_limit_secs,
        stacking_layer,
        time_left,
        current_search_space,
        smac_initial_run,
        base_ensemble_method,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ) -> List[Tuple]:\
        
        search_space_updates = search_space_updates if search_space_updates is not None else self.search_space_updates

        self._logger.debug(f"base_ensemble_method: {base_ensemble_method}, model_configs {model_configs} ")
        run_history, model_identifiers = run_models_on_dataset(
            time_left=time_left,
            func_eval_time_limit_secs=func_eval_time_limit_secs,
            model_configs=model_configs,
            logger=self._logger,
            logger_port=self._logger_port,
            metric=self._metric,
            dask_client=self._dask_client,
            backend=self._backend,
            memory_limit=self._memory_limit,
            disable_file_output=self._disable_file_output,
            all_supported_metrics=self._all_supported_metrics,
            base_ensemble_method=base_ensemble_method,
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=search_space_updates,
            pipeline_options=self.pipeline_options,
            seed=self.seed,
            multiprocessing_context=self._multiprocessing_context,
            n_jobs=self.n_jobs,
            current_search_space=current_search_space,
            smac_initial_run=smac_initial_run
        )

        self._logger.debug("Run history for layer: {}: {}".format(stacking_layer, run_history))
        # add run history of traditional to api run history
        self.run_history.update(run_history, DataOrigin.EXTERNAL_SAME_INSTANCES)
        run_history.save_json(os.path.join(self._backend.internals_directory, f'run_history_{stacking_layer}.json'),
                              save_external=True)
        return model_identifiers

    def _reset_datamanager_in_backend(self, datamanager)-> None:
        self._backend.save_datamanager(datamanager)

    def _posthoc_fit_ensemble(
        self,
        optimize_metric,
        time_left_for_ensemble,
        last_successful_smac_initial_num_run,
        ensemble_size,
        iteration,
        enable_traditional_pipeline=False,
        cleanup=True,
        func_eval_time_limit_secs: int = 50,
        base_ensemble_method = BaseLayerEnsembleSelectionTypes.ensemble_selection,
        stacking_ensemble_method = None
    ):
        self.fit_ensemble(
            optimize_metric=optimize_metric,
            precision=self.precision,
            ensemble_size=ensemble_size,
            ensemble_nbest=self.ensemble_nbest,
            initial_num_run=last_successful_smac_initial_num_run,
            time_for_task=time_left_for_ensemble,
            enable_traditional_pipeline=enable_traditional_pipeline,
            func_eval_time_limit_secs=func_eval_time_limit_secs,
            iteration=iteration,
            cleanup=cleanup,
            base_ensemble_method=base_ensemble_method,
            stacking_ensemble_method=stacking_ensemble_method,
            load_models=False
        )
        final_ensemble: EnsembleSelection = self._backend.load_ensemble(self.seed)
        final_model_identifiers = final_ensemble.get_selected_model_identifiers()
        final_model_identifiers = [identifier for identifier in final_model_identifiers[-1] if identifier is not None] if isinstance(final_ensemble, EnsembleOptimisationStackingEnsemble) else final_model_identifiers
        final_model_identifiers_dict = {identifier: identifier for identifier in final_model_identifiers}
        models_with_weights = final_ensemble.get_models_with_weights(final_model_identifiers_dict)
        final_model_identifiers = [identifier[1] for identifier in models_with_weights]
        final_weights = [identifier[0] for identifier in models_with_weights]
        return final_model_identifiers,final_weights

    def _run_autogluon_stacking(
        self,
        optimize_metric: str,
        dataset: BaseDataset,
        max_budget: int = 50,
        budget_type: str = 'epochs',
        total_walltime_limit: int = 100,
        func_eval_time_limit_secs: Optional[int] = None,
        memory_limit: Optional[int] = 4096,
        all_supported_metrics: bool = True,
        precision: int = 32,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
        dask_client: Optional[dask.distributed.Client] = None,
    ):
        """
        This function can be used to create a stacking ensemble
        Args:
            current_task_name (str): name of the current task,
            runtime_limit (int): time limit for fitting traditional models,
            func_eval_time_limit_secs (int): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit.
        """
        experiment_task_name: str = 'runStacking'
        self._init_required_args(
            experiment_task_name=experiment_task_name,
            optimize_metric=optimize_metric,
            dataset=dataset,
            budget_type=budget_type,
            max_budget=max_budget,
            total_walltime_limit=total_walltime_limit,
            memory_limit=memory_limit,
            all_supported_metrics=all_supported_metrics,
            precision=precision,
            disable_file_output=disable_file_output,
            dask_client=dask_client
        )
        self.pipeline_options['func_eval_time_limit_secs'] = func_eval_time_limit_secs
        self.precision = precision
        available_classifiers = get_available_traditional_learners(dataset_properties=self._get_dataset_properties(self.dataset))
        model_configs = [(key, self.pipeline_options[self.pipeline_options['budget_type']]) for key in available_classifiers.keys()]

        if self.feat_types is None:
            raise ValueError("Cant run autogluon stacking without information about dataset features passed with `feat_type`")
        autogluon_nn_search_space_updates = get_autogluon_default_nn_config(feat_types=self.feat_types)
        autogluon_nn_search_space = self._get_search_space(
                self.dataset,
                include_components=self.include_components,
                exclude_components=self.exclude_components,
                search_space_updates=autogluon_nn_search_space_updates)

        default_nn_config = autogluon_nn_search_space.get_default_configuration()
        model_configs.append((default_nn_config, self.pipeline_options[self.pipeline_options['budget_type']]))
        self._logger.info("Starting Autogluon Stacking.")

        model_identifiers = []
        stacked_weights = []
        last_successful_smac_initial_num_run = None
        for stacking_layer in range(self.num_stacking_layers):
            smac_initial_run=self._backend.get_next_num_run()
            updated_model_configs, current_search_space = self._update_configs_for_current_config_space(
                model_configs,
                dataset,
                autogluon_nn_search_space_updates,
                assert_skew_transformer_quantile=True)
            layer_model_identifiers = self._fit_models_on_dataset(
                updated_model_configs,
                func_eval_time_limit_secs,
                stacking_layer,
                time_left=(0.9*total_walltime_limit)/(self.num_stacking_layers),
                current_search_space=current_search_space,
                smac_initial_run=smac_initial_run,
                search_space_updates=autogluon_nn_search_space_updates,
                base_ensemble_method=self.base_ensemble_method)
            nonnull_identifiers = [identifier for identifier in layer_model_identifiers if identifier is not None]
            if len(nonnull_identifiers) > 0:
                model_identifiers.append(
                    nonnull_identifiers
                )
                last_successful_smac_initial_num_run = smac_initial_run
                ensemble_size = len(nonnull_identifiers)
                weights = [1/ensemble_size] * ensemble_size
                stacked_weights.append(weights)
            _, previous_layer_predictions_train, previous_layer_predictions_test = self._get_previous_predictions(smac_initial_run, model_identifiers[-1], weights, ensemble_size)
            dataset = get_appended_dataset(
                original_dataset=self.dataset,
                previous_layer_predictions_train=previous_layer_predictions_train,
                previous_layer_predictions_test=previous_layer_predictions_test,
                resampling_strategy=self.resampling_strategy,
                resampling_strategy_args=self.resampling_strategy_args,
            )
            self._reset_datamanager_in_backend(datamanager=dataset)

        ensemble = AutogluonStackingEnsemble()
        iteration = 0
        time_left_for_ensemble = total_walltime_limit-self._stopwatch.wall_elapsed(experiment_task_name)
        final_model_identifiers, final_weights = self._posthoc_fit_ensemble(
            optimize_metric,
            time_left_for_ensemble,
            last_successful_smac_initial_num_run,
            ensemble_size,
            iteration)
        model_identifiers[-1] = final_model_identifiers
        stacked_weights[-1] = final_weights
        ensemble = ensemble.fit(model_identifiers, stacked_weights)
        self._backend.save_ensemble(ensemble, iteration+1, self.seed)
        self._load_models()

    def _run_search_stacking(
        self,
        optimize_metric: str,
        min_budget,
        max_budget,
        precision,
        portfolio_selection,
        experiment_task_name,
        tae_func = None,
        posthoc_ensemble_fit: bool = False,
        enable_traditional_pipeline: bool = True,
        total_walltime_limit: int = 400,
        func_eval_time_limit_secs: Optional[int] = None,
        smac_scenario_args: Optional[Dict[str, Any]] = None,
        get_smac_object_callback: Optional[Callable] = None,
        smbo_class: Optional[SMBO] = None,
    ):
        stacking_task_name = "runStacking"
        self._logger.debug(f"Starting to run stacking")
        self._stopwatch.start_task(stacking_task_name)
        self.precision = precision
        self.opt_metric = optimize_metric
        time_left_for_search_base_models = math.floor(0.5*total_walltime_limit)
        if posthoc_ensemble_fit:
            time_left_for_search_base_models = math.floor(TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT*time_left_for_search_base_models)
            time_left_for_posthoc_ensemble = math.floor((1-TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT)*time_left_for_search_base_models)
        proc_ensemble = None
        if time_left_for_search_base_models <= 0:
            # Fit only raises error when ensemble_size is not zero but
            # time_left_for_search_base_models is zero.
            if self.ensemble_size > 0:
                raise ValueError("Not starting ensemble builder because there "
                                "is no time left. Try increasing the value "
                                "of time_left_for_this_task.")
        elif self.ensemble_size <= 0:
            self._logger.info("Not starting ensemble builder as ensemble size is 0")
        else:
            proc_ensemble = self._init_ensemble_builder(time_left_for_ensembles=time_left_for_search_base_models,
                                                        ensemble_size=self.ensemble_size,
                                                        ensemble_nbest=self.ensemble_nbest,
                                                        precision=precision,
                                                        optimize_metric=self.opt_metric,
                                                        base_ensemble_method=self.base_ensemble_method, # TODO: enter base ensemble method as currently it defaults to ensemble selection for the base layer.
                                                        stacking_ensemble_method=self.stacking_ensemble_method,
                                                        num_stacking_layers=1,
                                                        )

        smac_initial_run = self._run_smbo(
            min_budget=min_budget,
            max_budget=max_budget,
            total_walltime_limit=time_left_for_search_base_models,
            func_eval_time_limit_secs=func_eval_time_limit_secs,
            smac_scenario_args=smac_scenario_args,
            get_smac_object_callback=get_smac_object_callback,
            tae_func=tae_func,
            portfolio_selection=portfolio_selection,
            experiment_task_name=experiment_task_name,
            proc_ensemble=proc_ensemble,
            num_stacking_layers=1,
            smbo_class=smbo_class
        )
        if proc_ensemble is not None:
            self._collect_results_ensemble(proc_ensemble)
        
        if posthoc_ensemble_fit:
            self._logger.debug(f"time_left_for_ensemble : {time_left_for_posthoc_ensemble}")
            self._posthoc_fit_ensemble(
                optimize_metric,
                time_left_for_posthoc_ensemble,
                0,
                self.ensemble_size,
                proc_ensemble.iteration + 1,
                enable_traditional_pipeline=enable_traditional_pipeline,
                base_ensemble_method=BaseLayerEnsembleSelectionTypes.ensemble_selection,
                stacking_ensemble_method=self.stacking_ensemble_method,
                cleanup=False)

        base_ensemble = self._backend.load_ensemble(self.seed)
        # TODO: refactor to remove if
        model_identifiers = [base_ensemble.get_selected_model_identifiers()] if isinstance(base_ensemble, EnsembleSelection) else base_ensemble.get_selected_model_identifiers()
        model_identifiers[-1] = [identifier for identifier in model_identifiers[-1] if identifier is not None]
        ensemble = RepeatModelsStackingEnsemble(base_ensemble=base_ensemble)

        weights = [weight for weight in base_ensemble.weights_ if weight > 0]
        ensemble_size = self.ensemble_size
        model_configs, previous_layer_predictions_train, previous_layer_predictions_test = self._get_previous_predictions(smac_initial_run, model_identifiers[-1], weights, ensemble_size)

        self._logger.debug(f"Finished search for base models, starting fitting next layers")
        for stacking_layer in range(1, self.num_stacking_layers):
            smac_layer_initial_run = self._backend.get_next_num_run()
            time_left_for_higher_stacking_layers = total_walltime_limit - self._stopwatch.wall_elapsed(stacking_task_name)
            if time_left_for_higher_stacking_layers < func_eval_time_limit_secs:
                break
            self._logger.debug(f"Original feat types len: {len(self.dataset.feat_types)}")
            nonnull_model_predictions_train = [pred for pred in previous_layer_predictions_train if pred is not None]
            nonnull_model_predictions_test = [pred for pred in previous_layer_predictions_test if pred is not None]
            assert len(nonnull_model_predictions_train) == len(nonnull_model_predictions_test)
            self._logger.debug(f"length Non null predictions: {len(nonnull_model_predictions_train)}")
            dataset = get_appended_dataset(
                original_dataset=self.dataset,
                previous_layer_predictions_train=nonnull_model_predictions_train,
                previous_layer_predictions_test=nonnull_model_predictions_test,
                resampling_strategy=self.resampling_strategy,
                resampling_strategy_args=self.resampling_strategy_args,
            )
            self._logger.debug(f"new feat_types len: {len(dataset.feat_types)}")
            self._logger.debug(f"model_configs going into update configs {model_configs}")
            updated_model_configs, current_search_space = self._update_configs_for_current_config_space(model_configs, dataset)
            self._reset_datamanager_in_backend(datamanager=dataset)
            layer_model_identifiers = self._fit_models_on_dataset(
                updated_model_configs, func_eval_time_limit_secs, stacking_layer, time_left=time_left_for_higher_stacking_layers/(self.num_stacking_layers - 1), current_search_space=current_search_space, smac_initial_run=smac_layer_initial_run,
                base_ensemble_method=BaseLayerEnsembleSelectionTypes.ensemble_selection)
            nonnull_layer_model_identifiers = [identifier for identifier in layer_model_identifiers if identifier is not None]
            self._logger.debug(f"For layer: {stacking_layer}, layer_model_identifiers: {layer_model_identifiers}, nonnull_layer_model_identifiers: {nonnull_layer_model_identifiers}")
            if len(nonnull_layer_model_identifiers) > 0:
                model_identifiers.append(
                    nonnull_layer_model_identifiers
                )
            _, previous_layer_predictions_train, previous_layer_predictions_test = self._get_previous_predictions(smac_initial_run, model_identifiers[-1], weights, ensemble_size)

        self._logger.debug(f"Stacked ensemble identifiers going into Stacked repeats: {model_identifiers}, with weights: {ensemble.base_weights}")
        ensemble = ensemble.fit(model_identifiers)
        self._backend.save_ensemble(ensemble, proc_ensemble.iteration+10, self.seed)
        self._load_models()

    def _get_previous_predictions(self, smac_initial_run, model_identifiers, weights, ensemble_size):
        model_configs = []
        previous_layer_predictions_train = []
        previous_layer_predictions_test = []
        self._logger.debug(f'id_config: {self.run_history.ids_config}')
        for weight, model_identifier in zip(weights, model_identifiers):
            if model_identifier is None:
                model_configs.append(None)
                previous_layer_predictions_train.append(None)
                previous_layer_predictions_test.append(None)
                continue
            seed, num_run, budget = model_identifier
            
            self._logger.debug(f'num_run: {num_run}')
            config = get_config_from_run_history(self.run_history, num_run=num_run) #  self.run_history.ids_config.get(num_run-smac_initial_run, None)
            self._logger.debug(f'Configuration from previous layer: {config}')
            model_configs.append((config, budget))
            previous_layer_predictions_train.extend(
                [np.load(os.path.join(
                    self._backend.get_numrun_directory(seed=seed, num_run=num_run, budget=budget),
                    self._backend.get_prediction_filename('ensemble', seed, num_run, budget)
                    ), allow_pickle=True)] * int(weight * ensemble_size))
            previous_layer_predictions_test.extend([np.load(os.path.join(
                self._backend.get_numrun_directory(seed=seed, num_run=num_run, budget=budget),
                self._backend.get_prediction_filename('test', seed, num_run, budget)
                ), allow_pickle=True)] * int(weight * ensemble_size))
        return model_configs,previous_layer_predictions_train,previous_layer_predictions_test

    def _update_configs_for_current_config_space(
        self,
        model_descriptions: List[Tuple],
        dataset: BaseDataset,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
        assert_skew_transformer_quantile: bool = False
        ) -> List[Tuple]:
        
        search_space_updates = search_space_updates if search_space_updates is not None else self.search_space_updates

        dataset_properties = self._get_dataset_properties(dataset=dataset)
        current_search_space = self._get_search_space(
                                                dataset,
                                                include_components=self.include_components,
                                                exclude_components=self.exclude_components,
                                                search_space_updates=search_space_updates)
        self._logger.debug(f"dataset properties after appending predictions: {dict_repr(dataset_properties)}")
        n_numerical_in_incumbent_on_task_id = len(self.dataset.numerical_columns)
        num_numerical = len(dataset.numerical_columns)
        updated_model_descriptions = []
        for model_description in model_descriptions:
            if model_description is None:
                continue
            config, budget = model_description
            if config is None:
                continue

            if not isinstance(config, (Configuration, dict)):
                updated_model_descriptions.append((config, budget))
                continue

            updated_config = validate_config(
                        config=config,
                        search_space=current_search_space,
                        num_numerical=num_numerical,
                        n_numerical_in_incumbent_on_task_id=n_numerical_in_incumbent_on_task_id,
                        assert_autogluon_numerical_hyperparameters=assert_skew_transformer_quantile
                    )
            updated_model_descriptions.append((updated_config, budget))
        return updated_model_descriptions, current_search_space

    def _run_smbo(
        self,
        min_budget,
        max_budget,
        total_walltime_limit,
        func_eval_time_limit_secs,
        smac_scenario_args,
        portfolio_selection,
        experiment_task_name,
        proc_ensemble,
        num_stacking_layers,
        get_smac_object_callback=None,
        tae_func=None,
        smbo_class=None,
    ) -> int:
        smac_initial_num_run = self._backend.get_next_num_run(peek=True)
        proc_runhistory_updater = None
        if (
            self.base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation
            and smbo_class is not None
        ):
            proc_runhistory_updater = self._init_result_history_updater(initial_num_run=smac_initial_num_run)

        # ==> Run SMAC
        smac_task_name: str = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(experiment_task_name)
        time_left_for_smac = max(0, total_walltime_limit - elapsed_time)

        self._logger.info("Starting SMAC with %5.2f sec time left" % time_left_for_smac)
        if time_left_for_smac <= 0:
            self._logger.warning(" Not starting SMAC because there is no time left")
        else:
            _proc_smac = AutoMLSMBO(
                config_space=self.search_space,
                dataset_name=str(self.dataset_name),
                backend=self._backend,
                total_walltime_limit=total_walltime_limit,
                func_eval_time_limit_secs=func_eval_time_limit_secs,
                dask_client=self._dask_client,
                memory_limit=self._memory_limit,
                n_jobs=self.n_jobs,
                watcher=self._stopwatch,
                metric=self._metric,
                seed=self.seed,
                include=self.include_components,
                exclude=self.exclude_components,
                disable_file_output=self._disable_file_output,
                all_supported_metrics=self._all_supported_metrics,
                smac_scenario_args=smac_scenario_args,
                get_smac_object_callback=get_smac_object_callback,
                pipeline_config=self.pipeline_options,
                min_budget=min_budget,
                max_budget=max_budget,
                ensemble_callback=proc_ensemble,
                base_ensemble_method=self.base_ensemble_method,
                stacking_ensemble_method=self.stacking_ensemble_method,
                logger_port=self._logger_port,
                resampling_strategy=self.resampling_strategy,
                resampling_strategy_args=self.resampling_strategy_args,
                # We do not increase the num_run here, this is something
                # smac does internally
                start_num_run=smac_initial_num_run,
                search_space_updates=self.search_space_updates,
                portfolio_selection=portfolio_selection,
                pynisher_context=self._multiprocessing_context,
                smbo_class=smbo_class,
                use_ensemble_opt_loss=self.use_ensemble_opt_loss,
                other_callbacks=[proc_runhistory_updater] if proc_runhistory_updater is not None else None,
                num_stacking_layers=num_stacking_layers
            )
            try:
                run_history, self._results_manager.trajectory, budget_type = \
                    _proc_smac.run_smbo(func=tae_func)
                self.run_history.update(run_history, DataOrigin.INTERNAL)
                trajectory_filename = os.path.join(
                    self._backend.get_smac_output_directory_for_run(self.seed),
                    'trajectory.json')

                assert self.trajectory is not None  # mypy check
                saveable_trajectory = \
                    [list(entry[:2]) + [entry[2].get_dictionary()] + list(entry[3:])
                     for entry in self.trajectory]
                try:
                    with open(trajectory_filename, 'w') as fh:
                        json.dump(saveable_trajectory, fh)
                except Exception as e:
                    self._logger.warning(f"Cannot save {trajectory_filename} due to {e}...")
            except Exception as e:
                self._logger.exception(str(e))
                raise
        return smac_initial_num_run

    def _search(
        self,
        optimize_metric: str,
        dataset: BaseDataset,
        budget_type: str = 'epochs',
        min_budget: int = 5,
        max_budget: int = 50,
        total_walltime_limit: int = 100,
        func_eval_time_limit_secs: Optional[int] = None,
        enable_traditional_pipeline: bool = True,
        memory_limit: Optional[int] = 4096,
        smac_scenario_args: Optional[Dict[str, Any]] = None,
        get_smac_object_callback: Optional[Callable] = None,
        tae_func: Optional[Callable] = None,
        all_supported_metrics: bool = True,
        precision: int = 32,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
        load_models: bool = True,
        portfolio_selection: Optional[str] = None,
        dask_client: Optional[dask.distributed.Client] = None,
        smbo_class: Optional[SMBO] = None,
        use_ensemble_opt_loss: bool = False,
        posthoc_ensemble_fit: bool = False
    ) -> 'BaseTask':
        """
        Search for the best pipeline configuration for the given dataset.

        Fit both optimizes the machine learning models and builds an ensemble out of them.
        To disable ensembling, set ensemble_size==0.
        using the optimizer.

        Args:
            dataset (Dataset):
                The argument that will provide the dataset splits. It is
                a subclass of the  base dataset object which can
                generate the splits based on different restrictions.
                Providing X_train, y_train and dataset together is not supported.
            optimize_metric (str): name of the metric that is used to
                evaluate a pipeline.
            budget_type (str):
                Type of budget to be used when fitting the pipeline.
                It can be one of:

                + `epochs`: The training of each pipeline will be terminated after
                    a number of epochs have passed. This number of epochs is determined by the
                    budget argument of this method.
                + `runtime`: The training of each pipeline will be terminated after
                    a number of seconds have passed. This number of seconds is determined by the
                    budget argument of this method. The overall fitting time of a pipeline is
                    controlled by func_eval_time_limit_secs. 'runtime' only controls the allocated
                    time to train a pipeline, but it does not consider the overall time it takes
                    to create a pipeline (data loading and preprocessing, other i/o operations, etc.).
                    budget_type will determine the units of min_budget/max_budget. If budget_type=='epochs'
                    is used, min_budget will refer to epochs whereas if budget_type=='runtime' then
                    min_budget will refer to seconds.
            min_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>`_ to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.
            total_walltime_limit (int: default=100):
                Time limit in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit_secs (Optional[int]):
                Time limit for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
                When set to None, this time will automatically be set to
                total_walltime_limit // 2 to allow enough time to fit
                at least 2 individual machine learning algorithms.
                Set to np.inf in case no time limit is desired.
            enable_traditional_pipeline (bool: default=True):
                We fit traditional machine learning algorithms
                (LightGBM, CatBoost, RandomForest, ExtraTrees, KNN, SVM)
                prior building PyTorch Neural Networks. You can disable this
                feature by turning this flag to False. All machine learning
                algorithms that are fitted during search() are considered for
                ensemble building.
            memory_limit (Optional[int]: default=4096):
                Memory limit in MB for the machine learning algorithm.
                Autopytorch will stop fitting the machine learning algorithm
                if it tries to allocate more than memory_limit MB. If None
                is provided, no memory limit is set. In case of multi-processing,
                memory_limit will be per job. This memory limit also applies to
                the ensemble creation process.
            smac_scenario_args (Optional[Dict]):
                Additional arguments inserted into the scenario of SMAC. See the
                `SMAC documentation <https://automl.github.io/SMAC3/master/options.html?highlight=scenario#scenario>`_
                for a list of available arguments.
            get_smac_object_callback (Optional[Callable]):
                Callback function to create an object of class
                `smac.optimizer.smbo.SMBO <https://automl.github.io/SMAC3/master/apidoc/smac.optimizer.smbo.html>`_.
                The function must accept the arguments scenario_dict,
                instances, num_params, runhistory, seed and ta. This is
                an advanced feature. Use only if you are familiar with
                `SMAC <https://automl.github.io/SMAC3/master/index.html>`_.
            tae_func (Optional[Callable]):
                TargetAlgorithm to be optimised. If None, `eval_function`
                available in autoPyTorch/evaluation/train_evaluator is used.
                Must be child class of AbstractEvaluator.
            all_supported_metrics (bool: default=True):
                If True, all metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            precision (int: default=32):
                Numeric precision used when loading ensemble data.
                Can be either '16', '32' or '64'.
            disable_file_output (Optional[List[Union[str, DisableFileOutputParameters]]]):
                Used as a list to pass more fine-grained
                information on what to save. Must be a member of `DisableFileOutputParameters`.
                Allowed elements in the list are:

                + `y_optimization`:
                    do not save the predictions for the optimization set,
                    which would later on be used to build an ensemble. Note that SMAC
                    optimizes a metric evaluated on the optimization set.
                + `pipeline`:
                    do not save any individual pipeline files
                + `pipelines`:
                    In case of cross validation, disables saving the joint model of the
                    pipelines fit on each fold.
                + `y_test`:
                    do not save the predictions for the test set.
                + `all`:
                    do not save any of the above.
                For more information check `autoPyTorch.evaluation.utils.DisableFileOutputParameters`.
            load_models (bool: default=True):
                Whether to load the models after fitting AutoPyTorch.
            portfolio_selection (Optional[str]):
                This argument controls the initial configurations that
                AutoPyTorch uses to warm start SMAC for hyperparameter
                optimization. By default, no warm-starting happens.
                The user can provide a path to a json file containing
                configurations, similar to (...herepathtogreedy...).
                Additionally, the keyword 'greedy' is supported,
                which would use the default portfolio from
                `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>`_

        Returns:
            self

        """
        experiment_task_name: str = 'runSearch'

        self._init_required_args(
            experiment_task_name=experiment_task_name,
            optimize_metric=optimize_metric,
            dataset=dataset,
            budget_type=budget_type,
            max_budget=max_budget,
            total_walltime_limit=total_walltime_limit,
            memory_limit=memory_limit,
            all_supported_metrics=all_supported_metrics,
            precision=precision,
            disable_file_output=disable_file_output,
            dask_client=dask_client
        )

        # Handle time resource allocation
        elapsed_time = self._stopwatch.wall_elapsed(experiment_task_name)
        time_left_for_modelfit = int(max(0, total_walltime_limit - elapsed_time))
        if func_eval_time_limit_secs is None or func_eval_time_limit_secs > time_left_for_modelfit:
            self._logger.warning(
                'Time limit for a single run is higher than total time '
                'limit. Capping the limit for a single run to the total '
                'time given to SMAC (%f)' % time_left_for_modelfit
            )
            func_eval_time_limit_secs = time_left_for_modelfit

        # Make sure that at least 2 models are created for the ensemble process
        num_models = time_left_for_modelfit // func_eval_time_limit_secs
        if num_models < 2 and self.ensemble_size > 0:
            func_eval_time_limit_secs = time_left_for_modelfit // 2
            self._logger.warning(
                "Capping the func_eval_time_limit_secs to {} to have "
                "time for a least 2 models to ensemble.".format(
                    func_eval_time_limit_secs
                )
            )

        posthoc_ensemble_fit = posthoc_ensemble_fit \
            and self.base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation
        self.pipeline_options['func_eval_time_limit_secs'] = func_eval_time_limit_secs
        # ============> Run dummy predictions
        # We only want to run dummy predictions in case we want to build an ensemble
        if (
            self.ensemble_size > 0
            and (
                 self.base_ensemble_method != BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation
                 or (
                    self.base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation
                    and self.stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models
                 )
                 )
            ):
            dummy_task_name = 'runDummy'
            self._stopwatch.start_task(dummy_task_name)
            self._do_dummy_prediction()
            self._stopwatch.stop_task(dummy_task_name)

        # ============> Run traditional ml
        # We only want to run traditional predictions in case we want to build an ensemble
        # We want time for at least 1 Neural network in SMAC
        if (
            enable_traditional_pipeline
            and self.ensemble_size > 0
            and (
                 self.base_ensemble_method != BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation
                 or (
                    self.base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation
                    and self.stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models
                 )
            )
            ):
            traditional_runtime_limit = int(self._time_for_task - func_eval_time_limit_secs)
            self.run_traditional_ml(current_task_name=self.dataset_name,
                                    runtime_limit=traditional_runtime_limit,
                                    func_eval_time_limit_secs=func_eval_time_limit_secs)

        # ============> Starting ensemble
        self.use_ensemble_opt_loss = use_ensemble_opt_loss
        if self.stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models:
            elapsed_time = self._stopwatch.wall_elapsed(self.dataset_name)
            time_left_for_stacking = max(0, total_walltime_limit - elapsed_time)
            self._run_search_stacking(
                optimize_metric=optimize_metric,
                min_budget=min_budget,
                max_budget=max_budget,
                smac_scenario_args=smac_scenario_args,
                total_walltime_limit=time_left_for_stacking,
                func_eval_time_limit_secs=func_eval_time_limit_secs,
                portfolio_selection=portfolio_selection,
                tae_func=tae_func,
                precision=precision,
                experiment_task_name=experiment_task_name,
                posthoc_ensemble_fit=posthoc_ensemble_fit,
                smbo_class=smbo_class,
                enable_traditional_pipeline=enable_traditional_pipeline
            )
        else:
            self.precision = precision
            self.opt_metric = optimize_metric
            elapsed_time = self._stopwatch.wall_elapsed(self.dataset_name)
            time_left_for_ensembles = max(0, total_walltime_limit - elapsed_time)
            time_left_for_ensembles = int(time_left_for_ensembles * TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT) if posthoc_ensemble_fit else time_left_for_ensembles
            proc_ensemble = None
            if time_left_for_ensembles <= 0:
                # Fit only raises error when ensemble_size is not zero but
                # time_left_for_ensembles is zero.
                if self.ensemble_size > 0:
                    raise ValueError("Not starting ensemble builder because there "
                                    "is no time left. Try increasing the value "
                                    "of time_left_for_this_task.")
            elif self.ensemble_size <= 0:
                self._logger.info("Not starting ensemble builder as ensemble size is 0")
            else:
                proc_ensemble = self._init_ensemble_builder(time_left_for_ensembles=time_left_for_ensembles,
                                                            ensemble_size=self.ensemble_size,
                                                            ensemble_nbest=self.ensemble_nbest,
                                                            precision=precision,
                                                            optimize_metric=self.opt_metric,
                                                            base_ensemble_method=self.base_ensemble_method,
                                                            stacking_ensemble_method=self.stacking_ensemble_method,
                                                            num_stacking_layers=self.num_stacking_layers
                                                            )

            self._run_smbo(
                min_budget=min_budget,
                max_budget=max_budget,
                total_walltime_limit=total_walltime_limit * TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT \
                    if posthoc_ensemble_fit \
                        else total_walltime_limit,
                func_eval_time_limit_secs=func_eval_time_limit_secs,
                smac_scenario_args=smac_scenario_args,
                get_smac_object_callback=get_smac_object_callback,
                tae_func=tae_func,
                portfolio_selection=portfolio_selection,
                smbo_class=smbo_class,
                experiment_task_name=experiment_task_name,
                proc_ensemble=proc_ensemble,
                num_stacking_layers=self.num_stacking_layers
                )

            if proc_ensemble is not None:
                self._collect_results_ensemble(proc_ensemble)
            # Wait until the ensemble process is finished to avoid shutting down
            # while the ensemble builder tries to access the data
            self._logger.info("Starting Shutdown")

        if posthoc_ensemble_fit and self.stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_ensemble_bayesian_optimisation:
            ensemble = self._backend.load_ensemble(self.seed)
            initial_num_run = int(open(os.path.join(self._backend.internals_directory, 'ensemble_cutoff_run.txt'), 'r').read())
            time_for_post_fit_ensemble = max(0, total_walltime_limit-self._stopwatch.wall_elapsed(self.dataset_name))
            iteration = (self.num_stacking_layers+1)*ENSEMBLE_ITERATION_MULTIPLIER
            final_model_identifiers, final_weights = self._posthoc_fit_ensemble(
                optimize_metric=self.opt_metric,
                time_left_for_ensemble=time_for_post_fit_ensemble,
                last_successful_smac_initial_num_run=initial_num_run + 1,
                ensemble_size=self.ensemble_size,
                iteration=iteration,
                enable_traditional_pipeline=enable_traditional_pipeline,
                cleanup=False,
                func_eval_time_limit_secs=func_eval_time_limit_secs
            )
            ensemble.identifiers_ = final_model_identifiers
            stacked_ensemble_identifiers = ensemble.stacked_ensemble_identifiers
            broken = False
            for i, layer_identifiers in enumerate(stacked_ensemble_identifiers):
                if all([identifier is None for identifier in layer_identifiers]):
                    broken = True
                    break
            last_nonnull_layer = i-1 if broken else i
            self._logger.debug(f"broken: {broken}, lastnonnull layer: {last_nonnull_layer}, i: {i}")
            ensemble.stacked_ensemble_identifiers[last_nonnull_layer] = final_model_identifiers
            ensemble.weights_ = final_weights
            self._backend.save_ensemble(ensemble, iteration+1, self.seed)

        if load_models:
            self._logger.info("Loading models...")
            self._load_models()
            self._logger.info("Finished loading models...")

        self._cleanup()

        return self

    def _init_required_args(
        self,
        experiment_task_name: str,
        optimize_metric: str,
        dataset: BaseDataset,
        budget_type: str,
        max_budget: int,
        total_walltime_limit: int,
        memory_limit: int,
        all_supported_metrics: bool,
        precision: int,
        dask_client: Optional[dask.distributed.Client] = None,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None
    ) -> None:
        if self.task_type != dataset.task_type:
            raise ValueError("Incompatible dataset entered for current task,"
                             "expected dataset to have task type :{} but got "
                             ":{}".format(self.task_type, dataset.task_type))
        if precision not in [16, 32, 64]:
            raise ValueError("precision must be one of 16, 32, 64 but got {}".format(precision))

        # Initialise information needed for the experiment
        dataset_properties = self._get_dataset_properties(dataset)
        self._stopwatch.start_task(experiment_task_name)
        self.dataset_name = dataset.dataset_name
        assert self.dataset_name is not None

        if self._logger is None:
            self._logger = self._get_logger(self.dataset_name)

        # Setup the logger for the backend
        self._backend.setup_logger(port=self._logger_port)

        self._all_supported_metrics = all_supported_metrics
        self._disable_file_output = disable_file_output if disable_file_output is not None else []
        if (
            DisableFileOutputParameters.y_optimization in self._disable_file_output
            and self.ensemble_size > 1
        ):
            self._logger.warning(f"No ensemble will be created when {DisableFileOutputParameters.y_optimization}"
                                 f" is in disable_file_output")

        self._memory_limit = memory_limit
        self._time_for_task = total_walltime_limit
        # Save start time to backend
        self._backend.save_start_time(str(self.seed))

        self._backend.save_datamanager(dataset)

        # Print debug information to log
        self._print_debug_info_to_log()

        self._metric = get_metrics(
            names=[optimize_metric], dataset_properties=dataset_properties)[0]

        self.pipeline_options['optimize_metric'] = optimize_metric

        if all_supported_metrics:
            self._scoring_functions = get_metrics(dataset_properties=dataset_properties,
                                                  all_supported_metrics=True)
        else:
            self._scoring_functions = [self._metric]

        self.search_space = self.get_search_space(dataset)

        # Incorporate budget to pipeline config
        if budget_type not in ('epochs', 'runtime'):
            raise ValueError("Budget type must be one ('epochs', 'runtime')"
                             f" yet {budget_type} was provided")
        self.pipeline_options['budget_type'] = budget_type

        # Here the budget is set to max because the SMAC intensifier can be:
        # Hyperband: in this case the budget is determined on the fly and overwritten
        #            by the ExecuteTaFuncWithQueue
        # SimpleIntensifier (and others): in this case, we use max_budget as a target
        #                                 budget, and hece the below line is honored
        self.pipeline_options[budget_type] = max_budget

        if self.task_type is None:
            raise ValueError("Cannot interpret task type from the dataset")

        # If no dask client was provided, we create one, so that we can
        # start a ensemble process in parallel to smbo optimize
        if self.n_jobs == 1:
            self._dask_client = SingleThreadedClient()
        elif dask_client is None:
            self._create_dask_client()
        else:
            self._dask_client = dask_client
            self._is_dask_client_internally_created = False
        return

    def _get_dataset_properties(self, dataset):
        dataset_requirements = get_dataset_requirements(
            info=dataset.get_required_dataset_info(),
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates)
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        return dataset_properties

    def _get_fit_dictionary(
        self,
        dataset_properties: Dict[str, BaseDatasetPropertiesType],
        dataset: BaseDataset,
        split_id: int = 0
    ) -> Dict[str, Any]:
        X_test = dataset.test_tensors[0].copy() if dataset.test_tensors is not None else None
        y_test = dataset.test_tensors[1].copy() if dataset.test_tensors is not None else None
        X: Dict[str, Any] = dict({'dataset_properties': dataset_properties,
                                  'backend': self._backend,
                                  'X_train': dataset.train_tensors[0].copy(),
                                  'y_train': dataset.train_tensors[1].copy(),
                                  'X_test': X_test,
                                  'y_test': y_test,
                                  'train_indices': dataset.splits[split_id][0],
                                  'val_indices': dataset.splits[split_id][1],
                                  'split_id': split_id,
                                  'num_run': self._backend.get_next_num_run(),
                                  })
        X.update(self.pipeline_options)
        return X

    def refit(
        self,
        dataset: BaseDataset,
        split_id: int = 0
    ) -> "BaseTask":
        """
        Refit all models found with fit to new data.

        Necessary when using cross-validation. During training, autoPyTorch
        fits each model k times on the dataset, but does not keep any trained
        model and can therefore not be used to predict for new data points.
        This methods fits all models found during a call to fit on the data
        given. This method may also be used together with holdout to avoid
        only using 66% of the training data to fit the final model.

        Refit uses the estimator pipeline_config attribute, which the user
        can interact via the get_pipeline_config()/set_pipeline_config()
        methods.

        Args:
            dataset (Dataset):
                The argument that will provide the dataset splits. It can either
                be a dictionary with the splits, or the dataset object which can
                generate the splits based on different restrictions.
            split_id (int):
                split id to fit on.
        Returns:
            self
        """

        self.dataset_name = dataset.dataset_name

        if self._logger is None:
            self._logger = self._get_logger(str(self.dataset_name))

        dataset_requirements = get_dataset_requirements(
            info=dataset.get_required_dataset_info(),
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates)
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # Refit is not applicable when ensemble_size is set to zero.
        if self.ensemble_ is None:
            raise ValueError("Refit can only be called if 'ensemble_size != 0'")

        for identifier in self.models_:
            model = self.models_[identifier]
            # this updates the model inplace, it can then later be used in
            # predict method

            # try to fit the model. If it fails, shuffle the data. This
            # could alleviate the problem in algorithms that depend on
            # the ordering of the data.
            X = self._get_fit_dictionary(
                dataset_properties=dataset_properties,
                dataset=dataset,
                split_id=split_id)
            fit_and_suppress_warnings(self._logger, model, X, y=None)

        self._clean_logger()

        return self

    def fit_pipeline(
        self,
        configuration: Configuration,
        *,
        dataset: Optional[BaseDataset] = None,
        X_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_train: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        X_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[List, pd.DataFrame, np.ndarray]] = None,
        dataset_name: Optional[str] = None,
        resampling_strategy: Optional[ResamplingStrategies] = None,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        run_time_limit_secs: int = 60,
        memory_limit: Optional[int] = None,
        eval_metric: Optional[str] = None,
        all_supported_metrics: bool = False,
        budget_type: Optional[str] = None,
        include_components: Optional[Dict[str, Any]] = None,
        exclude_components: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
        budget: Optional[float] = None,
        pipeline_options: Optional[Dict] = None,
        disable_file_output: Optional[List[Union[str, DisableFileOutputParameters]]] = None,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue, BaseDataset]:
        """
        Fit a pipeline on the given task for the budget.
        A pipeline configuration can be specified if None,
        uses default

        Fit uses the estimator pipeline_config attribute, which the user
        can interact via the get_pipeline_config()/set_pipeline_config()
        methods.

        Args:
            configuration (Configuration):
                configuration to fit the pipeline with.
            dataset (BaseDataset):
                An object of the appropriate child class of `BaseDataset`,
                that will be used to fit the pipeline
            X_train, y_train, X_test, y_test: Union[np.ndarray, List, pd.DataFrame]
                A pair of features (X_train) and targets (y_train) used to fit a
                pipeline. Additionally, a holdout of this pairs (X_test, y_test) can
                be provided to track the generalization performance of each stage.
            dataset_name (Optional[str]):
                Name of the dataset, if None, random value is used.
            resampling_strategy (Optional[RESAMPLING_STRATEGIES]):
                Strategy to split the training data. if None, uses
                HoldoutValTypes.holdout_validation.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                Arguments required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            dataset_name (Optional[str]):
                name of the dataset, used as experiment name.
            run_time_limit_secs (int: default=60):
                Time limit for a single call to the machine learning model.
                Model fitting will be terminated if the machine learning algorithm
                runs over the time limit. Set this value high enough so that
                typical machine learning algorithms can be fit on the training
                data.
            memory_limit (Optional[int]):
                Memory limit in MB for the machine learning algorithm. autopytorch
                will stop fitting the machine learning algorithm if it tries
                to allocate more than memory_limit MB. If None is provided,
                no memory limit is set. In case of multi-processing, memory_limit
                will be per job. This memory limit also applies to the ensemble
                creation process.
            eval_metric (Optional[str]):
                Name of the metric that is used to evaluate a pipeline.
            all_supported_metrics (bool: default=True):
                if True, all metrics supporting current task will be calculated
                for each pipeline and results will be available via cv_results
            budget_type (str):
                Type of budget to be used when fitting the pipeline.
                It can be one of:

                + `epochs`: The training of each pipeline will be terminated after
                    a number of epochs have passed. This number of epochs is determined by the
                    budget argument of this method.
                + `runtime`: The training of each pipeline will be terminated after
                    a number of seconds have passed. This number of seconds is determined by the
                    budget argument of this method. The overall fitting time of a pipeline is
                    controlled by func_eval_time_limit_secs. 'runtime' only controls the allocated
                    time to train a pipeline, but it does not consider the overall time it takes
                    to create a pipeline (data loading and preprocessing, other i/o operations, etc.).
            include_components (Optional[Dict[str, Any]]):
                Dictionary containing components to include. Key is the node
                name and Value is an Iterable of the names of the components
                to include. Only these components will be present in the
                search space.
            exclude_components (Optional[Dict[str, Any]]):
                Dictionary containing components to exclude. Key is the node
                name and Value is an Iterable of the names of the components
                to exclude. All except these components will be present in
                the search space.
            search_space_updates(Optional[HyperparameterSearchSpaceUpdates]):
                Updates to be made to the hyperparameter search space of the pipeline
            budget (Optional[float]):
                Budget to fit a single run of the pipeline. If not
                provided, uses the default in the pipeline config
            pipeline_options (Optional[Dict]):
                Valid config options include "device",
                "torch_num_threads", "early_stopping", "use_tensorboard_logger",
                "metrics_during_training"
            disable_file_output (Optional[List[Union[str, DisableFileOutputParameters]]]):
                Used as a list to pass more fine-grained
                information on what to save. Must be a member of `DisableFileOutputParameters`.
                Allowed elements in the list are:

                + `y_optimization`:
                    do not save the predictions for the optimization set,
                    which would later on be used to build an ensemble. Note that SMAC
                    optimizes a metric evaluated on the optimization set.
                + `pipeline`:
                    do not save any individual pipeline files
                + `pipelines`:
                    In case of cross validation, disables saving the joint model of the
                    pipelines fit on each fold.
                + `y_test`:
                    do not save the predictions for the test set.
                + `all`:
                    do not save any of the above.
                For more information check `autoPyTorch.evaluation.utils.DisableFileOutputParameters`.

        Returns:
            (BasePipeline):
                fitted pipeline
            (RunInfo):
                Run information
            (RunValue):
                Result of fitting the pipeline
            (BaseDataset):
                Dataset created from the given tensors
        """

        if dataset is None:
            if (
                X_train is not None
                and y_train is not None
            ):
                raise ValueError("No dataset provided, must provide X_train, y_train tensors")
            dataset = self.get_dataset(X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test,
                                       y_test=y_test,
                                       resampling_strategy=resampling_strategy,
                                       resampling_strategy_args=resampling_strategy_args,
                                       dataset_name=dataset_name
                                       )

        # dataset_name is created inside the constructor of BaseDataset
        # we expect it to be not None. This is for mypy
        assert dataset.dataset_name is not None

        # TAE expects each configuration to have a config_id.
        # For fitting a pipeline as it is not part of the
        # search process, it makes sense to set it to 0
        configuration.__setattr__('config_id', 0)

        # get dataset properties
        dataset_requirements = get_dataset_requirements(
            info=dataset.get_required_dataset_info(),
            include=self.include_components,
            exclude=self.exclude_components,
            search_space_updates=self.search_space_updates)
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        self._backend.save_datamanager(dataset)

        if self._logger is None:
            self._logger = self._get_logger(dataset.dataset_name)

        include_components = self.include_components if include_components is None else include_components
        exclude_components = self.exclude_components if exclude_components is None else exclude_components
        search_space_updates = self.search_space_updates if search_space_updates is None else search_space_updates

        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = run_time_limit_secs
        # This stats object is a hack - maybe the SMAC stats object should
        # already be generated here!
        stats = Stats(scenario_mock)

        if memory_limit is None and getattr(self, '_memory_limit', None) is not None:
            memory_limit = self._memory_limit

        metric = get_metrics(dataset_properties=dataset_properties,
                             names=[eval_metric] if eval_metric is not None else None,
                             all_supported_metrics=False).pop()

        pipeline_options = self.pipeline_options.copy().update(pipeline_options) if pipeline_options is not None \
            else self.pipeline_options.copy()
        pipeline_options['func_eval_time_limit_secs'] = run_time_limit_secs
        assert pipeline_options is not None

        if budget_type is not None:
            pipeline_options.update({'budget_type': budget_type})
        else:
            budget_type = pipeline_options['budget_type']

        budget = budget if budget is not None else pipeline_options[budget_type]

        if disable_file_output is None:
            disable_file_output = getattr(self, '_disable_file_output', [])

        stats.start_timing()

        tae = ExecuteTaFuncWithQueue(
            backend=self._backend,
            seed=self.seed,
            metric=metric,
            multi_objectives=["cost"],
            logger_port=self._logger_port,
            cost_for_crash=get_cost_of_crash(metric),
            abort_on_first_run_crash=False,
            initial_num_run=self._backend.get_next_num_run(),
            stats=stats,
            memory_limit=memory_limit,
            disable_file_output=disable_file_output,
            all_supported_metrics=all_supported_metrics,
            budget_type=budget_type,
            include=include_components,
            exclude=exclude_components,
            search_space_updates=search_space_updates,
            pipeline_config=pipeline_options,
            pynisher_context=self._multiprocessing_context
        )

        run_info, run_value = tae.run_wrapper(
            RunInfo(config=configuration,
                    budget=budget,
                    seed=self.seed,
                    cutoff=run_time_limit_secs,
                    capped=False,
                    instance_specific=None,
                    instance=None)
        )

        fitted_pipeline = self._get_fitted_pipeline(
            dataset_name=dataset.dataset_name,
            pipeline_idx=run_info.config.config_id + tae.initial_num_run,
            run_info=run_info,
            run_value=run_value,
            disable_file_output=disable_file_output
        )

        self._clean_logger()

        return fitted_pipeline, run_info, run_value, dataset

    def _get_fitted_pipeline(
        self,
        dataset_name: str,
        pipeline_idx: int,
        run_info: RunInfo,
        run_value: RunValue,
        disable_file_output: List[Union[str, DisableFileOutputParameters]]
    ) -> Optional[BasePipeline]:

        if self._logger is None:
            self._logger = self._get_logger(str(dataset_name))

        if run_value.status != StatusType.SUCCESS:
            warnings.warn(f"Fitting pipeline failed with status: {run_value.status}"
                          f", additional_info: {run_value.additional_info}")
            return None
        elif any(disable_file_output for c in ['all', 'pipeline']):
            self._logger.warning("File output is disabled. No pipeline can returned")
            return None

        if self.resampling_strategy in CrossValTypes:
            load_function = self._backend.load_cv_model_by_seed_and_id_and_budget
        else:
            load_function = self._backend.load_model_by_seed_and_id_and_budget

        return load_function(  # type: ignore[no-any-return]
            seed=self.seed,
            idx=pipeline_idx,
            budget=float(run_info.budget),
        )

    def fit_ensemble(
            self,
            optimize_metric: Optional[str] = None,
            precision: Optional[int] = None,
            ensemble_nbest: int = 50,
            ensemble_size: int = 50,
            base_ensemble_method: int = BaseLayerEnsembleSelectionTypes.ensemble_selection,
            stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None,
            num_stacking_layers: int = 1,
            initial_num_run: int = 0,
            load_models: bool = True,
            time_for_task: int = 100,
            func_eval_time_limit_secs: int = 50,
            enable_traditional_pipeline: bool = True,
            iteration: int = 0,
            cleanup: bool = True
    ) -> 'BaseTask':
        """
        Enables post-hoc fitting of the ensemble after the `search()`
        method is finished. This method creates an ensemble using all
        the models stored on disk during the smbo run.

        Args:
            optimize_metric (str): name of the metric that is used to
                evaluate a pipeline. if not specified, value passed to search will be used
            precision (Optional[int]): Numeric precision used when loading
                ensemble data. Can be either 16, 32 or 64.
            ensemble_nbest (Optional[int]):
                only consider the ensemble_nbest models to build the ensemble.
                If None, uses the value stored in class attribute `ensemble_nbest`.
            ensemble_size (int) (default=50):
                Number of models added to the ensemble built by
                Ensemble selection from libraries of models.
                Models are drawn with replacement.
            enable_traditional_pipeline (bool), (default=True):
                We fit traditional machine learning algorithms
                (LightGBM, CatBoost, RandomForest, ExtraTrees, KNN, SVM)
                prior building PyTorch Neural Networks. You can disable this
                feature by turning this flag to False. All machine learning
                algorithms that are fitted during search() are considered for
                ensemble building.
            load_models (bool), (default=True): Whether to load the
                models after fitting AutoPyTorch.
            time_for_task (int), (default=100): Time limit
                in seconds for the search of appropriate models.
                By increasing this value, autopytorch has a higher
                chance of finding better models.
            func_eval_time_limit_secs (int), (default=None): Time limit
                for a single call to the machine learning model.
                Model fitting will be terminated if the machine
                learning algorithm runs over the time limit. Set
                this value high enough so that typical machine
                learning algorithms can be fit on the training
                data.
                When set to None, this time will automatically be set to
                total_walltime_limit // 2 to allow enough time to fit
                at least 2 individual machine learning algorithms.
                Set to np.inf in case no time limit is desired.

        Returns:
            self
        """
        # Make sure that input is valid
        if self.dataset is None:
            raise ValueError("fit_ensemble() can only be called after `search()`. "
                             "Please call the `search()` method of {} prior to "
                             "fit_ensemble().".format(self.__class__.__name__))

        precision = precision if precision is not None else self.precision
        if precision not in [16, 32, 64]:
            raise ValueError("precision must be one of 16, 32, 64 but got {}".format(precision))

        if self._logger is None:
            self._logger = self._get_logger(self.dataset.dataset_name)

        # Create a client if needed
        if self._dask_client is None:
            self._create_dask_client()
        else:
            self._is_dask_client_internally_created = False

        ensemble_fit_task_name = 'EnsembleFit'
        self._stopwatch.start_task(ensemble_fit_task_name)
        if enable_traditional_pipeline:
            if func_eval_time_limit_secs > time_for_task:
                self._logger.warning(
                    'Time limit for a single run is higher than total time '
                    'limit. Capping the limit for a single run to the total '
                    'time given to Ensemble fit (%f)' % time_for_task
                )
                func_eval_time_limit_secs = time_for_task

            # Make sure that at least 2 models are created for the ensemble process
            num_models = time_for_task // func_eval_time_limit_secs
            if num_models < 2:
                func_eval_time_limit_secs = time_for_task // 2
                self._logger.warning(
                    "Capping the func_eval_time_limit_secs to {} to have "
                    "time for at least 2 models to ensemble.".format(
                        func_eval_time_limit_secs
                    )
                )
        # ============> Run Dummy predictions
        dummy_task_name = 'runDummy'
        self._stopwatch.start_task(dummy_task_name)
        self._do_dummy_prediction()
        self._stopwatch.stop_task(dummy_task_name)

        # ============> Run traditional ml
        if enable_traditional_pipeline:
            self.run_traditional_ml(current_task_name=ensemble_fit_task_name,
                                    runtime_limit=time_for_task,
                                    func_eval_time_limit_secs=func_eval_time_limit_secs)

        elapsed_time = self._stopwatch.wall_elapsed(ensemble_fit_task_name)
        time_left_for_ensemble = int(time_for_task - elapsed_time)
        self._logger.debug(f"Starting to initialise ensemble with {time_left_for_ensemble}s")
        manager = self._init_ensemble_builder(
            time_left_for_ensembles=time_left_for_ensemble,
            optimize_metric=self.opt_metric if optimize_metric is None else optimize_metric,
            precision=precision,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            base_ensemble_method=base_ensemble_method,
            stacking_ensemble_method=stacking_ensemble_method,
            num_stacking_layers=num_stacking_layers,
            initial_num_run=initial_num_run,
            iteration=iteration
        )

        manager.build_ensemble(self._dask_client)
        if manager is not None:
            self._collect_results_ensemble(manager)

        if load_models:
            self._load_models()

        self._stopwatch.stop_task(ensemble_fit_task_name)

        if cleanup:
            self._cleanup()

        return self

    def _init_ensemble_builder(
            self,
            time_left_for_ensembles: float,
            optimize_metric: str,
            base_ensemble_method: int,
            ensemble_nbest: int,
            ensemble_size: int,
            stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None,
            num_stacking_layers: Optional[int] = None,
            precision: int = 32,
            initial_num_run: int = 0,
            iteration: int = 0,
    ) -> EnsembleBuilderManager:
        """
        Initializes an `EnsembleBuilderManager`.
        Args:
            time_left_for_ensembles (float):
                Time (in seconds) allocated to building the ensemble
            optimize_metric (str):
                Name of the metric to optimize the ensemble.
            ensemble_nbest (int):
                only consider the ensemble_nbest models to build the ensemble.
            ensemble_size (int):
                Number of models added to the ensemble built by
                Ensemble selection from libraries of models.
                Models are drawn with replacement.
            precision (int), (default=32): Numeric precision used when loading
                ensemble data. Can be either 16, 32 or 64.

        Returns:
            EnsembleBuilderManager
        """
        if self._logger is None:
            raise ValueError("logger should be initialized to fit ensemble")
        if self.dataset is None:
            raise ValueError("ensemble can only be initialised after or during `search()`. "
                             "Please call the `search()` method of {}.".format(self.__class__.__name__))

        self._logger.info("Starting ensemble")
        ensemble_task_name = 'ensemble'
        self._stopwatch.start_task(ensemble_task_name)

        # Use the current thread to start the ensemble builder process
        # The function ensemble_builder_process will internally create a ensemble
        # builder in the provide dask client
        required_dataset_properties = {'task_type': self.task_type,
                                       'output_type': self.dataset.output_type}
        metrics = get_metrics(
                dataset_properties=required_dataset_properties, names=[optimize_metric])
        self._logger.info(f"metrics are {metrics}")
        proc_ensemble = EnsembleBuilderManager(
            start_time=time.time(),
            time_left_for_ensembles=time_left_for_ensembles,
            backend=copy.deepcopy(self._backend),
            dataset_name=str(self.dataset.dataset_name),
            output_type=STRING_TO_OUTPUT_TYPES[self.dataset.output_type],
            task_type=STRING_TO_TASK_TYPES[self.task_type],
            metrics=get_metrics(
                dataset_properties=required_dataset_properties, names=[optimize_metric]),
            opt_metric=optimize_metric,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            base_ensemble_method=base_ensemble_method,
            stacking_ensemble_method=stacking_ensemble_method,
            max_models_on_disc=self.max_models_on_disc,
            seed=self.seed,
            max_iterations=None,
            read_at_most=sys.maxsize,
            ensemble_memory_limit=self._memory_limit,
            random_state=self.seed,
            precision=precision,
            logger_port=self._logger_port,
            use_ensemble_loss=self.use_ensemble_opt_loss,
            num_stacking_layers=num_stacking_layers,
            initial_num_run=initial_num_run,
            iteration=iteration
        )
        self._stopwatch.stop_task(ensemble_task_name)

        return proc_ensemble

    def _collect_results_ensemble(
        self,
        manager: EnsembleBuilderManager
    ) -> None:

        if self._logger is None:
            raise ValueError("logger should be initialized to fit ensemble")

        self._results_manager.ensemble_performance_history = list(manager.history)

        if len(manager.futures) > 0:
            # Also add ensemble runs that did not finish within smac time
            # and add them into the ensemble history
            self._logger.info("Ensemble script still running, waiting for it to finish.")
            result = manager.futures.pop().result()
            if result:
                ensemble_history, _, _, _ = result
                self._results_manager.ensemble_performance_history.extend(ensemble_history)
            self._logger.info("Ensemble script finished, continue shutdown.")

        # save the ensemble performance history file
        if len(self.ensemble_performance_history) > 0:
            pd.DataFrame(self.ensemble_performance_history).to_json(
                os.path.join(self._backend.internals_directory, 'ensemble_history.json'))

    def _init_result_history_updater(self, initial_num_run: int) -> RunHistoryUpdaterManager:
        if self.dataset is None:
            raise ValueError("runhistory updater can only be initialised after or during `search()`. "
                             "Please call the `search()` method of {}.".format(self.__class__.__name__))

        self._logger.info("Starting Runhistory updater")
        runhistory_task_name = 'runhistory_updater'
        self._stopwatch.start_task(runhistory_task_name)

        proc_runhistory_updater = RunHistoryUpdaterManager(
            backend=self._backend,
            dataset_name=self.dataset_name,
            resampling_strategy=self.resampling_strategy,
            resampling_strategy_args=self.resampling_strategy_args,
            logger_port=self._logger_port,
            initial_num_run=initial_num_run
        )

        self._stopwatch.stop_task(runhistory_task_name)

        return proc_runhistory_updater

    def predict(
        self,
        X_test: np.ndarray,
        batch_size: Optional[int] = None,
        n_jobs: int = 1
    ) -> np.ndarray:
        """Generate the estimator predictions.
        Generate the predictions based on the given examples from the test set.

        Args:
            X_test (np.ndarray):
                The test set examples.

        Returns:
            Array with estimator predictions.
        """

        # Parallelize predictions across models with n_jobs processes.
        # Each process computes predictions in chunks of batch_size rows.
        if self._logger is None:
            self._logger = self._get_logger("Predict-Logger")

        if self.ensemble_ is None and not self._load_models():
            raise ValueError("No ensemble found. Either fit has not yet "
                             "been called or no ensemble was fitted")

        # Mypy assert
        assert self.ensemble_ is not None, "Load models should error out if no ensemble"
        predictions = self._predict_with_ensemble(X_test=X_test, batch_size=batch_size, n_jobs=n_jobs)        

        self._cleanup()

        return predictions


    def _predict_with_ensemble(self, X_test, batch_size, n_jobs) -> np.ndarray:

        assert self.ensemble_ is not None, "Load models should error out if no ensemble"
        if isinstance(self.resampling_strategy, (HoldoutValTypes, NoResamplingStrategyTypes)):
            models = self.models_
        elif isinstance(self.resampling_strategy, (CrossValTypes, RepeatedCrossValTypes)):
            models = self.cv_models_

        X_test_copy = X_test.copy()
        if is_stacking(self.base_ensemble_method, self.stacking_ensemble_method):
            ensemble_identifiers = self.ensemble_.get_selected_model_identifiers()
            self._logger.debug(f"ensemble identifiers: {ensemble_identifiers}")
            for i, (model, layer_identifiers) in enumerate(zip(models, ensemble_identifiers)):
                if all([identifier is None for identifier in layer_identifiers]):
                    break
                self._logger.debug(f"layer : {i} of stacking ensemble,\n layer identifiers: {layer_identifiers},\n model: {model}")
                all_predictions = joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(_pipeline_predict)(
                        model[identifier], X_test_copy, batch_size, self._logger, STRING_TO_TASK_TYPES[self.task_type]
                    )
                    for identifier in layer_identifiers if identifier is not None
                )
                if (
                    self.base_ensemble_method in (BaseLayerEnsembleSelectionTypes.ensemble_autogluon, BaseLayerEnsembleSelectionTypes.ensemble_selection)
                    or self.stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models
                ):
                    concat_all_predictions = self.ensemble_.get_expanded_layer_stacking_ensemble_predictions(
                        stacking_layer=i, raw_stacking_layer_ensemble_predictions=all_predictions)
                else:
                    concat_all_predictions = all_predictions

                X_test_copy = np.concatenate([X_test, *concat_all_predictions], axis=1)
                self._logger.debug(f"shap of X_test after predict with layer : {i} = {X_test_copy.shape}")
        else:
            all_predictions = joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(_pipeline_predict)(
                        models[identifier], X_test_copy, batch_size, self._logger, STRING_TO_TASK_TYPES[self.task_type]
                    )
                    for identifier in self.ensemble_.get_selected_model_identifiers()
                )
    
        if len(all_predictions) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_.indices_)),
                                     str(list(self.models_))))

        predictions = self.ensemble_.predict(all_predictions)

        self._cleanup()

        return predictions

    def score(
        self,
        y_pred: np.ndarray,
        y_test: Union[np.ndarray, pd.DataFrame],
        metric: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate the score on the test set.
        Calculate the evaluation measure on the test set.

        Args:
            y_pred (np.ndarray):
                The test predictions
            y_test (np.ndarray):
                The test ground truth labels.

        Returns:
            Dict[str, float]:
                Value of the evaluation metric calculated on the test set.
        """
        if metric is not None:
            required_dataset_properties = {'task_type': self.task_type,
                                       'output_type': self.dataset.output_type}
            metric = get_metrics(
                dataset_properties=required_dataset_properties,
                names=[metric]
                )[0]
        else:
            metric = self._metric

        if self.task_type is None:
            raise ValueError("AutoPytorch failed to infer a task type from the dataset "
                             "Please check the log file for related errors. ")
        return calculate_score(target=y_test, prediction=y_pred,
                               task_type=STRING_TO_TASK_TYPES[self.task_type],
                               metrics=[metric])

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a client!
        self._dask_client = None
        self.logging_server = None  # type: ignore [assignment]
        self.stop_logging_server = None
        return self.__dict__

    def __del__(self) -> None:
        self._cleanup()

        # When a multiprocessing work is done, the
        # objects are deleted. We don't want to delete run areas
        # until the estimator is deleted
        if hasattr(self, '_backend'):
            self._backend.context.delete_directories(force=False)

    def get_incumbent_results(
        self,
        include_traditional: bool = False
    ) -> Tuple[Configuration, Dict[str, Union[int, str, float]]]:
        """
        Get Incumbent config and the corresponding results

        Args:
            include_traditional (bool):
                Whether to include results from tradtional pipelines

        Returns:
            Configuration (CS.ConfigurationSpace):
                The incumbent configuration
            Dict[str, Union[int, str, float]]:
                Additional information about the run of the incumbent configuration.
        """

        if self._metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        return self._results_manager.get_incumbent_results(metric=self._metric, include_traditional=include_traditional)

    def get_models_with_weights(self) -> List:
        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        assert self.ensemble_ is not None
        models_with_weights: List[Tuple[float, BasePipeline]] = self.ensemble_.get_models_with_weights(self.models_)
        return models_with_weights

    def show_models(self) -> str:
        """
        Returns a Markdown containing details about the final ensemble/configuration.

        Returns:
            str:
                Markdown table of models.
        """
        if is_stacking(self.base_ensemble_method, self.stacking_ensemble_method):
            df = []
            for layer, model_weight in enumerate(self.get_models_with_weights()):
                for weight, model in model_weight:
                    representation = model.get_pipeline_representation()
                    representation.update({'Weight': weight, "Stacking Layer": layer})
                    df.append(representation)
            models_markdown: str = pd.DataFrame(df).to_markdown()
            return models_markdown
        else:
            df = []
            for weight, model in self.get_models_with_weights():
                representation = model.get_pipeline_representation()
                representation.update({'Weight': weight})
                df.append(representation)
            models_markdown: str = pd.DataFrame(df).to_markdown()
            return models_markdown

    def _print_debug_info_to_log(self) -> None:
        """
        Prints to the log file debug information about the current estimator
        """
        assert self._logger is not None
        self._logger.debug("Starting to print environment information")
        self._logger.debug('  Python version: %s', sys.version.split('\n'))
        self._logger.debug('  System: %s', platform.system())
        self._logger.debug('  Machine: %s', platform.machine())
        self._logger.debug('  Platform: %s', platform.platform())
        self._logger.debug('  multiprocessing_context: %s', str(self._multiprocessing_context))
        for key, value in vars(self).items():
            self._logger.debug(f"\t{key}->{value}")

    def get_search_results(self) -> SearchResults:
        """
        Get the interface to obtain the search results easily.
        """
        if self._scoring_functions is None or self._metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        return self._results_manager.get_search_results(
            metric=self._metric,
            scoring_functions=self._scoring_functions
        )

    def sprint_statistics(self) -> str:
        """
        Prints statistics about the SMAC search.

        These statistics include:

        1. Optimisation Metric
        2. Best Optimisation score achieved by individual pipelines
        3. Total number of target algorithm runs
        4. Total number of successful target algorithm runs
        5. Total number of crashed target algorithm runs
        6. Total number of target algorithm runs that exceeded the time limit
        7. Total number of successful target algorithm runs that exceeded the memory limit

        Returns:
            (str):
                Formatted string with statistics
        """
        if self._scoring_functions is None or self._metric is None:
            raise RuntimeError("`search_results` is only available after a search has finished.")

        assert self.dataset_name is not None  # my check
        return self._results_manager.sprint_statistics(
            dataset_name=self.dataset_name,
            scoring_functions=self._scoring_functions,
            metric=self._metric
        )

    def plot_perf_over_time(
        self,
        metric_name: str,
        ax: Optional[plt.Axes] = None,
        plot_setting_params: PlotSettingParams = PlotSettingParams(),
        color_label_settings: ColorLabelSettings = ColorLabelSettings(),
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Visualize the performance over time using matplotlib.
        The plot related arguments are based on matplotlib.
        Please refer to the matplotlib documentation for more details.

        Args:
            metric_name (str):
                The name of metric to visualize.
                The names are available in
                    * autoPyTorch.metrics.CLASSIFICATION_METRICS
                    * autoPyTorch.metrics.REGRESSION_METRICS
            ax (Optional[plt.Axes]):
                axis to plot (subplots of matplotlib).
                If None, it will be created automatically.
            plot_setting_params (PlotSettingParams):
                Parameters for the plot.
            color_label_settings (ColorLabelSettings):
                The settings of a pair of color and label for each plot.
            args, kwargs (Any):
                Arguments for the ax.plot.

        Note:
            You might need to run `export DISPLAY=:0.0` if you are using non-GUI based environment.
        """

        if not hasattr(metrics, metric_name):
            raise ValueError(
                f'metric_name must be in {list(metrics.CLASSIFICATION_METRICS.keys())} '
                f'or {list(metrics.REGRESSION_METRICS.keys())}, but got {metric_name}'
            )
        if len(self.ensemble_performance_history) == 0:
            raise RuntimeError('Visualization is available only after ensembles are evaluated.')

        results = MetricResults(
            metric=getattr(metrics, metric_name),
            run_history=self.run_history,
            ensemble_performance_history=self.ensemble_performance_history
        )

        colors, labels = color_label_settings.extract_dicts(results)

        ResultsVisualizer().plot_perf_over_time(  # type: ignore
            results=results, plot_setting_params=plot_setting_params,
            colors=colors, labels=labels, ax=ax,
            *args, **kwargs
        )
