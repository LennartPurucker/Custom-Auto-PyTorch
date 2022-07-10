# -*- encoding: utf-8 -*-
import logging
import logging.handlers
import os
import pickle
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

import dask.distributed

import numpy as np

import pandas as pd



from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue, RunHistory

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.ensemble.utils import (
    BaseLayerEnsembleSelectionTypes,
    StackingEnsembleSelectionTypes,
    get_ensemble_builder_class,
    is_stacking
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.single_thread_client import DummyFuture


class EnsembleBuilderManager(IncorporateRunResultCallback):
    def __init__(
        self,
        start_time: float,
        time_left_for_ensembles: float,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        output_type: int,
        metrics: List[autoPyTorchMetric],
        opt_metric: str,
        ensemble_size: int,
        ensemble_nbest: int,
        base_ensemble_method: BaseLayerEnsembleSelectionTypes,
        max_models_on_disc: Union[float, int],
        seed: int,
        precision: int,
        max_iterations: Optional[int],
        read_at_most: int,
        ensemble_memory_limit: Optional[int],
        random_state: int,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        pynisher_context: str = 'fork',
        initial_num_run: int = 0,
        use_ensemble_loss=False,
        stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None,
        num_stacking_layers: Optional[int] = None,
        iteration=0,
        ensemble_slot_j = 0,
        run_history: Optional[RunHistory] = None,
    ):
        """ SMAC callback to handle ensemble building
        Args:
            start_time: int
                the time when this job was started, to account for any latency in job allocation
            time_left_for_ensemble: int
                How much time is left for the task. Job should finish within this allocated time
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            task_type: int
                what type of output is expected. If Binary, we need to argmax the one hot encoding.
            metrics: List[autoPyTorchMetric],
                A set of metrics that will be used to get performance estimates
            opt_metric: str
                name of the optimization metrics
            ensemble_size: int
                maximal size of ensemble (passed to ensemble_selection)
            ensemble_nbest: int/float
                if int: consider only the n best prediction
                if float: consider only this fraction of the best models
                Both wrt to validation predictions
                If performance_range_threshold > 0, might return less models
            max_models_on_disc: Union[float, int]
                Defines the maximum number of models that are kept in the disc.
                If int, it must be greater or equal than 1, and dictates the max number of
                models to keep.
                If float, it will be interpreted as the max megabytes allowed of disc space. That
                is, if the number of ensemble candidates require more disc space than this float
                value, the worst models will be deleted to keep within this budget.
                Models and predictions of the worst-performing models will be deleted then.
                If None, the feature is disabled.
                It defines an upper bound on the models that can be used in the ensemble.
            seed: int
                random seed
            max_iterations: int
                maximal number of iterations to run this script
                (default None --> deactivated)
            precision (int): [16,32,64,128]
                precision of floats to read the predictions
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
            read_at_most: int
                read at most n new prediction files in each iteration
            logger_port: int
                port in where to publish a msg
            pynisher_context: str
                The multiprocessing context for pynisher. One of spawn/fork/forkserver.

        Returns:
            List[Tuple[int, float, float, float]]:
                A list with the performance history of this ensemble, of the form
                [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
        """
        self.start_time = start_time
        self.time_left_for_ensembles = time_left_for_ensembles
        self.backend = backend
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.output_type = output_type
        self.metrics = metrics
        self.opt_metric = opt_metric
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.base_ensemble_method = base_ensemble_method
        self.stacking_ensemble_method = stacking_ensemble_method
        self.cur_stacking_layer = 0 if is_stacking(base_ensemble_method, stacking_ensemble_method) else None
        if (
            is_stacking(base_ensemble_method, stacking_ensemble_method)
            and num_stacking_layers is None
        ):
            raise ValueError("Cant be none for stacked ensembles")

        self.ensemble_slot_j = ensemble_slot_j
        self.num_stacking_layers = num_stacking_layers
        self.max_models_on_disc: Union[float, int] = max_models_on_disc
        self.seed = seed
        self.precision = precision
        self.max_iterations = max_iterations
        self.read_at_most = read_at_most
        self.ensemble_memory_limit = ensemble_memory_limit
        self.random_state = random_state
        self.logger_port = logger_port
        self.pynisher_context = pynisher_context
        self.run_history = run_history

        self.is_new_layer = False
        # Store something similar to SMAC's runhistory
        self.history: List[Dict[str, float]] = []

        # We only submit new ensembles when there is not an active ensemble job
        self.futures: List[dask.Future] = []

        # The last criteria is the number of iterations
        self.iteration = iteration

        # Keep track of when we started to know when we need to finish!
        self.start_time = time.time()

        self.use_ensemble_loss = use_ensemble_loss
        self.initial_num_run = initial_num_run
        self.ensemble_history_tmp_file = os.path.join(self.backend.internals_directory, 'temp_ensemble_history.pkl')
        if os.path.exists(self.ensemble_history_tmp_file):
            self.history = pickle.load(open(self.ensemble_history_tmp_file, 'rb'))

    def __call__(
        self,
        smbo: 'SMBO',
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> None:
        self.build_ensemble(smbo.tae_runner.client)

    def build_ensemble(
        self,
        dask_client: dask.distributed.Client,
        unit_test: bool = False
    ) -> None:

        # The second criteria is elapsed time
        elapsed_time = time.time() - self.start_time

        logger = get_named_client_logger(
            name='EnsembleBuilderManager',
            port=self.logger_port,
        )

        logger.debug(f"In EnsembleBuilderManager iteration: {self.iteration}")
        # First test for termination conditions
        if self.time_left_for_ensembles < elapsed_time:
            logger.info(
                "Terminate ensemble building as not time is left (run for {}s)".format(
                    elapsed_time
                ),
            )
            return
        if self.max_iterations is not None and self.max_iterations <= self.iteration:
            logger.info(
                "Terminate ensemble building because of max iterations: {} of {}".format(
                    self.max_iterations,
                    self.iteration
                )
            )
            return

        if len(self.futures) != 0:
            if self.futures[0].done():
                self.extend_history(elapsed_time, logger)

        # Only submit new jobs if the previous ensemble job finished
        if len(self.futures) == 0:

            # Add the result of the run
            # On the next while iteration, no references to
            # ensemble builder object, so it should be garbage collected to
            # save memory while waiting for resources
            # Also, notice how ensemble nbest is returned, so we don't waste
            # iterations testing if the deterministic predictions size can
            # be fitted in memory
            try:
                # Submit a Dask job from this job, to properly
                # see it in the dask diagnostic dashboard
                # Notice that the forked ensemble_builder_process will
                # wait for the below function to be done
                self.futures.append(dask_client.submit(
                    fit_and_return_ensemble,
                    backend=self.backend,
                    dataset_name=self.dataset_name,
                    task_type=self.task_type,
                    output_type=self.output_type,
                    metrics=self.metrics,
                    opt_metric=self.opt_metric,
                    ensemble_size=self.ensemble_size,
                    ensemble_nbest=self.ensemble_nbest,
                    base_ensemble_method=self.base_ensemble_method,
                    stacking_ensemble_method=self.stacking_ensemble_method,
                    max_models_on_disc=self.max_models_on_disc,
                    seed=self.seed,
                    precision=self.precision,
                    memory_limit=self.ensemble_memory_limit,
                    read_at_most=self.read_at_most,
                    random_state=self.seed,
                    end_at=self.start_time + self.time_left_for_ensembles,
                    iteration=self.iteration,
                    return_predictions=False,
                    priority=100,
                    pynisher_context=self.pynisher_context,
                    logger_port=self.logger_port,
                    unit_test=unit_test,
                    use_ensemble_opt_loss=self.use_ensemble_loss,
                    cur_stacking_layer=self.cur_stacking_layer,
                    is_new_layer=self.is_new_layer,
                    num_stacking_layers=self.num_stacking_layers,
                    initial_num_run=self.initial_num_run,
                    ensemble_slot_j=self.ensemble_slot_j,
                    run_history=self.run_history
                ))

                logger.info(
                    f'{self.futures[0]}/{dask_client} Started Ensemble builder job at {time.strftime("%Y.%m.%d-%H.%M.%S")} for iteration {self.iteration} with time_left: {self.time_left_for_ensembles} with initial_num_run: {self.initial_num_run}.'
                )
                self.iteration += 1
                # reset to False so only signal from smbo sets is_new_layer = True
                self.is_new_layer = False
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                logger.critical(exception_traceback)
                logger.critical(error_message)
        if isinstance(self.futures[0], DummyFuture):
            self.extend_history(elapsed_time, logger)

        pickle.dump(self.history, open(self.ensemble_history_tmp_file, 'wb'))

    def extend_history(self, elapsed_time, logger):
        result = self.futures.pop().result()
        if result:
            ensemble_history, self.ensemble_nbest, _, _ = result
            logger.debug("iteration={} @ elapsed_time={} has history={}".format(
                        self.iteration,
                        elapsed_time,
                        ensemble_history,
                    ))
            self.history.extend(ensemble_history)

    def update_for_new_stacking_layer(self, cur_stacking_layer: int, initial_num_run: int, is_iterative_hpo=False) -> None:
        if cur_stacking_layer > self.num_stacking_layers:
            raise ValueError(f"Unexpected value '{cur_stacking_layer}' for cur_stacking_layer. "
                             f"Max stacking layers are : {self.num_stacking_layers}.")
        self.cur_stacking_layer = cur_stacking_layer
        self.initial_num_run = initial_num_run
        if not is_iterative_hpo:
            self.iteration = 0
            self.is_new_layer = True


def fit_and_return_ensemble(
    backend: Backend,
    dataset_name: str,
    task_type: int,
    output_type: int,
    metrics: List[autoPyTorchMetric],
    opt_metric: str,
    ensemble_size: int,
    ensemble_nbest: int,
    base_ensemble_method: BaseLayerEnsembleSelectionTypes,
    max_models_on_disc: Union[float, int],
    seed: int,
    precision: int,
    memory_limit: Optional[int],
    read_at_most: int,
    random_state: int,
    end_at: float,
    iteration: int,
    return_predictions: bool,
    pynisher_context: str,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    unit_test: bool = False,
    use_ensemble_opt_loss=False,
    stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None,
    cur_stacking_layer: int = 0,
    is_new_layer: bool = False,
    num_stacking_layers: Optional[int] = None,
    initial_num_run: int = 0,
    ensemble_slot_j: int = 0,
    run_history: Optional[RunHistory] = None,
) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
]:
    """
    A short function to fit and create an ensemble. It is just a wrapper to easily send
    a request to dask to create an ensemble and clean the memory when finished
    Parameters
    ----------
        backend: util.backend.Backend
            backend to write and read files
        dataset_name: str
            name of dataset
        metrics: List[autoPyTorchMetric],
            A set of metrics that will be used to get performance estimates
        opt_metric:
            Name of the metric to optimize
        task_type: int
            type of output expected in the ground truth
        ensemble_size: int
            maximal size of ensemble (passed to ensemble.ensemble_selection)
        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models
        max_models_on_disc: int
           Defines the maximum number of models that are kept in the disc.
           If int, it must be greater or equal than 1, and dictates the max number of
           models to keep.
           If float, it will be interpreted as the max megabytes allowed of disc space. That
           is, if the number of ensemble candidates require more disc space than this float
           value, the worst models will be deleted to keep within this budget.
           Models and predictions of the worst-performing models will be deleted then.
           If None, the feature is disabled.
           It defines an upper bound on the models that can be used in the ensemble.
        seed: int
            random seed
        precision (int): [16,32,64,128]
            precision of floats to read the predictions
        memory_limit: Optional[int]
            memory limit in mb. If ``None``, no memory limit is enforced.
        read_at_most: int
            read at most n new prediction files in each iteration
        end_at: float
            At what time the job must finish. Needs to be the endtime and not the time left
            because we do not know when dask schedules the job.
        iteration: int
            The current iteration
        pynisher_context: str
            Context to use for multiprocessing, can be either fork, spawn or forkserver.
        logger_port: int
            The port where the logging server is listening to.
        unit_test: bool
            Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
            Having this is very bad coding style, but I did not find a way to make
            unittest.mock work through the pynisher with all spawn contexts. If you know a
            better solution, please let us know by opening an issue.
    Returns
    -------
        List[Tuple[int, float, float, float]]
            A list with the performance history of this ensemble, of the form
            [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
    """
    ensemble_builder = get_ensemble_builder_class(base_ensemble_method, stacking_ensemble_method=stacking_ensemble_method)
    ensemble_builder_run_kwargs = {
        'end_at': end_at,
        'iteration': iteration,
        'return_predictions': return_predictions,
        'pynisher_context': pynisher_context,
        'cur_stacking_layer': cur_stacking_layer} 

    if stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_ensemble_selection_per_layer:
        ensemble_builder_run_kwargs.update({'is_new_layer': is_new_layer})

    if base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_iterative_hpo:
        ensemble_builder_run_kwargs.update({'ensemble_slot_j': ensemble_slot_j})

    result = ensemble_builder(
        backend=backend,
        dataset_name=dataset_name,
        task_type=task_type,
        output_type=output_type,
        metrics=metrics,
        opt_metric=opt_metric,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
        seed=seed,
        precision=precision,
        memory_limit=memory_limit,
        read_at_most=read_at_most,
        random_state=random_state,
        logger_port=logger_port,
        unit_test=unit_test,
        use_ensemble_opt_loss=use_ensemble_opt_loss,
        num_stacking_layers=num_stacking_layers,
        initial_num_run=initial_num_run,
        run_history=run_history
    ).run(
        **ensemble_builder_run_kwargs
    )
    return result
