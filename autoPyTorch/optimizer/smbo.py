import copy
import json
import logging.handlers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

import ConfigSpace
from ConfigSpace.configuration_space import Configuration

import dask.distributed

from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.hyperband import Hyperband
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.serial_runner import SerialRunner
from smac.utils.io.traj_logging import TrajEntry

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import (
    ResamplingStrategies,
    DEFAULT_RESAMPLING_PARAMETERS,
    HoldoutValTypes,
    CrossValTypes
)
from autoPyTorch.datasets.utils import get_appended_dataset
from autoPyTorch.ensemble.ensemble_builder_manager import EnsembleBuilderManager
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble import EnsembleOptimisationStackingEnsemble
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble import EnsembleSelectionPerLayerStackingEnsemble
from autoPyTorch.ensemble.ensemble_selection_types import BaseLayerEnsembleSelectionTypes, StackingEnsembleSelectionTypes, is_stacking
from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.optimizer.utils import read_return_initial_configurations
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.common import delete_runs_except_ensemble
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.stopwatch import StopWatch


def get_smac_object(
    scenario_dict: Dict[str, Any],
    seed: int,
    ta: Callable,
    ta_kwargs: Dict[str, Any],
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client: Optional[dask.distributed.Client],
    smbo_class: Optional[SMBO] = None,
    initial_configurations: Optional[List[Configuration]] = None,
) -> SMAC4AC:
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines

    Args:
        scenario_dict (Dict[str, Any]): constrain on how to run
            the jobs
        seed (int): to make the job deterministic
        ta (Callable): the function to be intensifier by smac
        ta_kwargs (Dict[str, Any]): Arguments to the above ta
        n_jobs (int): Amount of cores to use for this task
        dask_client (dask.distributed.Client): User provided scheduler
        initial_configurations (List[Configuration]): List of initial
            configurations which smac will run before starting the search process

    Returns:
        (SMAC4AC): sequential model algorithm configuration object

    """
    intensifier = Hyperband

    rh2EPM = RunHistory2EPM4LogCost
    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=initial_configurations,
        run_id=seed,
        intensifier=intensifier,
        intensifier_kwargs={'initial_budget': initial_budget, 'max_budget': max_budget,
                            'eta': 2, 'min_chall': 1, 'instance_order': 'shuffle_once'},
        dask_client=dask_client,
        n_jobs=n_jobs,
        smbo_class=smbo_class
    )


class AutoMLSMBO(object):

    def __init__(self,
                 config_space: ConfigSpace.ConfigurationSpace,
                 dataset_name: str,
                 backend: Backend,
                 total_walltime_limit: float,
                 func_eval_time_limit_secs: float,
                 memory_limit: Optional[int],
                 metric: autoPyTorchMetric,
                 watcher: StopWatch,
                 n_jobs: int,
                 dask_client: Optional[dask.distributed.Client],
                 pipeline_config: Dict[str, Any],
                 start_num_run: int = 1,
                 seed: int = 1,
                 resampling_strategy: ResamplingStrategies = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: Union[bool, List[str]] = False,
                 smac_scenario_args: Optional[Dict[str, Any]] = None,
                 get_smac_object_callback: Optional[Callable] = None,
                 all_supported_metrics: bool = True,
                 ensemble_callback: Optional[EnsembleBuilderManager] = None,
                 num_stacking_layers: Optional[int] = None,
                 logger_port: Optional[int] = None,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 portfolio_selection: Optional[str] = None,
                 pynisher_context: str = 'spawn',
                 min_budget: int = 5,
                 max_budget: int = 50,
                 base_ensemble_method: BaseLayerEnsembleSelectionTypes = BaseLayerEnsembleSelectionTypes.ensemble_selection,
                 stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None,
                 other_callbacks: Optional[List] = None,
                 smbo_class: Optional[SMBO] = None,
                 use_ensemble_opt_loss: bool = False,
                 iteration: int = 0,
                 **kwargs
                 ):
        """
        Interface to SMAC. This method calls the SMAC optimize method, and allows
        to pass a callback (ensemble_callback) to make launch task at the end of each
        optimize() algorithm. The later is needed due to the nature of blocking long running
        tasks in Dask.

        Args:
            config_space (ConfigSpace.ConfigurationSpac):
                The configuration space of the whole process
            dataset_name (str):
                The name of the dataset, used to identify the current job
            backend (Backend):
                An interface with disk
            total_walltime_limit (float):
                The maximum allowed time for this job
            func_eval_time_limit_secs (float):
                How much each individual task is allowed to last
            memory_limit (Optional[int]):
                Maximum allowed CPU memory this task can use
            metric (autoPyTorchMetric):
                An scorer object to evaluate the performance of each jon
            watcher (StopWatch):
                A stopwatch object to debug time consumption
            n_jobs (int):
                How many workers are allowed in each task
            dask_client (Optional[dask.distributed.Client]):
                An user provided scheduler. Else smac will create its own.
            start_num_run (int):
                The ID index to start runs
            seed (int):
                To make the run deterministic
            resampling_strategy (str):
                What strategy to use for performance validation
            resampling_strategy_args (Optional[Dict[str, Any]]):
                Arguments to the resampling strategy -- like number of folds
            include (Optional[Dict[str, Any]] = None):
                Optimal Configuration space modifiers
            exclude (Optional[Dict[str, Any]] = None):
                Optimal Configuration space modifiers
            disable_file_output List:
                Support to disable file output to disk -- to reduce space
            smac_scenario_args (Optional[Dict[str, Any]]):
                Additional arguments to the smac scenario
            get_smac_object_callback (Optional[Callable]):
                Allows to create a user specified SMAC object
            pynisher_context (str):
                A string indicating the multiprocessing context to use
            ensemble_callback (Optional[EnsembleBuilderManager]):
                A callback used in this scenario to start ensemble building subtasks
            portfolio_selection (Optional[str]):
                This argument controls the initial configurations that
                AutoPyTorch uses to warm start SMAC for hyperparameter
                optimization. By default, no warm-starting happens.
                The user can provide a path to a json file containing
                configurations, similar to (autoPyTorch/configs/greedy_portfolio.json).
                Additionally, the keyword 'greedy' is supported,
                which would use the default portfolio from
                `AutoPyTorch Tabular <https://arxiv.org/abs/2006.13799>_`
            min_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>_` to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                min_budget states the minimum resource allocation a pipeline should have
                so that we can compare and quickly discard bad performing models.
                For example, if the budget_type is epochs, and min_budget=5, then we will
                run every pipeline to a minimum of 5 epochs before performance comparison.
            max_budget (int):
                Auto-PyTorch uses `Hyperband <https://arxiv.org/abs/1603.06560>_` to
                trade-off resources between running many pipelines at min_budget and
                running the top performing pipelines on max_budget.
                max_budget states the maximum resource allocation a pipeline is going to
                be ran. For example, if the budget_type is epochs, and max_budget=50,
                then the pipeline training will be terminated after 50 epochs.
        """
        super(AutoMLSMBO, self).__init__()
        # data related
        self.datamanager: Optional[BaseDataset] = None
        self.dataset_name = dataset_name
        self.metric = metric

        self.backend = backend
        self.all_supported_metrics = all_supported_metrics

        self.pipeline_config = pipeline_config
        # the configuration space
        self.config_space = config_space

        # the number of parallel workers/jobs
        self.n_jobs = n_jobs
        self.dask_client = dask_client

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = DEFAULT_RESAMPLING_PARAMETERS[resampling_strategy]
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metric)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit_secs = int(func_eval_time_limit_secs)
        self.memory_limit = memory_limit
        self.watcher = watcher
        self.seed = seed
        self.start_num_run = start_num_run
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback
        self.pynisher_context = pynisher_context
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.base_ensemble_method = base_ensemble_method

        self.ensemble_callback = ensemble_callback
        self.stacking_ensemble_method = stacking_ensemble_method
        if is_stacking(base_ensemble_method, stacking_ensemble_method) and num_stacking_layers is None:
            raise ValueError("'num_stacking_layers' can't be none for stacked ensembles")

        self.num_stacking_layers = num_stacking_layers

        self.iteration = iteration
        self.run_history = RunHistory()
        self.trajectory: List[TrajEntry] = []

        self.other_callbacks = other_callbacks
        self.smbo_class = smbo_class

        self.search_space_updates = search_space_updates

        if logger_port is None:
            self.logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        else:
            self.logger_port = logger_port
        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed, ":" + self.dataset_name)
        self.logger = get_named_client_logger(name=logger_name,
                                              port=self.logger_port)
        self.logger.info("initialised {}".format(self.__class__.__name__))

        self.use_ensemble_opt_loss = use_ensemble_opt_loss

        self.initial_configurations: Optional[List[Configuration]] = None
        if portfolio_selection is not None:
            self.initial_configurations = read_return_initial_configurations(config_space=config_space,
                                                                             portfolio_selection=portfolio_selection)
            if len(self.initial_configurations) == 0:
                self.initial_configurations = None
                self.logger.warning("None of the portfolio configurations are compatible"
                                    " with the current search space. Skipping initial configuration...")
        self.special_kwargs = kwargs if self.stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_fine_tuning else {}

    def reset_data_manager(self) -> None:
        if self.datamanager is not None:
            del self.datamanager
        self.datamanager = self.backend.load_datamanager()
        if self.datamanager is not None and self.datamanager.task_type is not None:
            self.task = self.datamanager.task_type

    def reset_attributes(self, datamanager: BaseDataset) -> None:
        self.backend.save_datamanager(datamanager=datamanager)

        dataset_requirements = get_dataset_requirements(
            info=datamanager.get_required_dataset_info(),
            include=self.include,
            exclude=self.exclude,
            search_space_updates=self.search_space_updates)
        self._dataset_requirements = dataset_requirements
        dataset_properties = datamanager.get_dataset_properties(dataset_requirements)
        self.config_space = get_configuration_space(dataset_properties, include=self.include, exclude=self.exclude, search_space_updates=self.search_space_updates)

    def _run_smbo(
        self,
        cur_stacking_layer: int,
        walltime_limit: int,
        initial_num_run: int,
        func: Optional[Callable] = None,
        ) -> Tuple[RunHistory, List[TrajEntry], str]:

        current_task_name = f'SMBO_{cur_stacking_layer}_{self.iteration}'

        self.watcher.start_task(current_task_name)
        self.logger.info(f"Started layer: {cur_stacking_layer} run of SMBO with initial_num_run: {initial_num_run}")

        # # == first things first: load the datamanager
        # self.reset_data_manager()

        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        # allocate a run history

        # Initialize some SMAC dependencies

        if isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = self.resampling_strategy_args['num_splits']
            instances = [[json.dumps({'task_id': self.dataset_name,
                                      'fold': fold_number})]
                         for fold_number in range(num_splits)]
        else:
            instances = [[json.dumps({'task_id': self.dataset_name})]]

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over
        ta_kwargs = dict(
            backend=copy.deepcopy(self.backend),
            seed=seed,
            initial_num_run=initial_num_run,
            include=self.include if self.include is not None else dict(),
            exclude=self.exclude if self.exclude is not None else dict(),
            metric=self.metric,
            memory_limit=self.memory_limit,
            disable_file_output=self.disable_file_output,
            ta=func,
            logger_port=self.logger_port,
            all_supported_metrics=self.all_supported_metrics,
            pipeline_config=self.pipeline_config,
            search_space_updates=self.search_space_updates,
            pynisher_context=self.pynisher_context,
            base_ensemble_method=self.base_ensemble_method,
            stacking_ensemble_method=self.stacking_ensemble_method,
            use_ensemble_opt_loss=self.use_ensemble_opt_loss,
        )

        ta_kwargs = {**ta_kwargs, **self.special_kwargs}
        ta = ExecuteTaFuncWithQueue
        self.logger.info(f"Finish creating Target Algorithm (TA) function with ta_kwargs: {ta_kwargs}")

        startup_time = self.watcher.wall_elapsed(current_task_name)
        walltime_limit = walltime_limit - startup_time - 5
        scenario_dict = {
            'abort_on_first_run_crash': False,
            'cs': self.config_space,
            'cutoff_time': self.func_eval_time_limit_secs,
            'deterministic': 'true',
            'instances': instances,
            'memory_limit': self.memory_limit,
            'output-dir': self.backend.get_smac_output_directory(),
            'run_obj': 'quality',
            'wallclock_limit': walltime_limit,
            'cost_for_crash': self.worst_possible_result,
        }
        if self.smac_scenario_args is not None:
            for arg in [
                'abort_on_first_run_crash',
                'cs',
                'deterministic',
                'instances',
                'output-dir',
                'run_obj',
                'shared-model',
                'cost_for_crash',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning('Cannot override scenario argument %s, '
                                        'will ignore this.', arg)
                    del self.smac_scenario_args[arg]
            for arg in [
                'cutoff_time',
                'memory_limit',
                'wallclock_limit',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        'Overriding scenario argument %s: %s with value %s',
                        arg,
                        scenario_dict[arg],
                        self.smac_scenario_args[arg]
                    )
            scenario_dict.update(self.smac_scenario_args)

        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(scenario_dict=scenario_dict,
                                                 seed=seed,
                                                 ta=ta,
                                                 ta_kwargs=ta_kwargs,
                                                 n_jobs=self.n_jobs,
                                                 initial_budget=self.min_budget,
                                                 max_budget=self.max_budget,
                                                 dask_client=self.dask_client,
                                                 initial_configurations=self.initial_configurations,
                                                 smbo_class=self.smbo_class)
        else:
            smac = get_smac_object(scenario_dict=scenario_dict,
                                   seed=seed,
                                   ta=ta,
                                   ta_kwargs=ta_kwargs,
                                   n_jobs=self.n_jobs,
                                   initial_budget=self.min_budget,
                                   max_budget=self.max_budget,
                                   dask_client=self.dask_client,
                                   initial_configurations=self.initial_configurations,
                                   smbo_class=self.smbo_class)

        if self.ensemble_callback is not None:
            if self.stacking_ensemble_method is not None and cur_stacking_layer > 0:
                self.logger.debug(f"Hey, I m here, updating the initial_num_run to {initial_num_run}")
                self.ensemble_callback.update_for_new_stacking_layer(cur_stacking_layer, initial_num_run)
            smac.register_callback(self.ensemble_callback)
        self.logger.debug(f"initial_num_run in {self.__class__.__name__}: {initial_num_run}")
        if self.other_callbacks is not None:
            for callback in self.other_callbacks:
                smac.register_callback(callback)

        self.logger.info("initialised SMBO, running SMBO.optimize()")

        smac.optimize()

        self.logger.info("finished SMBO.optimize()")

        runhistory = smac.solver.runhistory
        trajectory = smac.solver.intensifier.traj_logger.trajectory
        if isinstance(smac.solver.tae_runner, DaskParallelRunner):
            self._budget_type = smac.solver.tae_runner.single_worker.budget_type
        elif isinstance(smac.solver.tae_runner, SerialRunner):
            self._budget_type = smac.solver.tae_runner.budget_type
        else:
            raise NotImplementedError(type(smac.solver.tae_runner))

        self.watcher.stop_task(current_task_name)

        return runhistory, trajectory, self._budget_type

    def run_smbo(self, func: Optional[Callable] = None
                 ) -> Tuple[RunHistory, List[TrajEntry], str]:
        individual_wall_times = self.total_walltime_limit / self.num_stacking_layers
        initial_num_run = self.start_num_run
        self.reset_data_manager()
        for cur_stacking_layer in range(self.num_stacking_layers):
            if cur_stacking_layer == 0:
                self.logger.debug(f"Initial feat_types = {self.datamanager.feat_types}, special_kwargs: {self.special_kwargs}")
            run_history, trajectory, _ = self._run_smbo(
                walltime_limit=individual_wall_times,
                cur_stacking_layer=cur_stacking_layer,
                initial_num_run=initial_num_run,
                func=func
                )
            self.run_history.update(run_history, origin=DataOrigin.INTERNAL)
            self.trajectory.extend(trajectory)
            if self.num_stacking_layers <= 1:
                break 
            old_ensemble: Optional[Union[EnsembleSelectionPerLayerStackingEnsemble, EnsembleOptimisationStackingEnsemble]] = None
            ensemble_dir = self.backend.get_ensemble_dir()
            if os.path.exists(ensemble_dir) and len(os.listdir(ensemble_dir)) >= 1:
                old_ensemble = self.backend.load_ensemble(self.seed)
                assert isinstance(old_ensemble, (EnsembleOptimisationStackingEnsemble, EnsembleSelectionPerLayerStackingEnsemble))
                if cur_stacking_layer != self.num_stacking_layers -1:
                    delete_runs_except_ensemble(old_ensemble, self.backend)
            previous_layer_predictions_train = old_ensemble.get_layer_stacking_ensemble_predictions(stacking_layer=cur_stacking_layer)
            previous_layer_predictions_test = old_ensemble.get_layer_stacking_ensemble_predictions(stacking_layer=cur_stacking_layer, dataset='test')
            self.logger.debug(f"Original feat types len: {len(self.datamanager.feat_types)}")
            nonnull_model_predictions_train = [pred for pred in previous_layer_predictions_train if pred is not None]
            nonnull_model_predictions_test = [pred for pred in previous_layer_predictions_test if pred is not None]
            assert len(nonnull_model_predictions_train) == len(nonnull_model_predictions_test)
            self.logger.debug(f"length Non nulll predictions: {len(nonnull_model_predictions_train)}")
            datamanager = get_appended_dataset(
                original_dataset=self.datamanager,
                previous_layer_predictions_train=nonnull_model_predictions_train,
                previous_layer_predictions_test=nonnull_model_predictions_test,
                resampling_strategy=self.resampling_strategy,
                resampling_strategy_args=self.resampling_strategy_args,
            )
            self.logger.debug(f"new feat_types len: {len(datamanager.feat_types)}")
            self.reset_attributes(datamanager=datamanager)

            initial_num_run = self.backend.get_next_num_run()
            self.logger.debug(f"cutoff num_run: {initial_num_run}")

        return self.run_history, self.trajectory, self._budget_type

