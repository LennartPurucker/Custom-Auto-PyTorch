import time
import math
from typing import Any, Dict, List, Tuple, Union
import unittest

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import dask.distributed

from smac.runhistory.runhistory import DataOrigin, RunHistory, RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType

from autoPyTorch.evaluation.tae import ExecuteTaFuncWithQueue, get_cost_of_crash
from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.utils.configurations import is_configuration_traditional
from autoPyTorch.utils.common import dict_repr


def run_models_on_dataset(
    time_left: int,
    func_eval_time_limit_secs: int,
    model_configs: List[Tuple[Union[str, Configuration]]],
    logger,
    logger_port,
    metric,
    dask_client: dask.distributed.Client,
    backend: Backend,
    memory_limit: int,
    disable_file_output,
    all_supported_metrics: bool,
    base_ensemble_method,
    include,
    exclude,
    search_space_updates,
    pipeline_options,
    seed: int,
    multiprocessing_context,
    n_jobs: int,
    current_search_space: ConfigurationSpace,
    smac_initial_run: int
) -> Tuple[RunHistory, List[Tuple]]:
    starttime = time.time()
    run_history = RunHistory()
    memory_limit = memory_limit
    if memory_limit is not None:
        memory_limit = int(math.ceil(memory_limit))
    model_identifiers = []
    total_models = len(model_configs)
    dask_futures = []
    for n_r, (config, budget) in enumerate(model_configs):

        # Only launch a task if there is time
        start_time = time.time()
        if time_left >= func_eval_time_limit_secs:
            logger.info(f"{n_r}: Started fitting {config} with cutoff={func_eval_time_limit_secs}")
            scenario_mock = unittest.mock.Mock()
            scenario_mock.wallclock_limit = time_left
            # This stats object is a hack - maybe the SMAC stats object should
            # already be generated here!
            stats = Stats(scenario_mock)
            stats.start_timing()

            if isinstance(config, Configuration) and not is_configuration_traditional(config):
                config.config_id = n_r
                init_num_run = smac_initial_run
            else:
                init_num_run = smac_initial_run + n_r

            ta = ExecuteTaFuncWithQueue(
                pynisher_context=multiprocessing_context,
                backend=backend,
                seed=seed,
                metric=metric,
                multi_objectives=["cost"],
                logger_port=logger_port,
                pipeline_config=pipeline_options,
                cost_for_crash=get_cost_of_crash(metric),
                abort_on_first_run_crash=False,
                initial_num_run=init_num_run,
                stats=stats,
                memory_limit=memory_limit,
                disable_file_output=disable_file_output,
                all_supported_metrics=all_supported_metrics,
                base_ensemble_method=base_ensemble_method,
                include=include,
                exclude=exclude,
                search_space_updates=search_space_updates
            )
            dask_futures.append([
                config,
                dask_client.submit(
                    ta.run, config=config,
                    cutoff=func_eval_time_limit_secs,
                    budget=budget
                )
            ])

        # When managing time, we need to take into account the allocated time resources,
        # which are dependent on the number of cores. 'dask_futures' is a proxy to the number
        # of workers /n_jobs that we have, in that if there are 4 cores allocated, we can run at most
        # 4 task in parallel. Every 'cutoff' seconds, we generate up to 4 tasks.
        # If we only have 4 workers and there are 4 futures in dask_futures, it means that every
        # worker has a task. We would not like to launch another job until a worker is available. To this
        # end, the following if-statement queries the number of active jobs, and forces to wait for a job
        # completion via future.result(), so that a new worker is available for the next iteration.
        if len(dask_futures) >= n_jobs:

            # How many workers to wait before starting fitting the next iteration
            workers_to_wait = 1
            if n_r >= total_models - 1 or time_left <= func_eval_time_limit_secs:
                # If on the last iteration, flush out all tasks
                workers_to_wait = len(dask_futures)

            while workers_to_wait >= 1:
                workers_to_wait -= 1
                # We launch dask jobs only when there are resources available.
                # This allow us to control time allocation properly, and early terminate
                # the traditional machine learning pipeline
                cls, future = dask_futures.pop(0)
                status, cost, runtime, additional_info = future.result()

                if status == StatusType.SUCCESS:
                    logger.info(
                        "Fitting {} took {} [sec] and got performance: {}.\n"
                        "additional info:\n{}".format(cls, runtime, cost, dict_repr(additional_info))
                    ) 
                    origin = additional_info['configuration_origin']
                    config = additional_info['configuration']
                    budget = additional_info['budget']
                    if isinstance(config, dict) and not is_configuration_traditional(config):
                        configuration = Configuration(current_search_space, config)
                    else:
                        configuration = additional_info.pop('pipeline_configuration')

                    # additional_info.pop('pipeline_configuration')
                    run_history.add(config=configuration, cost=cost,
                                    time=runtime, status=status, seed=seed,
                                    starttime=starttime, endtime=starttime + runtime,
                                    origin=origin, additional_info=additional_info)
                    model_identifiers.append((seed, additional_info['num_run'], float(budget)))
                else:
                    if additional_info.get('exitcode') == -6:
                        logger.error(
                            "Traditional prediction for {} failed with run state {},\n"
                            "because the provided memory limits were too tight.\n"
                            "Please increase the 'ml_memory_limit' and try again.\n"
                            "If you still get the problem, please open an issue\n"
                            "and paste the additional info.\n"
                            "Additional info:\n{}".format(cls, str(status), dict_repr(additional_info))
                        )
                    else:
                        logger.error(
                            "Traditional prediction for {} failed with run state {}.\nAdditional info:\n{}".format(
                                cls, str(status), dict_repr(additional_info)
                            )
                        )
                    model_identifiers.append(None)
        # In the case of a serial execution, calling submit halts the run for a resource
        # dynamically adjust time in this case
        time_left -= int(time.time() - start_time)

        # Exit if no more time is available for a new classifier
        if time_left < func_eval_time_limit_secs:
            logger.warning("Not enough time to fit all machine learning models."
                           "Please consider increasing the run time to further improve performance.")
            break
    return run_history, model_identifiers
