from collections import OrderedDict
import glob
import gzip
import logging
import logging.handlers
import multiprocessing
import numbers
import os
import pickle
import re
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

import pandas as pd

import pynisher

from sklearn.utils.validation import check_random_state

from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import BINARY
from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.iterative_hpo_stacking_ensemble import IterativeHPOStackingEnsemble
from autoPyTorch.ensemble.utils import get_identifiers_from_num_runs, get_num_runs_from_identifiers, save_stacking_ensemble
from autoPyTorch.utils.common import read_np_fn
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.utils.common import (
    ENSEMBLE_ITERATION_MULTIPLIER,
    get_ensemble_cutoff_num_run_filename,
    get_ensemble_identifiers_filename,
    get_ensemble_unique_identifier_filename
)
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.parallel import preload_modules

Y_ENSEMBLE = 0
Y_TEST = 1

MODEL_FN_RE = r'_([0-9]*)_([0-9]*)_([0-9]+\.*[0-9]*)\.npy'


class IterativeHPOStackingEnsembleBuilder(object):
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        output_type: int,
        metrics: List[autoPyTorchMetric],
        opt_metric: str,
        run_history: Optional[Union[RunHistory, OrderedDict]] = None,
        ensemble_size: int = 10,
        ensemble_nbest: int = 100,
        seed: int = 1,
        precision: int = 32,
        memory_limit: Optional[int] = 1024,
        read_at_most: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        unit_test: bool = False,
        initial_num_run: int = 0,
        num_stacking_layers: Optional[int] = None,
        use_ensemble_opt_loss = False,
        max_models_on_disc: Union[float, int] = 100,
        performance_range_threshold: float = 0,
    ):
        """
            Constructor
            Parameters
            ----------
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            task_type: int
                type of ML task
            metrics: List[autoPyTorchMetric],
                name of metric to score predictions
            opt_metric: str
                name of the metric to optimize
            ensemble_size: int
                maximal size of ensemble (passed to ensemble.ensemble_selection)
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
            performance_range_threshold: float
                Keep only models that are better than:
                    dummy + (best - dummy)*performance_range_threshold
                E.g dummy=2, best=4, thresh=0.5 --> only consider models with score > 3
                Will at most return the minimum between ensemble_nbest models,
                and max_models_on_disc. Might return less
            seed: int
                random seed
            precision: [16,32,64,128]
                precision of floats to read the predictions
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
            read_at_most: int
                read at most n new prediction files in each iteration
            logger_port: int
                port that receives logging records
            unit_test: bool
                Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
                Having this is very bad coding style, but I did not find a way to make
                unittest.mock work through the pynisher with all spawn contexts. If you know a
                better solution, please let us know by opening an issue.
        """

        super(IterativeHPOStackingEnsembleBuilder, self).__init__()

        self.backend = backend  # communication with filesystem
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.output_type = output_type
        self.metrics = metrics
        self.opt_metric = opt_metric
        self.ensemble_size = ensemble_size
        assert run_history is not None, f"run history cant be None for {self.__class__.__name__}"
        self.run_history = run_history

        self.initial_num_run = initial_num_run
        if isinstance(ensemble_nbest, numbers.Integral) and ensemble_nbest < 1:
            raise ValueError("Integer ensemble_nbest has to be larger 1: %s" %
                             ensemble_nbest)
        elif not isinstance(ensemble_nbest, numbers.Integral):
            if ensemble_nbest < 0 or ensemble_nbest > 1:
                raise ValueError(
                    "Float ensemble_nbest best has to be >= 0 and <= 1: %s" %
                    ensemble_nbest)

        self.ensemble_nbest = ensemble_nbest

        self.max_resident_models: Optional[int] = None

        self.seed = seed
        self.precision = precision
        self.memory_limit = memory_limit
        self.read_at_most = read_at_most
        self.random_state = check_random_state(random_state)
        self.unit_test = unit_test

        # Setup the logger
        self.logger_port = logger_port
        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        if ensemble_nbest == 1:
            self.logger.debug("Behaviour depends on int/float: %s, %s (ensemble_nbest, type)" %
                              (ensemble_nbest, type(ensemble_nbest)))

        self.start_time = 0.0
        self.model_fn_re = re.compile(MODEL_FN_RE)

        self.last_hash = None  # hash of ensemble training data
        self.y_true_ensemble = None
        self.SAVE2DISC = True

        # already read prediction files
        # We read in back this object to give the ensemble the possibility to have memory
        # Every ensemble task is sent to dask as a function, that cannot take un-picklable
        # objects as attributes. For this reason, we dump to disk the stage of the past
        # ensemble iterations to kick-start the ensembling process
        # {"file name": {
        #    "ens_loss": float
        #    "mtime_ens": str,
        #    "mtime_test": str,
        #    "seed": int,
        #    "num_run": int,
        # }}
        self.read_losses = {}
        # {"file_name": {
        #    Y_ENSEMBLE: np.ndarray
        #    Y_TEST: np.ndarray
        #    }
        # }
        self.read_preds = {}

        # Depending on the dataset dimensions,
        # regenerating every iteration, the predictions
        # losses for self.read_preds
        # is too computationally expensive
        # As the ensemble builder is stateless
        # (every time the ensemble builder gets resources
        # from dask, it builds this object from scratch)
        # we save the state of this dictionary to memory
        # and read it if available
        self.ensemble_memory_file = os.path.join(
            self.backend.internals_directory,
            'iterative_ensemble_read_preds.pkl'
        )
        if os.path.exists(self.ensemble_memory_file):
            try:
                with (open(self.ensemble_memory_file, "rb")) as memory:
                    self.read_preds = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    "Could not load the previous iterations of ensemble_builder predictions."
                    "This might impact the quality of the run. Exception={} {}".format(
                        e,
                        traceback.format_exc(),
                    )
                )

        self.ensemble_loss_file = os.path.join(
            self.backend.internals_directory,
            'iterative_ensemble_read_losses.pkl'
        )
        if os.path.exists(self.ensemble_loss_file):
            try:
                with (open(self.ensemble_loss_file, "rb")) as memory:
                    self.read_losses = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    "Could not load the previous iterations of ensemble_builder losses."
                    "This might impact the quality of the run. Exception={} {}".format(
                        e,
                        traceback.format_exc(),
                    )
                )

        # hidden feature which can be activated via an environment variable. This keeps all
        # models and predictions which have ever been a candidate. This is necessary to post-hoc
        # compute the whole ensemble building trajectory.
        self._has_been_candidate: Set[str] = set()

        self.validation_performance_ = np.inf

        # Track the ensemble performance
        self.y_test = None
        datamanager = self.backend.load_datamanager()
        if datamanager.test_tensors is not None:
            self.y_test = datamanager.test_tensors[1]
        del datamanager
        self.ensemble_history: List[Dict[str, float]] = []
        self.use_ensemble_opt_loss = use_ensemble_opt_loss
        self.num_stacking_layers = num_stacking_layers

    def run(
        self,
        iteration: int,
        pynisher_context: str,
        time_left: Optional[float] = None,
        end_at: Optional[float] = None,
        time_buffer: int = 5,
        return_predictions: bool = False,
        cur_stacking_layer: int = 0,
        ensemble_slot_j: int = 0
    ) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This function is an interface to the main process and fundamentally calls main(), the
        later has the actual ensemble selection logic.

        The motivation towards this run() method is that it can be seen as a wrapper over the
        whole ensemble_builder.main() process so that pynisher can manage the memory/time limits.

        This is handy because this function reduces the number of members of the ensemble in case
        we run into memory issues. It does so in a halving fashion.

        Args:
            time_left (float):
                How much time is left for the ensemble builder process
            iteration (int):
                Which is the current iteration
            return_predictions (bool):
                Whether we want to return the predictions of the current model or not

        Returns:
            ensemble_history (Dict):
                A snapshot of both test and optimization performance. For debugging.
            ensemble_nbest (int):
                The user provides a direction on how many models to use in ensemble selection.
                This number can be reduced internally if the memory requirements force it.
            train_predictions (np.ndarray):
                The optimization prediction from the current ensemble.
            test_predictions (np.ndarray):
                The train prediction from the current ensemble.
        """
        self.cur_stacking_layer = cur_stacking_layer
        self.ensemble_slot_j = ensemble_slot_j

        if time_left is None and end_at is None:
            raise ValueError('Must provide either time_left or end_at.')
        elif time_left is not None and end_at is not None:
            raise ValueError('Cannot provide both time_left and end_at.')

        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )
        self.logger.debug(f"Starting ensemble building main job with {time_left}, {end_at}")

        process_start_time = time.time()
        while True:

            if time_left is not None:
                time_elapsed = time.time() - process_start_time
                time_left -= time_elapsed
            elif end_at is not None:
                current_time = time.time()
                if current_time > end_at:
                    break
                else:
                    time_left = end_at - current_time
            else:
                raise NotImplementedError()

            wall_time_in_s = int(time_left - time_buffer)
            if wall_time_in_s < 1:
                break
            context = multiprocessing.get_context(pynisher_context)
            preload_modules(context)

            safe_ensemble_script = pynisher.enforce_limits(
                wall_time_in_s=wall_time_in_s,
                mem_in_mb=self.memory_limit,
                logger=self.logger,
                context=context,
            )(self.main)
            safe_ensemble_script(time_left, iteration, return_predictions)
            if safe_ensemble_script.exit_status is pynisher.MemorylimitException:
                # if ensemble script died because of memory error,
                # reduce nbest to reduce memory consumption and try it again

                if isinstance(self.ensemble_nbest, numbers.Integral) and self.ensemble_nbest <= 1:
                    if self.read_at_most == 1:
                        self.logger.error(
                            "Memory Exception -- Unable to further reduce the number of ensemble "
                            "members and can no further limit the number of ensemble members "
                            "loaded per iteration -- please restart autoPytorch with a higher "
                            "value for the argument `memory_limit` (current limit is %s MB). "
                            "The ensemble builder will keep running to delete files from disk in "
                            "case this was enabled.", self.memory_limit
                        )
                        self.ensemble_nbest = 0
                    else:
                        self.read_at_most = 1
                        self.logger.warning(
                            "Memory Exception -- Unable to further reduce the number of ensemble "
                            "members -- Now reducing the number of predictions per call to read "
                            "at most to 1."
                        )
                else:
                    if isinstance(self.ensemble_nbest, numbers.Integral):
                        self.ensemble_nbest = max(1, int(self.ensemble_nbest / 2))
                    else:
                        self.ensemble_nbest = int(self.ensemble_nbest / 2)
                    self.logger.warning("Memory Exception -- restart with "
                                        "less ensemble_nbest: %d" % self.ensemble_nbest)
                    return [], self.ensemble_nbest, None, None
            else:
                return safe_ensemble_script.result

        return [], self.ensemble_nbest, None, None

    def main(
        self, time_left: float, iteration: int, return_predictions: bool,
    ) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This is the main function of the ensemble builder process and can be considered
        a wrapper over the ensemble selection method implemented y EnsembleSelection class.

        This method is going to be called multiple times by the main process, to
        build and ensemble, in case the SMAC process produced new models and to provide
        anytime results.

        On this regard, this method mainly:
            1- select from all the individual models that smac created, the N-best candidates
               (this in the scenario that N > ensemble_nbest argument to this class). This is
               done based on a score calculated via the metrics argument.
            2- This pre-selected candidates are provided to the ensemble selection method
               and if a ensemble is found under the provided memory/time constraints, a new
               ensemble is proposed.
            3- Because this process will be called multiple times, it performs checks to make
               sure a new ensenmble is only proposed if new predictions are available, as well
               as making sure we do not run out of resources (like disk space)

        Args:
            time_left (float):
                How much time is left for the ensemble builder process
            iteration (int):
                Which is the current iteration
            return_predictions (bool):
                Whether we want to return the predictions of the current model or not

        Returns:
            ensemble_history (Dict):
                A snapshot of both test and optimization performance. For debugging.
            ensemble_nbest (int):
                The user provides a direction on how many models to use in ensemble selection.
                This number can be reduced internally if the memory requirements force it.
            train_predictions (np.ndarray):
                The optimization prediction from the current ensemble.
            test_predictions (np.ndarray):
                The train prediction from the current ensemble.
        """

        # Pynisher jobs inside dask 'forget'
        # the logger configuration. So we have to set it up
        # accordingly
        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        self.start_time = time.time()
        train_pred, test_pred = None, None

        used_time = time.time()  - self.start_time
        self.logger.debug(
            'Starting iteration %d, time left: %f',
            iteration,
            time_left - used_time,
        )

        self.metric = [m for m in self.metrics if m.name == self.opt_metric][0]
        if not self.metric:
            raise ValueError(f"Cannot optimize for {self.opt_metric} in {self.metrics} "
                             "as more than one unique optimization metric was found.")


        self.current_ensemble_identifiers = self._load_current_ensemble_identifiers(cur_stacking_layer=self.cur_stacking_layer)
        best_model_identifier = self.get_identifiers_from_run_history()[-1]
        selected_key = self.read_model_predictions(best_model_identifier)

        # train ensemble
        ensemble = self.fit_ensemble(selected_keys=[selected_key])

        # Save the ensemble for later use in the main module!
        if ensemble is not None and self.SAVE2DISC:
            ensemble_identifiers = save_stacking_ensemble(
                iteration=int(self.cur_stacking_layer * ENSEMBLE_ITERATION_MULTIPLIER + iteration),
                ensemble=ensemble,
                seed=self.seed,
                cur_stacking_layer=self.cur_stacking_layer,
                backend=self.backend)
            self.logger.debug(f"ensemble_identifiers being saved are {ensemble_identifiers}")

        # Save the read losses status for the next iteration
        with open(self.ensemble_loss_file, "wb") as memory:
            pickle.dump(self.read_losses, memory)

        if ensemble is not None:
            train_pred = self.predict(set_="train",
                                      ensemble=ensemble,
                                      selected_keys=ensemble_identifiers,
                                      n_preds=len(ensemble_identifiers),
                                      index_run=iteration)
            # TODO if predictions fails, build the model again during the
            #  next iteration!
            test_pred = self.predict(set_="test",
                                     ensemble=ensemble,
                                     selected_keys=ensemble_identifiers,
                                     n_preds=len(ensemble_identifiers),
                                     index_run=iteration)

            # Add a score to run history to see ensemble progress
            self._add_ensemble_trajectory(
                train_pred,
                test_pred
            )
        
        # The loaded predictions and the hash can only be saved after the ensemble has been
        # built, because the hash is computed during the construction of the ensemble
        with open(self.ensemble_memory_file, "wb") as memory:
            pickle.dump(self.read_preds, memory)

        if return_predictions:
            return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
        else:
            return self.ensemble_history, self.ensemble_nbest, None, None

    def read_model_predictions(self, model_identifier) -> str:
        """
            returns the preds for the specified model
        """

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: %s",
                    traceback.format_exc(),
                )
                return False

        self.logger.debug("Read ensemble data set predictions")

        pred_path = os.path.join(
            glob.escape(self.backend.get_runs_directory()),
            '%d_*_*' % self.seed,
            'predictions_ensemble_%s_*_*.npy*' % self.seed,
        )
        y_ens_files = glob.glob(pred_path)
        y_ens_files = [y_ens_file for y_ens_file in y_ens_files
                       if y_ens_file.endswith('.npy') or y_ens_file.endswith('.npy.gz')]
        self.y_ens_files = y_ens_files
        # no validation predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug("Found no prediction files on ensemble data set:"
                              " %s" % pred_path)
            return False

        # First sort files chronologically
        to_read = []
        for y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(y_ens_fn)
            if match is None:
                continue
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))
            if (_seed, _num_run, _budget) == model_identifier:
                to_read.append([y_ens_fn, match, _seed, _num_run, _budget])

        if len(to_read) < 0:
            raise ValueError(f"Could not read model predictions at iteration {self.iteration}")

        n_read_files = 0
        # Now read file wrt to num_run
        # Mypy assumes sorted returns an object because of the lambda. Can't get to recognize the list
        # as a returning list, so as a work-around we skip next line
        for y_ens_fn, match, _seed, _num_run, _budget in sorted(to_read, key=lambda x: x[3]):  # type: ignore

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                raise RuntimeError('Error loading file (not .npy or .npy.gz): %s', y_ens_fn)

            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    Y_ENSEMBLE: None,
                    Y_TEST: None,
                }

            if not self.read_losses.get(y_ens_fn):
                self.read_losses[y_ens_fn] = {
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    # 3 - deleted from disk due to space constraints
                    "loaded": 0
                }

            # actually read the predictions and compute their respective loss
            try:
                ensemble_idenitfiers = self.current_ensemble_identifiers.copy()
                ensemble_idenitfiers[self.ensemble_slot_j] = y_ens_fn

                self.read_preds[y_ens_fn][Y_ENSEMBLE] = self._read_np_fn(y_ens_fn)
                success_keys_test = self.get_test_preds(self.read_preds.keys())
                if len(success_keys_test) < 1:
                    raise RuntimeError("Something went wrong when loading the test predictions of the model")
                self.read_losses[y_ens_fn]['loaded'] = 1
                n_read_files += 1

            except Exception:
                self.logger.warning(
                    'Error loading %s: %s',
                    y_ens_fn,
                    traceback.format_exc(),
                )

        self.logger.debug(
            'Done reading %d new prediction files. Loaded %d predictions in '
            'total.',
            n_read_files,
            n_read_files,
        )
        return y_ens_fn

    def get_test_preds(self, selected_keys: List[str]) -> List[str]:
        """
        test predictions from disc
        and store them in self.read_preds
        Parameters
        ---------
        selected_keys: list
            list of selected keys of self.read_preds
        Return
        ------
        success_keys:
            all keys in selected keys for which we could read the valid and
            test predictions
        """
        success_keys_test = []

        for k in selected_keys:
            test_fn = glob.glob(
                os.path.join(
                    glob.escape(self.backend.get_runs_directory()),
                    '%d_%d_%s' % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                    'predictions_test_%d_%d_%s.npy*' % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"]
                    )
                )
            )
            test_fn = [tfn for tfn in test_fn if tfn.endswith('.npy') or tfn.endswith('.npy.gz')]

            if len(test_fn) == 0:
                # self.logger.debug("Not found test prediction file (although "
                #                   "ensemble predictions available):%s" %
                #                   test_fn)
                pass
            else:
                if (
                    k in self.read_preds
                    and self.read_preds[k][Y_TEST] is not None
                ):
                    success_keys_test.append(k)
                    continue
                try:
                    y_test = self._read_np_fn(test_fn[0])
                    self.read_preds[k][Y_TEST] = y_test
                    success_keys_test.append(k)
                    self.read_losses[k]["mtime_test"] = os.path.getmtime(test_fn[0])
                except Exception:
                    self.logger.warning('Error loading %s: %s',
                                        test_fn, traceback.format_exc())

        return success_keys_test

    def fit_ensemble(self, selected_keys: List[str]) -> Optional[EnsembleSelection]:
        """
            fit ensemble

            Parameters
            ---------
            selected_keys: list
                list of selected keys of self.read_losses

            Returns
            -------
            ensemble: EnsembleSelection
                trained Ensemble
        """

        if self.unit_test:
            raise MemoryError()

        best_model_identifier = selected_keys[0]

        predictions_train = [self.read_preds[k][Y_ENSEMBLE] if k is not None else None for k in self.current_ensemble_identifiers]

        best_model_predictions_ensemble = self.read_preds[best_model_identifier][Y_ENSEMBLE]
        best_model_predictions_test = self.read_preds[best_model_identifier][Y_TEST]

        ensemble_num_runs = get_num_runs_from_identifiers(self.backend, self.model_fn_re, self.current_ensemble_identifiers)

        best_model_num_run = (
                            self.read_losses[best_model_identifier]["seed"],
                            self.read_losses[best_model_identifier]["num_run"],
                            self.read_losses[best_model_identifier]["budget"],
                            )
        stacked_ensemble_identifiers = self._load_stacked_ensemble_identifiers()
        self.logger.debug(f"Stacked ensemble identifiers: {stacked_ensemble_identifiers}")
        stacked_ensemble_num_runs = [
            get_num_runs_from_identifiers(self.backend, self.model_fn_re, layer_identifiers)
            for layer_identifiers in stacked_ensemble_identifiers
        ]

        predictions_stacking_ensemble = [
            [
                {'ensemble': self.read_preds[k][Y_ENSEMBLE], 'test': self.read_preds[k][Y_TEST]} if k is not None else None for k in layer_identifiers
            ]
            for layer_identifiers in stacked_ensemble_identifiers
        ]
        unique_identifiers = self._load_ensemble_unique_identifier()
        ensemble = IterativeHPOStackingEnsemble(
            ensemble_size=self.ensemble_size,
            metric=self.metric,
            random_state=self.random_state,
            task_type=self.task_type,
            ensemble_slot_j=self.ensemble_slot_j,
            cur_stacking_layer=self.cur_stacking_layer,
            stacked_ensemble_identifiers=stacked_ensemble_num_runs,
            predictions_stacking_ensemble=predictions_stacking_ensemble,
            unique_identifiers=unique_identifiers
        )
        try:
            self.logger.debug(
                "Fitting the single best ensemble",
            )
            start_time = time.time()

            ensemble.fit(
                predictions_train, 
                best_model_predictions_ensemble,
                best_model_predictions_test,
                self.y_true_ensemble,
                ensemble_num_runs,
                best_model_num_run
                )

            end_time = time.time()
            self.logger.debug(
                "Fitting the ensemble took %.2f seconds.",
                end_time - start_time,
            )
            self.logger.info(str(ensemble))
            self.validation_performance_ = min(
                self.validation_performance_,
                ensemble.get_validation_performance(),
            )

        except ValueError:
            self.logger.error('Caught ValueError: %s', traceback.format_exc())
            return None
        except IndexError:
            self.logger.error('Caught IndexError: %s' + traceback.format_exc())
            return None
        finally:
            # Explicitly free memory
            del predictions_train

        return ensemble

    def predict(self, set_: str,
                ensemble: AbstractEnsemble,
                selected_keys: list,
                n_preds: int,
                index_run: int) -> np.ndarray:
        """
            save preditions on ensemble, validation and test data on disc
            Parameters
            ----------
            set_: ["test"]
                data split name
            ensemble: EnsembleSelection
                trained Ensemble
            selected_keys: list
                list of selected keys of self.read_losses
            n_preds: int
                number of prediction models used for ensemble building
                same number of predictions on valid and test are necessary
            index_run: int
                n-th time that ensemble predictions are written to disc
            Return
            ------
            y: np.ndarray
        """
        self.logger.debug("Predicting the %s set with the ensemble!", set_)

        if set_ == 'test':
            pred_set = Y_TEST
        else:
            pred_set = Y_ENSEMBLE

        predictions = [self.read_preds[k][pred_set] if k is not None else None for k in selected_keys]

        if n_preds == len(predictions):
            y = ensemble.predict(predictions)
            if self.output_type == BINARY:
                y = y[:, 1]
            if self.SAVE2DISC:
                self.backend.save_predictions_as_txt(
                    predictions=y,
                    subset=set_,
                    idx=index_run,
                    prefix=self.dataset_name,
                    precision=8,
                )
            return y
        else:
            self.logger.info(
                "Found inconsistent number of predictions and models (%d vs "
                "%d) for subset %s",
                len(predictions),
                n_preds,
                set_,
            )
            return None

    def _add_ensemble_trajectory(self, train_pred: np.ndarray, test_pred: np.ndarray) -> None:
        """
        Records a snapshot of how the performance look at a given training
        time.
        Parameters
        ----------
        ensemble: EnsembleSelection
            The ensemble selection object to record
        test_pred: np.ndarray
            The predictions on the test set using ensemble
        """
        performance_stamp = {
            'Timestamp': pd.Timestamp.now(),
        }
        if self.output_type == BINARY:
            if len(train_pred.shape) == 1 or train_pred.shape[1] == 1:
                train_pred = np.vstack(
                    ((1 - train_pred).reshape((1, -1)), train_pred.reshape((1, -1)))
                ).transpose()
            if test_pred is not None and (len(test_pred.shape) == 1 or test_pred.shape[1] == 1):
                test_pred = np.vstack(
                    ((1 - test_pred).reshape((1, -1)), test_pred.reshape((1, -1)))
                ).transpose()

        train_scores = calculate_score(
            metrics=self.metrics,
            target=self.y_true_ensemble,
            prediction=train_pred,
            task_type=self.task_type,
        )
        performance_stamp.update({'train_' + str(key): val for key, val in train_scores.items()})
        if self.y_test is not None and test_pred is not None:
            test_scores = calculate_score(
                metrics=self.metrics,
                target=self.y_test,
                prediction=test_pred,
                task_type=self.task_type,
            )
            performance_stamp.update(
                {'test_' + str(key): val for key, val in test_scores.items()})

        self.ensemble_history.append(performance_stamp)

    def _read_np_fn(self, path: str) -> np.ndarray:
        return read_np_fn(self.precision, path)

    def get_identifiers_from_run_history(self) -> List[Tuple[int, int, float]]:
        """
        This method parses the run history, to identify
        the best performing model
        It populates the identifiers attribute, which is used
        by the backend to access the actual model
        """
        best_model_identifier = []
        best_model_score = self.metric._worst_possible_result

        data = self.run_history.data if isinstance(self.run_history, RunHistory) else self.run_history
        for run_key in data.keys():
            run_value = data[run_key]
            if run_value.status == StatusType.CRASHED:
                continue

            score = self.metric._optimum - (self.metric._sign * run_value.cost)

            if (score > best_model_score and self.metric._sign > 0) \
                    or (score < best_model_score and self.metric._sign < 0):

                # Make sure that the individual best model actually exists
                model_dir = self.backend.get_numrun_directory(
                    self.seed,
                    run_value.additional_info['num_run'],
                    run_key.budget,
                )
                model_file_name = self.backend.get_model_filename(
                    self.seed,
                    run_value.additional_info['num_run'],
                    run_key.budget,
                )
                file_path = os.path.join(model_dir, model_file_name)
                if not os.path.exists(file_path):
                    continue

                best_model_identifier = [(
                    self.seed,
                    run_value.additional_info['num_run'],
                    run_key.budget,
                )]
                best_model_score = score

        if not best_model_identifier:
            raise ValueError(
                "No valid model found in run history. This means smac was not able to fit"
                " a valid model. Please check the log file for errors."
            )

        self.best_performance = best_model_score

        return best_model_identifier

    
    def _save_ensemble_cutoff_num_run(self, cutoff_num_run: int) -> None:
        with open(get_ensemble_cutoff_num_run_filename(self.backend), "w") as file:
            file.write(str(cutoff_num_run))

    def _save_ensemble_unique_identifier(self, ensemble_unique_identifier: dict()) -> None:
        pickle.dump(ensemble_unique_identifier, open(get_ensemble_unique_identifier_filename(self.backend), 'wb'))

    def _load_ensemble_unique_identifier(self):
        if os.path.exists(get_ensemble_unique_identifier_filename(self.backend)):
            ensemble_unique_identifier = pickle.load(open(get_ensemble_unique_identifier_filename(self.backend), "rb"))   
        else:
            ensemble_unique_identifier = dict()
        return ensemble_unique_identifier

    def _load_ensemble_cutoff_num_run(self) -> Optional[int]:
        if os.path.exists(get_ensemble_cutoff_num_run_filename(self.backend)):
            with open(get_ensemble_cutoff_num_run_filename(self.backend), "r") as file:
                cutoff_num_run = int(file.read())
        else:
            cutoff_num_run = None
        return cutoff_num_run

    def _save_current_ensemble_identifiers(self, ensemble_identifiers: List[Optional[str]], cur_stacking_layer) -> None:
        with open(get_ensemble_identifiers_filename(self.backend, cur_stacking_layer=cur_stacking_layer), "wb") as file:
            pickle.dump(ensemble_identifiers, file=file)
    
    def _load_current_ensemble_identifiers(self, cur_stacking_layer) -> List[Optional[str]]:
        file_name = get_ensemble_identifiers_filename(self.backend,cur_stacking_layer)
        if os.path.exists(file_name):
            with open(file_name, "rb") as file:
                identifiers = pickle.load(file)
        else:
            identifiers = [None]*self.ensemble_size
        return identifiers

    def _load_stacked_ensemble_identifiers(self) -> List[List[Optional[str]]]:
        ensemble_identifiers = list()
        for i in range(self.num_stacking_layers):
            ensemble_identifiers.append(self._load_current_ensemble_identifiers(cur_stacking_layer=i))
        return ensemble_identifiers

