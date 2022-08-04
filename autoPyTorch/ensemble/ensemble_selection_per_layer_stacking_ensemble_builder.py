import glob
import logging
import logging.handlers
import os
import pickle
import re
import time
import traceback
import warnings
from typing import Dict, List, Optional, Tuple, Union
import zlib

import numpy as np
from smac.runhistory.runhistory import RunHistory

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import BINARY
from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble import EnsembleSelectionPerLayerStackingEnsemble
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss, calculate_score
from autoPyTorch.utils.common import ENSEMBLE_ITERATION_MULTIPLIER, MODEL_FN_RE
from autoPyTorch.utils.logging_ import get_named_client_logger

Y_ENSEMBLE = 0
Y_TEST = 1


class EnsembleSelectionPerLayerStackingEnsembleBuilder(EnsembleBuilder):
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        output_type: int,
        metrics: List[autoPyTorchMetric],
        opt_metric: str,
        run_history: Optional[RunHistory] = None,
        ensemble_size: int = 10,
        ensemble_nbest: int = 100,
        max_models_on_disc: Union[float, int] = 100,
        performance_range_threshold: float = 0,
        seed: int = 1,
        precision: int = 32,
        memory_limit: Optional[int] = 1024,
        read_at_most: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        unit_test: bool = False,
        use_ensemble_opt_loss=False,
        num_stacking_layers: int = 2,
        cur_stacking_layer: int = 0,
        initial_num_run: int = 0
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

        super(EnsembleSelectionPerLayerStackingEnsembleBuilder, self).__init__(
            backend=backend, dataset_name=dataset_name, task_type=task_type,
            output_type=output_type, metrics=metrics, opt_metric=opt_metric,
            ensemble_size=ensemble_size, ensemble_nbest=ensemble_nbest,
            max_models_on_disc=max_models_on_disc,
            performance_range_threshold=performance_range_threshold,
            seed=seed, precision=precision, memory_limit=memory_limit,
            read_at_most=read_at_most, random_state=random_state,
            logger_port=logger_port, unit_test=unit_test, initial_num_run=initial_num_run if cur_stacking_layer==0 else 1)

        self.num_stacking_layers = num_stacking_layers
        self.cur_stacking_layer = cur_stacking_layer
        self.ensembles = None
        self.ensemble_predictions = None
        old_ensemble: Optional[EnsembleSelectionPerLayerStackingEnsemble] = None
        if os.path.exists(self.backend.get_ensemble_dir()) and len(os.listdir(self.backend.get_ensemble_dir())) >= 1:
            old_ensemble = self.backend.load_ensemble(seed=seed)
            self.ensembles = old_ensemble.ensembles
            self.ensemble_predictions = old_ensemble.ensemble_predictions

    def run(
        self,
        iteration: int,
        pynisher_context: str,
        cur_stacking_layer: int,
        time_left: Optional[float] = None,
        end_at: Optional[float] = None,
        time_buffer: int = 5,
        return_predictions: bool = False,
        is_new_layer: bool = False,
        ) -> Tuple[List[Dict[str, float]], int, Optional[np.ndarray], Optional[np.ndarray]]:
        self.cur_stacking_layer = cur_stacking_layer
        self.is_new_layer = is_new_layer
        return super().run(iteration, pynisher_context, time_left, end_at, time_buffer, return_predictions)

    # This is the main wrapper to the EnsembleSelection class which fits the ensemble
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

        used_time = time.time() - self.start_time
        self.logger.debug(
            f'Starting iteration {iteration}, time left: {time_left - used_time} and initial num_run: {self.initial_num_run}',
        )

        if not self.compute_loss_per_model():
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None

        # Only the models with the n_best predictions are candidates
        # to be in the ensemble
        candidate_models = self.get_n_best_preds()
        if not candidate_models:  # no candidates yet
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None

        # populates test predictions in self.read_preds
        # reduces selected models if file reading failed
        n_sel_test = self.get_test_preds(selected_keys=candidate_models)

        # If any of n_sel_* is not empty and overlaps with candidate_models,
        # then ensure candidate_models AND n_sel_test are sorted the same
        candidate_models_set = set(candidate_models)
        if candidate_models_set.intersection(n_sel_test):
            candidate_models = sorted(list(candidate_models_set.intersection(
                n_sel_test)))
            n_sel_test = candidate_models
        else:
            # This has to be the case
            n_sel_test = []

        if os.environ.get('ENSEMBLE_KEEP_ALL_CANDIDATES'):
            for candidate in candidate_models:
                self._has_been_candidate.add(candidate)

        # self.logger.debug(f"for iteration {iteration}, best_model_identifier: {best_model_identifier} \n candidate_models: \n{candidate_models}")
        # train ensemble
        ensemble = self.fit_ensemble(selected_keys=candidate_models)
 
        # Save the ensemble for later use in the main module!
        if ensemble is not None and self.SAVE2DISC:
            self.backend.save_ensemble(ensemble, int(self.cur_stacking_layer * ENSEMBLE_ITERATION_MULTIPLIER + iteration), self.seed)
            # self._save_ensemble_cutoff_num_run(cutoff_num_run=self.cutoff_num_run)
        # Delete files of non-candidate models - can only be done after fitting the ensemble and
        # saving it to disc so we do not accidentally delete models in the previous ensemble
        if self.max_resident_models is not None:
            self._delete_excess_models(selected_keys=candidate_models)

        # Save the read losses status for the next iteration
        with open(self.ensemble_loss_file, "wb") as memory:
            pickle.dump(self.read_losses, memory)

        if ensemble is not None:
            train_pred = self.predict(set_="train",
                                      ensemble=ensemble,
                                      selected_keys=candidate_models,
                                      n_preds=len(candidate_models),
                                      index_run=iteration)
            # TODO if predictions fails, build the model again during the
            #  next iteration!
            test_pred = self.predict(set_="test",
                                     ensemble=ensemble,
                                     selected_keys=n_sel_test,
                                     n_preds=len(candidate_models),
                                     index_run=iteration)

            # Add a score to run history to see ensemble progress
            self._add_ensemble_trajectory(
                train_pred,
                test_pred
            )

        # The loaded predictions and the hash can only be saved after the ensemble has been
        # built, because the hash is computed during the construction of the ensemble
        with open(self.ensemble_memory_file, "wb") as memory:
            pickle.dump((self.read_preds, self.last_hash), memory)

        if return_predictions:
            return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
        else:
            return self.ensemble_history, self.ensemble_nbest, None, None

    def compute_loss_per_model(self) -> bool:
        """
            Compute the loss of the predictions on ensemble building data set;
            populates self.read_preds and self.read_losses
        """

        self.logger.debug("Read ensemble data set predictions")

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: %s",
                    traceback.format_exc(),
                )
                return False

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
                raise ValueError(f"Could not interpret file {y_ens_fn} "
                                 "Something went wrong while scoring predictions")
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            to_read.append([y_ens_fn, match, _seed, _num_run, _budget])

        n_read_files = 0
        # Now read file wrt to num_run
        # Mypy assumes sorted returns an object because of the lambda. Can't get to recognize the list
        # as a returning list, so as a work-around we skip next line
        for y_ens_fn, match, _seed, _num_run, _budget in sorted(to_read, key=lambda x: x[3]):  # type: ignore
            # skip models that were part of previous stacking layer
            if _num_run < self.initial_num_run:
                if y_ens_fn in self.read_losses:
                    self.logger.debug(f"deleting {y_ens_fn} in ensemble builder")
                    del self.read_losses[y_ens_fn]
                continue

            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                self.logger.info('Error loading file (not .npy or .npy.gz): %s', y_ens_fn)
                continue

            if not self.read_losses.get(y_ens_fn):
                self.read_losses[y_ens_fn] = {
                    "ens_loss": np.inf,
                    "mtime_ens": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    "disc_space_cost_mb": None,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    # 3 - deleted from disk due to space constraints
                    "loaded": 0
                }
            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    Y_ENSEMBLE: None,
                    Y_TEST: None,
                }

            if self.read_losses[y_ens_fn]["mtime_ens"] == os.path.getmtime(y_ens_fn):
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._read_np_fn(y_ens_fn)
                losses = calculate_loss(
                    metrics=self.metrics,
                    target=self.y_true_ensemble,
                    prediction=y_ensemble,
                    task_type=self.task_type,
                )

                if np.isfinite(self.read_losses[y_ens_fn]["ens_loss"]):
                    self.logger.debug(
                        'Changing ensemble loss for file %s from %f to %f '
                        'because file modification time changed? %f - %f',
                        y_ens_fn,
                        self.read_losses[y_ens_fn]["ens_loss"],
                        losses[self.opt_metric],
                        self.read_losses[y_ens_fn]["mtime_ens"],
                        os.path.getmtime(y_ens_fn),
                    )

                self.read_losses[y_ens_fn]["ens_loss"] = losses[self.opt_metric]

                # It is not needed to create the object here
                # To save memory, we just compute the loss.
                self.read_losses[y_ens_fn]["mtime_ens"] = os.path.getmtime(y_ens_fn)
                self.read_losses[y_ens_fn]["loaded"] = 2
                self.read_losses[y_ens_fn]["disc_space_cost_mb"] = self.get_disk_consumption(
                    y_ens_fn
                )

                n_read_files += 1

            except Exception:
                self.logger.warning(
                    'Error loading %s: %s',
                    y_ens_fn,
                    traceback.format_exc(),
                )
                self.read_losses[y_ens_fn]["ens_loss"] = np.inf

        self.logger.debug(
            'Done reading %d new prediction files. Loaded %d predictions in '
            'total.',
            n_read_files,
            np.sum([pred["loaded"] > 0 for pred in self.read_losses.values()])
        )
        return True

    def fit_ensemble(
        self,
        selected_keys: List[str]
    ) -> Optional[EnsembleSelectionPerLayerStackingEnsemble]:
        """
            fit ensemble

            Parameters
            ---------
            selected_keys: list
                list of selected keys of self.read_losses

            Returns
            -------
            ensemble: StackingEnsemble
                trained Ensemble
        """

        if self.unit_test:
            raise MemoryError()

        predictions_train = [self.read_preds[k][Y_ENSEMBLE] for k in selected_keys]
        include_num_runs = [
            (
                self.read_losses[k]["seed"],
                self.read_losses[k]["num_run"],
                self.read_losses[k]["budget"],
            )
            for k in selected_keys]

        # check hash if ensemble training data changed
        current_hash = "".join([
            str(zlib.adler32(predictions_train[i].data.tobytes()))
            for i in range(len(predictions_train))
        ])
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                "-- current performance: %f",
                self.validation_performance_,
            )

            return None
        self.last_hash = current_hash

        opt_metric = [m for m in self.metrics if m.name == self.opt_metric][0]
        if not opt_metric:
            raise ValueError(f"Cannot optimize for {self.opt_metric} in {self.metrics} "
                             "as more than one unique optimization metric was found.")


        cur_ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            metric=opt_metric,
            random_state=self.random_state,
            task_type=self.task_type,
        )

        try:
            # self.logger.debug(
            #     "Fitting the ensemble on %d models.",
            #     len(predictions_train),
            # )

            start_time = time.time()
            cur_ensemble.fit(
                predictions_train,
                self.y_true_ensemble,
                include_num_runs,
            )

            end_time = time.time()
            self.logger.debug(
                "Fitting the ensemble took %.2f seconds.",
                end_time - start_time,
            )
            # self.logger.debug(f"weights = {ensemble.weights_}")
            self.logger.info(str(cur_ensemble))
            self.validation_performance_ = min(
                self.validation_performance_,
                cur_ensemble.get_validation_performance(),
            )
            cur_ensemble_model_identifiers = self._get_identifiers_from_num_runs(
                cur_ensemble.get_selected_model_identifiers()
                )

            ensemble = EnsembleSelectionPerLayerStackingEnsemble(
                num_stacking_layers=self.num_stacking_layers,
                cur_stacking_layer=self.cur_stacking_layer,
                ensembles=self.ensembles,
                ensemble_predictions=self.ensemble_predictions
            )
            cur_ensemble_predictions_ensemble_set = [self.read_preds[k][Y_ENSEMBLE] for k in cur_ensemble_model_identifiers]
            cur_ensemble_predictions_test_set = [self.read_preds[k][Y_TEST] for k in cur_ensemble_model_identifiers]
            ensemble.fit(cur_ensemble=cur_ensemble, cur_ensemble_predictions={
                'ensemble': cur_ensemble_predictions_ensemble_set,
                'test': cur_ensemble_predictions_test_set
            })

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

    def _get_ensemble_identifiers_filename(self, cur_stacking_layer) -> str:
        return os.path.join(self.backend.internals_directory, f'ensemble_identifiers_{cur_stacking_layer}.pkl')

    def _get_ensemble_cutoff_num_run_filename(self):
        return os.path.join(self.backend.internals_directory, 'ensemble_cutoff_run.txt')

    def _save_ensemble_cutoff_num_run(self, cutoff_num_run: int) -> None:
        with open(self._get_ensemble_cutoff_num_run_filename(), "w") as file:
            file.write(str(cutoff_num_run))
    
    def _load_ensemble_cutoff_num_run(self) -> Optional[int]:
        if os.path.exists(self._get_ensemble_cutoff_num_run_filename()):
            with open(self._get_ensemble_cutoff_num_run_filename(), "r") as file:
                cutoff_num_run = int(file.read())
        else:
            cutoff_num_run = None
        return cutoff_num_run

    def _save_current_ensemble_identifiers(self, ensemble_identifiers: List[Optional[str]], cur_stacking_layer) -> None:
        with open(self._get_ensemble_identifiers_filename(cur_stacking_layer=cur_stacking_layer), "wb") as file:
            pickle.dump(ensemble_identifiers, file=file)
    
    def _load_current_ensemble_identifiers(self, cur_stacking_layer) -> List[Optional[str]]:
        file_name = self._get_ensemble_identifiers_filename(cur_stacking_layer)
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

    def _get_identifiers_from_num_runs(self, num_runs, subset='ensemble') -> List[Optional[str]]:
        identifiers: List[Optional[str]] = []
        for num_run in num_runs:
            identifier = None
            if num_run is not None:
                seed, idx, budget = num_run
                identifier = os.path.join(
                    self.backend.get_numrun_directory(seed, idx, budget),
                    self.backend.get_prediction_filename(subset, seed, idx, budget)
                )
            identifiers.append(identifier)
        return identifiers

    def _get_num_runs_from_identifiers(self, identifiers) -> List[Optional[Tuple[int, int, float]]]:
        num_runs: List[Optional[Tuple[int, int, float]]] = []
        for identifier in identifiers:
            num_run = None
            if identifier is not None:
                match = self.model_fn_re.search(identifier)
                if match is None:
                    raise ValueError(f"Could not interpret file {identifier} "
                                    "Something went wrong while scoring predictions")
                _seed = int(match.group(1))
                _num_run = int(match.group(2))
                _budget = float(match.group(3))
                num_run = (_seed, _num_run, _budget)
            num_runs.append(num_run)

        return num_runs