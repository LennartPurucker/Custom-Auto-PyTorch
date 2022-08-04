from collections import OrderedDict
import logging
import logging.handlers
import os
import pickle
import time
import traceback
from typing import List, Optional, Tuple, Union

import numpy as np

import pandas as pd

from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import BINARY
from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.iterative_hpo_stacking_ensemble_builder import IterativeHPOStackingEnsembleBuilder
from autoPyTorch.ensemble.stacking_finetune_ensemble import StackingFineTuneEnsemble
from autoPyTorch.utils.common import read_np_fn, MODEL_FN_RE
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score
from autoPyTorch.utils.common import (
    get_ensemble_cutoff_num_run_filename,
    get_ensemble_identifiers_filename,
    get_ensemble_unique_identifier_filename
)

Y_ENSEMBLE = 0
Y_TEST = 1


class FineTuneStackingEnsembleBuilder(IterativeHPOStackingEnsembleBuilder):
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

        super(FineTuneStackingEnsembleBuilder, self).__init__(
            backend=backend,
            dataset_name=dataset_name,
            task_type=task_type,
            output_type=output_type,
            metrics=metrics,
            opt_metric=opt_metric,
            run_history=run_history,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            seed=seed,
            precision=precision,
            memory_limit=memory_limit,
            read_at_most=read_at_most,
            random_state=random_state,
            logger_port=logger_port,
            unit_test=unit_test,
            initial_num_run=initial_num_run,
            num_stacking_layers=num_stacking_layers,
            use_ensemble_opt_loss=use_ensemble_opt_loss,
            max_models_on_disc=max_models_on_disc,
            performance_range_threshold=performance_range_threshold,
        )

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

        predictions_train = []
        for (_seed, _num_run, _budget) in self._get_num_runs_from_identifiers(self.current_ensemble_identifiers):
            y_ens_fn = self._get_identifiers_from_num_runs([(_seed, _num_run, _budget)])[0]
            if _num_run < self.initial_num_run:
                ensemble_preds = self._read_np_fn(y_ens_fn.replace('predictions_ensemble', 'predictions_hpo_ensemble'))
                test_preds = self._read_np_fn(y_ens_fn.replace('predictions_ensemble', 'predictions_hpo_test'))
                predictions_train.append(ensemble_preds)
                self.logger.debug(f"adding num_run: {_num_run} to self.read_preds, shape: {ensemble_preds.shape}")
                self.read_preds[y_ens_fn] = {Y_ENSEMBLE: None, Y_TEST: None}
                self.read_preds[y_ens_fn][Y_ENSEMBLE] =  ensemble_preds
                self.read_preds[y_ens_fn][Y_TEST] =  test_preds
            else:
                predictions_train.append(self.read_preds[y_ens_fn][Y_ENSEMBLE])

        best_model_predictions_ensemble = self.read_preds[best_model_identifier][Y_ENSEMBLE]
        best_model_predictions_test = self.read_preds[best_model_identifier][Y_TEST]

        ensemble_num_runs = self._get_num_runs_from_identifiers(self.current_ensemble_identifiers)

        best_model_num_run = (
                            self.read_losses[best_model_identifier]["seed"],
                            self.read_losses[best_model_identifier]["num_run"],
                            self.read_losses[best_model_identifier]["budget"],
                            )
        stacked_ensemble_identifiers = self._load_stacked_ensemble_identifiers()
        self.logger.debug(f"Stacked ensemble identifiers: {stacked_ensemble_identifiers}, predictions_train shape: {predictions_train[0].shape}")
        stacked_ensemble_num_runs = [
            self._get_num_runs_from_identifiers(layer_identifiers)
            for layer_identifiers in stacked_ensemble_identifiers
        ]

        predictions_stacking_ensemble = []
        for layer_identifiers in stacked_ensemble_identifiers:
            predictions_layer = []
            for y_ens_fn in layer_identifiers:
                if y_ens_fn is not None:
                    if y_ens_fn not in self.read_preds:
                        ensemble_preds = self._read_np_fn(y_ens_fn.replace('predictions_ensemble', 'predictions_hpo_ensemble'))
                        test_preds = self._read_np_fn(y_ens_fn.replace('predictions_ensemble', 'predictions_hpo_test'))
                        self.read_preds[y_ens_fn] = {Y_ENSEMBLE: None, Y_TEST: None}
                        self.read_preds[y_ens_fn][Y_ENSEMBLE] =  ensemble_preds
                        self.read_preds[y_ens_fn][Y_TEST] =  test_preds

                    predictions_layer.append({'ensemble': self.read_preds[y_ens_fn][Y_ENSEMBLE], 'test': self.read_preds[y_ens_fn][Y_TEST]})
                else:
                    predictions_layer.append(None)
            predictions_stacking_ensemble.append(predictions_layer)

        self.logger.debug(f"in fine tune stacked builder, predictions_stacking_ensemble: {predictions_stacking_ensemble}")
        unique_identifiers = self._load_ensemble_unique_identifier()
        ensemble = StackingFineTuneEnsemble(
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
                f"Fitting the single best ensemble with y_true_ensemble: {self.y_true_ensemble.shape}",
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
