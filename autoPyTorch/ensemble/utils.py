from enum import IntEnum
import gzip
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble import EnsembleOptimisationStackingEnsemble
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble_builder import EnsembleOptimisationStackingEnsembleBuilder
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble_builder import EnsembleSelectionPerLayerStackingEnsembleBuilder
from autoPyTorch.ensemble.iterative_hpo_stacking_ensemble_builder import IterativeHPOStackingEnsembleBuilder
from autoPyTorch.utils.common import (
    get_ensemble_cutoff_num_run_filename,
    get_ensemble_identifiers_filename,
    get_ensemble_unique_identifier_filename
)

class BaseLayerEnsembleSelectionTypes(IntEnum):
    ensemble_selection = 1
    ensemble_bayesian_optimisation = 2
    ensemble_autogluon = 3
    ensemble_iterative_hpo = 4

    def is_stacking_ensemble(self) -> bool:
        return getattr(self, self.name) in (self.ensemble_bayesian_optimisation, self.ensemble_iterative_hpo)


class StackingEnsembleSelectionTypes(IntEnum):
    stacking_ensemble_bayesian_optimisation = 1
    stacking_ensemble_selection_per_layer = 2
    stacking_repeat_models = 3
    stacking_autogluon = 4
    stacking_ensemble_iterative_hpo = 5


def is_stacking(base_ensemble_method: BaseLayerEnsembleSelectionTypes, stacking_ensemble_method: Optional[StackingEnsembleSelectionTypes] = None) -> bool:
    is_base_ensemble_method_stacking = base_ensemble_method.is_stacking_ensemble()
    is_stacking_ensemble_method_stacking = stacking_ensemble_method is not None
    return is_base_ensemble_method_stacking or is_stacking_ensemble_method_stacking


def get_ensemble_builder_class(base_ensemble_method: int, stacking_ensemble_method: Optional[int] = None):
    if base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_selection:
        if stacking_ensemble_method is None or stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models:
            return EnsembleBuilder
        elif stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_ensemble_selection_per_layer:
            return EnsembleSelectionPerLayerStackingEnsembleBuilder
        else:
            raise ValueError(f"Expected stacking_ensemble_method: {stacking_ensemble_method} to be in "
                             f"[StackingEnsembleSelectionTypes.stacking_repeat_models, StackingEnsembleSelectionTypes.stacking_ensemble_selection_per_layer"
                             f" None]")
    elif base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation:
        if stacking_ensemble_method is None or stacking_ensemble_method in (StackingEnsembleSelectionTypes.stacking_repeat_models, StackingEnsembleSelectionTypes.stacking_ensemble_bayesian_optimisation):
            return EnsembleOptimisationStackingEnsembleBuilder
    elif base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_iterative_hpo:
        if stacking_ensemble_method is None or stacking_ensemble_method in (StackingEnsembleSelectionTypes.stacking_repeat_models, StackingEnsembleSelectionTypes.stacking_ensemble_iterative_hpo):
            return IterativeHPOStackingEnsembleBuilder


def get_identifiers_from_num_runs(backend, num_runs, subset='ensemble') -> List[Optional[str]]:
        identifiers: List[Optional[str]] = []
        for num_run in num_runs:
            identifier = None
            if num_run is not None:
                seed, idx, budget = num_run
                identifier = os.path.join(
                    backend.get_numrun_directory(seed, idx, budget),
                    backend.get_prediction_filename(subset, seed, idx, budget)
                )
            identifiers.append(identifier)
        return identifiers

def get_num_runs_from_identifiers(backend, model_fn_re, identifiers) -> List[Optional[Tuple[int, int, float]]]:
    num_runs: List[Optional[Tuple[int, int, float]]] = []
    for identifier in identifiers:
        num_run = None
        if identifier is not None:
            match = model_fn_re.search(identifier)
            if match is None:
                raise ValueError(f"Could not interpret file {identifier} "
                                "Something went wrong while scoring predictions")
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))
            num_run = (_seed, _num_run, _budget)
        num_runs.append(num_run)
    return num_runs


def save_ensemble_cutoff_num_run(backend, cutoff_num_run: int) -> None:
        with open(get_ensemble_cutoff_num_run_filename(backend), "w") as file:
            file.write(str(cutoff_num_run))

def save_ensemble_unique_identifier(backend, ensemble_unique_identifier: dict()) -> None:
    pickle.dump(ensemble_unique_identifier, open(get_ensemble_unique_identifier_filename(backend), 'wb'))

def load_ensemble_unique_identifier(backend):
    if os.path.exists(get_ensemble_unique_identifier_filename(backend)):
        ensemble_unique_identifier = pickle.load(open(get_ensemble_unique_identifier_filename(backend), "rb"))   
    else:
        ensemble_unique_identifier = dict()
    return ensemble_unique_identifier

def load_ensemble_cutoff_num_run(backend) -> Optional[int]:
    if os.path.exists(get_ensemble_cutoff_num_run_filename(backend)):
        with open(get_ensemble_cutoff_num_run_filename(backend), "r") as file:
            cutoff_num_run = int(file.read())
    else:
        cutoff_num_run = None
    return cutoff_num_run

def save_current_ensemble_identifiers(backend, ensemble_identifiers: List[Optional[str]], cur_stacking_layer) -> None:
    with open(get_ensemble_identifiers_filename(backend, cur_stacking_layer=cur_stacking_layer), "wb") as file:
        pickle.dump(ensemble_identifiers, file=file)

def load_current_ensemble_identifiers(backend, ensemble_size, cur_stacking_layer) -> List[Optional[str]]:
    file_name = get_ensemble_identifiers_filename(backend, cur_stacking_layer)
    if os.path.exists(file_name):
        with open(file_name, "rb") as file:
            identifiers = pickle.load(file)
    else:
        identifiers = [None]*ensemble_size
    return identifiers

def load_stacked_ensemble_identifiers(num_stacking_layers) -> List[List[Optional[str]]]:
    ensemble_identifiers = list()
    for i in range(num_stacking_layers):
        ensemble_identifiers.append(load_current_ensemble_identifiers(cur_stacking_layer=i))
    return ensemble_identifiers


def save_stacking_ensemble(iteration, ensemble, seed, cur_stacking_layer, backend):
    backend.save_ensemble(ensemble, iteration, seed)
    ensemble_identifiers = get_identifiers_from_num_runs(backend, ensemble.stacked_ensemble_identifiers[cur_stacking_layer])
    save_current_ensemble_identifiers(
            backend=backend,
            ensemble_identifiers=ensemble_identifiers,
            cur_stacking_layer=cur_stacking_layer
            )
    save_ensemble_cutoff_num_run(backend=backend, cutoff_num_run=self.initial_num_run)
    save_ensemble_unique_identifier(backend=backend, ensemble_unique_identifiers=ensemble.unique_identifiers)
    return ensemble_identifiers