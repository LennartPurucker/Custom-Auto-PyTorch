import os
import pickle
from typing import List, Optional, Tuple

from autoPyTorch.utils.common import (
    get_ensemble_cutoff_num_run_filename,
    get_ensemble_identifiers_filename,
    get_ensemble_unique_identifier_filename
)


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


def save_stacking_ensemble(iteration, ensemble, seed, cur_stacking_layer, backend, initial_num_run=None):
    backend.save_ensemble(ensemble, iteration, seed)
    ensemble_identifiers = get_identifiers_from_num_runs(backend, ensemble.stacked_ensemble_identifiers[cur_stacking_layer])
    save_current_ensemble_identifiers(
            backend=backend,
            ensemble_identifiers=ensemble_identifiers,
            cur_stacking_layer=cur_stacking_layer
            )
    if initial_num_run is not None:
        save_ensemble_cutoff_num_run(backend=backend, cutoff_num_run=initial_num_run)
    save_ensemble_unique_identifier(backend=backend, ensemble_unique_identifier=ensemble.unique_identifiers)
    return ensemble_identifiers