import os
from enum import Enum
import gzip
from math import floor
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Type, Union

import pickle

import re

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import glob

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

import torch
from torch.utils.data.dataloader import default_collate

from autoPyTorch.automl_common.common.utils.backend import Backend, BackendContext


MODEL_FN_RE = r'_([0-9]*)_([0-9]*)_([0-9]+\.*[0-9]*)\.npy'


HyperparameterValueType = Union[int, str, float]


ENSEMBLE_ITERATION_MULTIPLIER = 1e10
TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT_TRUE = 0.85
TIME_ALLOCATION_FACTOR_POSTHOC_ENSEMBLE_FIT_FALSE = 0.9
TIME_FOR_BASE_MODELS_SEARCH = 0.5


def get_train_phase_cutoff_numrun_filename(backend: 'autoPyTorchBackend') -> str:
    return os.path.join(backend.internals_directory, 'train_phase_cutoff_numrun.txt')


def ispandas(X: Any) -> bool:
    """ Whether X is pandas.DataFrame or pandas.Series """
    return hasattr(X, "iloc")


class FitRequirement(NamedTuple):
    """
    A class that holds inputs required to fit a pipeline. Also indicates whether
    requirements have to be user specified or are generated by the pipeline itself.

    Attributes:
        name (str): The name of the variable expected in the input dictionary
        supported_types (Iterable[Type]): An iterable of all types that are supported
        user_defined (bool): If false, this requirement does not have to be given to the pipeline
        dataset_property (bool): If True, this requirement is automatically inferred
            by the Dataset class
    """

    name: str
    supported_types: Iterable[Type]
    user_defined: bool
    dataset_property: bool

    def __str__(self) -> str:
        """
        String representation for the requirements
        """
        return "Name: %s | Supported types: %s | User defined: %s | Dataset property: %s" % (
            self.name, self.supported_types, self.user_defined, self.dataset_property)


class HyperparameterSearchSpace(NamedTuple):
    """
    A class that holds the search space for an individual hyperparameter.
    Attributes:
        hyperparameter (str):
            name of the hyperparameter
        value_range (Sequence[HyperparameterValueType]):
            range of the hyperparameter, can be defined as min and
            max values for Numerical hyperparameter or a list of
            choices for a Categorical hyperparameter
        default_value (HyperparameterValueType):
            default value of the hyperparameter
        log (bool):
            whether to sample hyperparameter on a log scale
    """
    hyperparameter: str
    value_range: Sequence[HyperparameterValueType]
    default_value: HyperparameterValueType
    log: bool = False

    def __str__(self) -> str:
        """
        String representation for the Search Space
        """
        return "Hyperparameter: %s | Range: %s | Default: %s | log: %s" % (
            self.hyperparameter, self.value_range, self.default_value, self.log)


class autoPyTorchEnum(str, Enum):
    """
    Utility class for enums in autoPyTorch.
    Allows users to use strings, while we internally use
    this enum
    """
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, autoPyTorchEnum):
            return type(self) == type(other) and self.value == other.value
        elif isinstance(other, str):
            return bool(self.value == other)
        else:
            enum_name = self.__class__.__name__
            raise RuntimeError(f"Unsupported type {type(other)}. "
                               f"{enum_name} only supports `str` and"
                               f"`{enum_name}`")

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return str(self.value)


def replace_prefix_in_config_dict(config: Dict[str, Any], prefix: str, replace: str = "") -> Dict[str, Any]:
    """
    Replace the prefix in all keys with the specified replacement string (the empty string by
    default to remove the prefix from the key). The functions makes sure that the prefix is a proper config
    prefix by checking if it ends with ":", if not it appends ":" to the prefix.

    :param config: config dictionary where the prefixed of the keys should be replaced
    :param prefix: prefix to be replaced in each key
    :param replace: the string to replace the prefix with
    :return: updated config dictionary
    """
    # make sure that prefix ends with the config separator ":"
    if not prefix.endswith(":"):
        prefix = prefix + ":"
    # only replace first occurrence of the prefix
    return {k.replace(prefix, replace, 1): v
            for k, v in config.items() if
            k.startswith(prefix)}


def custom_collate_fn(batch: List) -> List[Optional[torch.Tensor]]:
    """
    In the case of not providing a y tensor, in a
    dataset of form {X, y}, y would be None.

    This custom collate function allows to yield
    None data for functions that require only features,
    like predict.

    Args:
        batch (List): a batch from a dataset

    Returns:
        List[Optional[torch.Tensor]]
    """

    items = list(zip(*batch))

    # The feature will always be available
    items[0] = default_collate(items[0])
    if None in items[1]:
        items[1] = list(items[1])
    else:
        items[1] = default_collate(items[1])
    return items


def dict_repr(d: Optional[Dict[Any, Any]]) -> str:
    """ Display long message in dict as it is. """
    if isinstance(d, dict):
        return "\n".join(["{}: {}".format(k, v) for k, v in d.items()])
    else:
        return "None"


def replace_string_bool_to_bool(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to replace string-type bool to
    bool when a dict is read from json

    Args:
        dictionary (Dict[str, Any])
    Returns:
        Dict[str, Any]
    """
    for key, item in dictionary.items():
        if isinstance(item, str):
            if item.lower() == "true":
                dictionary[key] = True
            elif item.lower() == "false":
                dictionary[key] = False
    return dictionary


def get_device_from_fit_dictionary(X: Dict[str, Any]) -> torch.device:
    """
    Get a torch device object by checking if the fit dictionary specifies a device. If not, or if no GPU is available
    return a CPU device.

    Args:
        X (Dict[str, Any]): A fit dictionary to control how the pipeline is fitted
            See autoPyTorch/pipeline/components/base_component.py::autoPyTorchComponent for more details
            about fit_dictionary

    Returns:
        torch.device: Device to be used for training/inference
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(X.get("device", "cpu"))


def subsampler(data: Union[np.ndarray, pd.DataFrame, spmatrix],
               x: Union[np.ndarray, List[int]]
               ) -> Union[np.ndarray, pd.DataFrame, spmatrix]:
    return data[x] if isinstance(data, (np.ndarray, spmatrix)) else data.iloc[x]


def get_hyperparameter(hyperparameter: HyperparameterSearchSpace,
                       hyperparameter_type: Type[Hyperparameter]) -> Hyperparameter:
    """
    Given a hyperparameter search space, return a ConfigSpace Hyperparameter
    Args:
        hyperparameter (HyperparameterSearchSpace):
            the search space for the hyperparameter
        hyperparameter_type (Hyperparameter):
            the type of the hyperparameter

    Returns:
        Hyperparameter
    """
    if len(hyperparameter.value_range) == 0:
        raise ValueError(hyperparameter.hyperparameter + ': The range has to contain at least one element')
    if len(hyperparameter.value_range) == 1 and hyperparameter_type != CategoricalHyperparameter:
        return Constant(hyperparameter.hyperparameter, hyperparameter.value_range[0])
    if len(hyperparameter.value_range) == 2 and hyperparameter.value_range[0] == hyperparameter.value_range[1]:
        return Constant(hyperparameter.hyperparameter, hyperparameter.value_range[0])
    if hyperparameter_type == CategoricalHyperparameter:
        return CategoricalHyperparameter(hyperparameter.hyperparameter,
                                         choices=hyperparameter.value_range,
                                         default_value=hyperparameter.default_value)
    if hyperparameter_type == UniformFloatHyperparameter:
        assert len(hyperparameter.value_range) == 2, \
            "Float HP range update for %s is specified by the two upper " \
            "and lower values. %s given." % (hyperparameter.hyperparameter, len(hyperparameter.value_range))
        return UniformFloatHyperparameter(hyperparameter.hyperparameter,
                                          lower=hyperparameter.value_range[0],
                                          upper=hyperparameter.value_range[1],
                                          log=hyperparameter.log,
                                          default_value=hyperparameter.default_value)
    if hyperparameter_type == UniformIntegerHyperparameter:
        assert len(hyperparameter.value_range) == 2, \
            "Int HP range update for %s is specified by the two upper " \
            "and lower values. %s given." % (hyperparameter.hyperparameter, len(hyperparameter.value_range))
        return UniformIntegerHyperparameter(hyperparameter.hyperparameter,
                                            lower=hyperparameter.value_range[0],
                                            upper=hyperparameter.value_range[1],
                                            log=hyperparameter.log,
                                            default_value=hyperparameter.default_value)
    raise ValueError('Unknown type: %s for hp %s' % (hyperparameter_type, hyperparameter.hyperparameter))


def add_hyperparameter(cs: ConfigurationSpace,
                       hyperparameter: HyperparameterSearchSpace,
                       hyperparameter_type: Type[Hyperparameter]) -> None:
    """
    Adds the given hyperparameter to the given configuration space

    Args:
        cs (ConfigurationSpace):
            Configuration space where the hyperparameter must be added
        hyperparameter (HyperparameterSearchSpace):
            search space of the hyperparameter
        hyperparameter_type (Hyperparameter):
            type of the hyperparameter

    Returns:
        None
    """
    cs.add_hyperparameter(get_hyperparameter(hyperparameter, hyperparameter_type))


def check_none(p: Any) -> bool:
    """
    utility function to check if `p` is None.

    Args:
        p (str):
            variable to check

    Returns:
        bool:
            True, if `p` is in (None, "none", "None")
    """
    if p in ("None", "none", None):
        return True
    return False


def validate_config(config: Configuration, search_space: ConfigurationSpace, n_numerical_in_incumbent_on_task_id, num_numerical, assert_autogluon_numerical_hyperparameters: bool=False):
    modified_config = config.get_dictionary().copy() if isinstance(config, Configuration) else config.copy()

    if num_numerical > 0:
        imputer_numerical_hyperparameter = "imputer:numerical_strategy" 
        if imputer_numerical_hyperparameter not in modified_config:
            modified_config[imputer_numerical_hyperparameter] = search_space.get_hyperparameter(imputer_numerical_hyperparameter).default_value if not assert_autogluon_numerical_hyperparameters else 'median'
        if assert_autogluon_numerical_hyperparameters:
            quantile_hp_name = 'QuantileTransformer'
            skew_transformer_choice = modified_config.get('skew_transformer:__choice__', None)
            if skew_transformer_choice is not None:
                if skew_transformer_choice != quantile_hp_name:
                    to_remove_hps = [hyp.name for hyp in search_space.get_children_of('skew_transformer:__choice__') if skew_transformer_choice in hyp.name]
                    [modified_config.pop(remove_hp, None) for remove_hp in to_remove_hps]

            to_add_hps = [hyp for hyp in search_space.get_children_of('skew_transformer:__choice__') if quantile_hp_name in hyp.name]
            modified_config['skew_transformer:__choice__'] = quantile_hp_name
            for add_hp in to_add_hps:
                modified_config[add_hp.name] = add_hp.default_value

    feature_preprocessing_choice = modified_config['feature_preprocessor:__choice__']

    to_adjust_hyperparams = ['n_clusters', 'n_components', 'target_dim']
    children_hyperparameters = [hyp for hyp in search_space.get_children_of('feature_preprocessor:__choice__') if feature_preprocessing_choice in hyp.name]
    for hyp in children_hyperparameters:
        children = search_space.get_children_of(hyp)
        if len(children) > 0:
            children_hyperparameters.extend(children)
    children_hyperparameters = [hyp for hyp in children_hyperparameters if hyp.name in modified_config and any([ta_hyp in hyp.name for ta_hyp in to_adjust_hyperparams])]

    for child_hyperparam in children_hyperparameters:
        modified_config[child_hyperparam.name] = floor(modified_config[child_hyperparam.name]/n_numerical_in_incumbent_on_task_id * num_numerical)

    return Configuration(search_space, modified_config)


def read_np_fn(precision,  path: str) -> np.ndarray:
    if path.endswith("gz"):
        fp = gzip.open(path, 'rb')
    elif path.endswith("npy"):
        fp = open(path, 'rb')
    else:
        raise ValueError("Unknown filetype %s" % path)
    if precision == 16:
        predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float16)
    elif precision == 32:
        predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float32)
    elif precision == 64:
        predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float64)
    else:
        predictions = np.load(fp, allow_pickle=True)
    fp.close()
    return predictions


def delete_other_runs(ensemble_runs, runs_directory):
    all_runs = os.listdir(runs_directory)
    for run in all_runs:
        if run not in ensemble_runs:
            shutil.rmtree(os.path.join(runs_directory, run))


def delete_runs_except_ensemble(old_ensemble, backend):
    selected_identifiers = old_ensemble.get_selected_model_identifiers()[old_ensemble.cur_stacking_layer]
    nonnull_identifiers = [identifier for identifier in selected_identifiers if identifier is not None]
    ensemble_runs = [backend.get_numrun_directory(seed=seed, num_run=num_run, budget=budget).split('/')[-1] for seed, num_run, budget in nonnull_identifiers]
    delete_other_runs(ensemble_runs=ensemble_runs, runs_directory=backend.get_runs_directory())


def get_column_data(column, data):
    col_data = data[column] if isinstance(data, pd.DataFrame) else data[:, column]
    return col_data


def get_ensemble_identifiers_filename(backend, cur_stacking_layer) -> str:
    return os.path.join(backend.internals_directory, f'ensemble_identifiers_{cur_stacking_layer}.pkl')


def get_ensemble_cutoff_num_run_filename(backend):
    return os.path.join(backend.internals_directory, 'ensemble_cutoff_run.txt')


def get_ensemble_unique_identifier_filename(backend):
        return os.path.join(backend.internals_directory, 'ensemble_unique_identifier.txt')


def read_predictions(backend, seed, initial_num_run, precision, data_set='ensemble', run_history_pred_path=None):
    if run_history_pred_path is not None and os.path.exists(run_history_pred_path):
        read_preds = pickle.load(open(run_history_pred_path, 'rb'))
    else:
        read_preds = {}

    pred_path = os.path.join(
            glob.escape(backend.get_runs_directory()),
            '%d_*_*' % seed,
            f'predictions_{data_set}_{seed}_*_*.npy*',
        )

    y_ens_files = glob.glob(pred_path)
    y_ens_files = [y_ens_file for y_ens_file in y_ens_files
                    if y_ens_file.endswith('.npy') or y_ens_file.endswith('.npy.gz')]
    # no validation predictions so far -- no files
    if len(y_ens_files) == 0:
        return False
    model_fn_re = re.compile(MODEL_FN_RE)
    # First sort files chronologically
    to_read = []
    for y_ens_fn in y_ens_files:
        match = model_fn_re.search(y_ens_fn)
        _seed = int(match.group(1))
        _num_run = int(match.group(2))
        _budget = float(match.group(3))

        to_read.append([y_ens_fn, match, _seed, _num_run, _budget])

    # Now read file wrt to num_run
    # Mypy assumes sorted returns an object because of the lambda. Can't get to recognize the list
    # as a returning list, so as a work-around we skip next line
    for y_ens_fn, match, _seed, _num_run, _budget in sorted(to_read, key=lambda x: x[3]):  # type: ignore
        # skip models that were part of previous stacking layer
        dict_key = (_seed, _num_run, _budget)
        if _num_run < initial_num_run:
            continue


        if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
            continue

        if dict_key not in read_preds:
            read_preds[dict_key] = read_np_fn(precision, y_ens_fn)

    if run_history_pred_path is not None:
        pickle.dump(read_preds, open(run_history_pred_path, 'wb'))
    return read_preds


class autoPyTorchBackend(Backend):
    def save_targets_ensemble(self, targets: np.ndarray) -> str:
        self._make_internals_directory()
        if not isinstance(targets, np.ndarray):
            raise ValueError("Targets must be of type np.ndarray, but is %s" % type(targets))

        filepath = self._get_targets_ensemble_filename()

        # Try to open the file without locking it, this will reduce the
        # number of times where we erroneously keep a lock on the ensemble
        # targets file although the process already was killed
        try:
            existing_targets = np.load(filepath, allow_pickle=True)
            if (
                existing_targets.shape == targets.shape and np.allclose(existing_targets, targets)
            ):

                return filepath
        except Exception:
            pass

        with tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(filepath), delete=False) as fh_w:
            np.save(fh_w, targets.astype(np.float32))
            tempname = fh_w.name

        os.rename(tempname, filepath)

        return filepath


def create(
    temporary_directory: str,
    output_directory: Optional[str],
    prefix: str,
    delete_tmp_folder_after_terminate: bool = True,
    delete_output_folder_after_terminate: bool = True,
) -> "Backend":
    context = BackendContext(
        temporary_directory,
        output_directory,
        delete_tmp_folder_after_terminate,
        delete_output_folder_after_terminate,
        prefix=prefix,
    )
    backend = autoPyTorchBackend(context, prefix)

    return backend