from enum import IntEnum
import gzip
from typing import Optional

import numpy as np

from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble import EnsembleOptimisationStackingEnsemble
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble_builder import EnsembleOptimisationStackingEnsembleBuilder
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble_builder import EnsembleSelectionPerLayerStackingEnsembleBuilder
from autoPyTorch.ensemble.iterative_hpo_stacking_ensemble_builder import IterativeHPOStackingEnsembleBuilder


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