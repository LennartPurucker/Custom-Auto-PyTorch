from enum import IntEnum
from typing import Optional

from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble_builder import EnsembleOptimisationStackingEnsembleBuilder
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble_builder import EnsembleSelectionPerLayerStackingEnsembleBuilder


class BaseLayerEnsembleSelectionTypes(IntEnum):
    ensemble_selection = 1
    ensemble_bayesian_optimisation = 2

    def is_stacking_ensemble(self) -> bool:
        return False


class StackingEnsembleSelectionTypes(IntEnum):
    stacking_ensemble_bayesian_optimisation = 1
    stacking_ensemble_selection_per_layer = 2
    stacking_repeat_models = 3
    stacking_autogluon = 4

    def is_stacking_ensemble(self) -> bool:
        return True


def get_ensemble_builder_class(base_ensemble_method: int, stacking_ensemble_method: Optional[int] = None):
    if base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_selection:
        if stacking_ensemble_method is None or stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models:
            return EnsembleBuilder
    elif base_ensemble_method == BaseLayerEnsembleSelectionTypes.ensemble_bayesian_optimisation:
        if stacking_ensemble_method is None or stacking_ensemble_method in (StackingEnsembleSelectionTypes.stacking_repeat_models, StackingEnsembleSelectionTypes.stacking_ensemble_bayesian_optimisation):
            return EnsembleOptimisationStackingEnsembleBuilder
    elif base_ensemble_method == StackingEnsembleSelectionTypes.stacking_ensemble_selection_per_layer:
        return EnsembleSelectionPerLayerStackingEnsembleBuilder
