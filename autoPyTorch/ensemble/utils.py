from enum import IntEnum

from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble_builder import EnsembleOptimisationStackingEnsembleBuilder
from autoPyTorch.ensemble.ensemble_selection_per_layer_stacking_ensemble_builder import EnsembleSelectionPerLayerStackingEnsembleBuilder


class EnsembleSelectionTypes(IntEnum):
    ensemble_selection = 1
    stacking_optimisation_ensemble = 2
    stacking_ensemble_selection_per_layer = 3
    stacking_repeat_models = 4
    stacking_autogluon = 5

    def is_stacking_ensemble(self) -> bool:
        stacked = [self.stacking_optimisation_ensemble,
                   self.stacking_ensemble_selection_per_layer,
                   self.stacking_repeat_models,
                   self.stacking_autogluon]
        return getattr(self, self.name) in stacked


def get_ensemble_builder_class(ensemble_method: int):
    if (
        ensemble_method == EnsembleSelectionTypes.ensemble_selection
        or ensemble_method == EnsembleSelectionTypes.stacking_repeat_models
        ):
        return EnsembleBuilder
    elif ensemble_method == EnsembleSelectionTypes.stacking_optimisation_ensemble:
        return EnsembleOptimisationStackingEnsembleBuilder
    elif ensemble_method == EnsembleSelectionTypes.stacking_ensemble_selection_per_layer:
        return EnsembleSelectionPerLayerStackingEnsembleBuilder
