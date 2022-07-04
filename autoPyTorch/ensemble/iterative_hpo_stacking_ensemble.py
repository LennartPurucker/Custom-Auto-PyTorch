from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_optimisation_stacking_ensemble import EnsembleOptimisationStackingEnsemble
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss


class IterativeHPOStackingEnsemble(EnsembleOptimisationStackingEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        metric: autoPyTorchMetric,
        task_type: int,
        random_state: np.random.RandomState,
        ensemble_slot_j: int,
        cur_stacking_layer: int,
        stacked_ensemble_identifiers: List[List[Optional[Tuple[int, int, float]]]],
        predictions_stacking_ensemble: List[List[Dict[str, Optional[np.ndarray]]]]
    ) -> None:
        super().__init__(
            ensemble_size=ensemble_size,
            metric=metric,
            task_type=task_type,
            random_state=random_state,
            ensemble_slot_j=ensemble_slot_j,
            cur_stacking_layer=cur_stacking_layer,
            stacked_ensemble_identifiers=stacked_ensemble_identifiers,
            predictions_stacking_ensemble=predictions_stacking_ensemble,
        )

    def __str__(self) -> str:
        return f"Iterative HPO Stacking Ensemble:\n\tWeights: {self.weights_}\
            \n\tIdentifiers: {' '.join([str(identifier) for idx, identifier in enumerate(self.identifiers_) if self.weights_[idx] > 0])}"
