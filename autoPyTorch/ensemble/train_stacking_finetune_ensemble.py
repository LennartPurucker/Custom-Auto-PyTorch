from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.autogluon_stacking_ensemble import AutogluonStackingEnsemble
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss


class StackingTrainFineTuneEnsemble(AutogluonStackingEnsemble):

    def __str__(self) -> str:
        return 'Train Fine-Tune Stacking Ensemble:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.ensemble_weights[-1],
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.ensemble_weights[-1][idx] > 0]))

    def get_models_with_weights(
        self,
        models: Dict[Any, BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        """
        Handy function to tag the provided input models with a given weight.

        Args:
            models (List[Tuple[float, BasePipeline]]):
                A dictionary that maps a model's name to it's actual python object.

        Returns:
            output (List[Tuple[float, BasePipeline]]):
                each model with the related weight, sorted by ascending
                performance. Notice that ensemble selection solves a minimization
                problem.
        """
        outputs = []
        for layer_models, identifiers, layer_weights in zip(models, self.ensemble_identifiers, self.ensemble_weights):
            output = []
            for identifier, weight in zip(identifiers, layer_weights):
                model = layer_models[identifier]
                output.append((weight, model))
            output.sort(reverse=True, key=lambda t: t[0])
            outputs.append(output)

        return outputs

    def get_expanded_layer_stacking_ensemble_predictions(
        self,
        stacking_layer,
        raw_stacking_layer_ensemble_predictions
    ) -> List[np.ndarray]:
        layer_weights = self.ensemble_weights[stacking_layer]
        layer_size = len(self.ensemble_weights[stacking_layer])
        ensemble_predictions = []
        for weight, pred in zip(layer_weights, raw_stacking_layer_ensemble_predictions):
            ensemble_predictions.extend([pred] * int(weight * layer_size))
        return ensemble_predictions

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        """
        After training of ensemble selection, not all models will be used.
        Some of them will have zero weight. This procedure filters this models
        out.

        Returns:
            output (List[Tuple[int, int, float]]):
                The models actually used by ensemble selection
        """
        return self.ensemble_identifiers

    def get_validation_performance(self) -> float:
        """
        Returns the best optimization performance seen during hill climbing

        Returns:
            (float):
                best ensemble training performance
        """
        return 0

