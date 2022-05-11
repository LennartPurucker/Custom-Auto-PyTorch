from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss


class AutogluonStackingEnsemble(AbstractEnsemble):
    def __init__(
        self,
    ) -> None:
        self.ensemble_identifiers: Optional[List[List[Tuple[int, int, float]]]] = None
        self.ensemble_weights: Optional[List[List]] = None

    def fit(
        self,
        identifiers: List[List[Tuple[int, int, float]]],
        weights: List[List]
    ) -> AbstractEnsemble:
        """
        Builds a ensemble given the individual models out of fold predictions.
        Fundamentally, defines a set of weights on how to perform a soft-voting
        aggregation of the models in the given identifiers.

        Args:
            predictions (List[np.ndarray]):
                A list of individual model predictions of shape (n_datapoints, n_targets)
                corresponding to the OutOfFold estimate of the ground truth
            labels (np.ndarray):
                The ground truth targets of shape (n_datapoints, n_targets)
            identifiers: List[Tuple[int, int, float]]
                A list of model identifiers, each with the form
                (seed, number of run, budget)

        Returns:
            A copy of self
        """
        self.ensemble_identifiers = identifiers
        self.ensemble_weights = weights
        return self

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Given a list of predictions from the individual model, this method
        aggregates the predictions using a soft voting scheme with the weights
        found during training.

        Args:
            predictions (List[np.ndarray]):
                A list of predictions from the individual base models.

        Returns:
            average (np.ndarray): Soft voting predictions of ensemble models, using
                                the weights found during ensemble selection (self._weights)
        """

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.ensemble_weights[-1]):
            for pred, weight in zip(predictions, self.ensemble_weights[-1]):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.ensemble_weights[-1]):
            non_null_weights = [w for w in self.ensemble_weights[-1] if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average

    def __str__(self) -> str:
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
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

