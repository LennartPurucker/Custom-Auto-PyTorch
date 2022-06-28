from copyreg import pickle
from ctypes import cast
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np


from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss
from autoPyTorch.automl_common.common.utils.backend import Backend


class EnsembleSelectionPerLayerStackingEnsemble(AbstractEnsemble):
    def __init__(
        self,
        num_stacking_layers,
        cur_stacking_layer,
        ensembles = None,
        ensemble_predictions = None,
    ) -> None:
        self.ensembles: List[Optional[AbstractEnsemble]] = [None] * num_stacking_layers if ensembles is None else ensembles
        self.cur_stacking_layer = cur_stacking_layer
        self.ensemble_predictions = [None] * num_stacking_layers if ensemble_predictions is None else ensemble_predictions

    # def __getstate__(self) -> Dict[str, Any]:
    #     # Cannot serialize a metric if
    #     # it is user defined.
    #     # That is, if doing pickle dump
    #     # the metric won't be the same as the
    #     # one in __main__. we don't use the metric
    #     # in the EnsembleSelection so this should
    #     # be fine
    #     self.metric = None  # type: ignore
    #     return self.__dict__

    def fit(
        self,
        cur_ensemble: AbstractEnsemble,
        cur_ensemble_predictions,
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
        self.ensembles[self.cur_stacking_layer] = cur_ensemble
        self.ensemble_predictions[self.cur_stacking_layer] = cur_ensemble_predictions
        return self

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        # should be the last layer
        return self.ensembles[self.cur_stacking_layer].predict(predictions)

    def __str__(self) -> str:
        return f"Ensemble Selection Per Layer Stacking Ensemble:\n\tWeights: {self.ensembles[self.cur_stacking_layer].weights_}\
            \n\tIdentifiers: {' '.join([str(identifier) for idx, identifier in enumerate(self.ensembles[self.cur_stacking_layer].identifiers_) if self.ensembles[self.cur_stacking_layer].weights_[idx] > 0])}"

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        """
        After training of ensemble selection, not all models will be used.
        Some of them will have zero weight. This procedure filters this models
        out.

        Returns:
            output (List[Tuple[int, int, float]]):
                The models actually used by ensemble selection
        """
        ensemble_identifiers = list()
        for ensemble in self.ensembles:
            if ensemble is None:
                return ensemble_identifiers
            ensemble_identifiers.append(ensemble.get_selected_model_identifiers())
        
        return ensemble_identifiers

    def get_validation_performance(self) -> float:
        """
        Returns the best optimization performance seen during hill climbing

        Returns:
            (float):
                best ensemble training performance
        """
        return self.ensembles[self.cur_stacking_layer].trajectory_[-1]

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
        for ensemble, layer_models in zip(self.ensembles, models):
            outputs.append(ensemble.get_models_with_weights(layer_models))

        return outputs

    def get_expanded_layer_stacking_ensemble_predictions(
        self,
        stacking_layer: int,
        raw_stacking_layer_ensemble_predictions
    ) -> List[np.ndarray]:
        layer_weights = [weight for weight in self.ensembles[stacking_layer].weights_ if weight > 0]
        layer_size = self.ensembles[stacking_layer].ensemble_size
        ensemble_predictions = []
        for weight, pred in zip(layer_weights, raw_stacking_layer_ensemble_predictions):
            ensemble_predictions.extend([pred] * int(weight * layer_size))
        return ensemble_predictions

    def get_layer_stacking_ensemble_predictions(
        self,
        stacking_layer: int,
        dataset: str = 'ensemble'
    ) -> List[Optional[np.ndarray]]:
        raw_stacking_layer_ensemble_predictions = self.ensemble_predictions[stacking_layer][dataset]

        return self.get_expanded_layer_stacking_ensemble_predictions(stacking_layer=stacking_layer, raw_stacking_layer_ensemble_predictions=raw_stacking_layer_ensemble_predictions)
