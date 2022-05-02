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

# TODO: Think of functionality of the functions in this class adjusted for stacking.
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

    # def _load_ensemble_for_layer(self, backend: Backend, seed, layer):
    #     path = backend.get_ensemble_dir()
    #     cur_layer_ensembles = glob.glob(f"{path}/{seed}.{layer}*.ensemble").sort()
    #     with open(cur_layer_ensembles[-1], "rb") as fh:
    #         latest_ensemble_for_layer = cast(AbstractEnsemble, pickle.load(fh))
    #     return latest_ensemble_for_layer

    # TODO: return 1 for models in layer 0, 2 for next and so on
    # TODO: 0 for models that are not in stack
    # def _calculate_weights(self) -> None:
    #     """
    #     Calculates the contribution each of the individual models
    #     should have, in the final ensemble soft voting. It does so by
    #     a frequency counting scheme. In particular, how many times a model
    #     was used during hill climbing optimization.
    #     """
    #     weights = np.zeros(
    #         self.ensemble_size,
    #         dtype=np.float64,
    #     )
    #     current_size = len([id for id in self.identifiers_ if id is not None])
    #     for i, identifier in enumerate(self.identifiers_):
    #         if identifier is not None:
    #             weights[i] = (1. / float(current_size))

    #     self.weights_ = weights

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        # should be the last layer
        return self.ensembles[self.cur_stacking_layer].predict(predictions)
        # last_ensemble = [ensemble for ensemble in self.ensembles if ensemble is not None][-1]
        # return last_ensemble.predict(predictions)

    # def _predict(self, predictions, weights):
    #     """
    #     Given a list of predictions from the individual model, this method
    #     aggregates the predictions using a soft voting scheme with the weights
    #     found during training.

    #     Args:
    #         predictions (List[np.ndarray]):
    #             A list of predictions from the individual base models.

    #     Returns:
    #         average (np.ndarray): Soft voting predictions of ensemble models, using
    #                             the weights
    #     """

    #     average = np.zeros_like(predictions[0], dtype=np.float64)
    #     tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

    #     # if prediction model.shape[0] == len(non_null_weights),
    #     # predictions do not include those of zero-weight models.
    #     if len([pred for pred in predictions if pred is not None]) == np.count_nonzero(weights):
    #         for pred, weight in zip(predictions, weights):
    #             if pred is not None:
    #                 np.multiply(pred, weight, out=tmp_predictions)
    #                 np.add(average, tmp_predictions, out=average)

    #     # If none of the above applies, then something must have gone wrong.
    #     else:
    #         raise ValueError(f"{len(predictions)}, {self.weights_}\n"
    #                          f"The dimensions of non null ensemble predictions"
    #                          f" and ensemble weights do not match!")
    #     del tmp_predictions
    #     return average

    def __str__(self) -> str:
        return f"Ensemble Selection:\n\tWeights: {self.weights_}\
            \n\tIdentifiers: {' '.join([str(identifier) for idx, identifier in enumerate(self.identifiers_) if self.weights_[idx] > 0])}"

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

    def get_layer_stacking_ensemble_predictions(
        self,
        stacking_layer: int,
        dataset: str = 'ensemble'
    ) -> List[Optional[np.ndarray]]:

        return self.ensemble_predictions[stacking_layer][dataset]
