from collections import Counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss


# TODO: for now we can use this and pass this to stacking evaluator.
# TODO: This can be achieved by using `backend.load_ensemble`
# TODO: it loads the last stored ensemble. So we have access to it.
# TODO: the ensemble is a pickled file containing the fitted ensemble of this class.
# TODO: Think of functionality of the functions in this class adjusted for stacking.
class StackingEnsemble(AbstractEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        metric: autoPyTorchMetric,
        task_type: int,
        random_state: np.random.RandomState,
        ensemble_slot_j: int
    ) -> None:
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.random_state = random_state
        self.task_type = task_type
        self.ensemble_slot_j = ensemble_slot_j

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        self.metric = None  # type: ignore
        return self.__dict__

    def fit(
        self,
        predictions_ensemble: List[np.ndarray],
        best_model_predictions: np.ndarray,
        labels: np.ndarray,
        ensemble_identifiers: List[Tuple[int, int, float]],
        best_model_identifier: Tuple[int, int, float],
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
        predictions_ensemble[self.ensemble_slot_j] = best_model_predictions
        ensemble_identifiers[self.ensemble_slot_j] = best_model_identifier
        self._fit(predictions_ensemble, labels)
        self.identifiers_ = ensemble_identifiers
        self._calculate_weights()
        return self

    # TODO: fit a stacked ensemble.
    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """
        Implemenation of Lévesque et al.

        For more details, please check the paper
        "Bayesian hyperparameter optimization for ensemble learning" by Lévesque (2004)

        Args:
            predictions (List[np.ndarray]):
                A list of individual model predictions of shape (n_datapoints, n_targets)
                corresponding to the OutOfFold estimate of the ground truth
            identifiers (List[Tuple[int, int, float]]):
                A list of model identifiers, each with the form
                (seed, number of run, budget)
        """

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )

        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )

        nonnull_predictions = [pred for pred in predictions if pred is not None]
        size = len(nonnull_predictions)
        for pred in nonnull_predictions:
            np.add(
                weighted_ensemble_prediction,
                pred,
                out=fant_ensemble_prediction
            )
            np.multiply(
                fant_ensemble_prediction,
                (1. / float(size)),
                out=fant_ensemble_prediction
            )
        
        # Calculate loss is versatile and can return a dict of slosses
        loss = calculate_loss(
            metrics=[self.metric],
            target=labels,
            prediction=fant_ensemble_prediction,
            task_type=self.task_type,
        )[self.metric.name]

        # store list of preds for later use
        self.ensemble_predictions = predictions

        self.train_loss_: float = loss

    # TODO: return 1 for models in layer 0, 2 for next and so on
    # TODO: 0 for models that are not in stack
    def _calculate_weights(self) -> None:
        """
        Calculates the contribution each of the individual models
        should have, in the final ensemble soft voting. It does so by
        a frequency counting scheme. In particular, how many times a model
        was used during hill climbing optimization.
        """
        weights = np.zeros(
            self.ensemble_size,
            dtype=np.float64,
        )
        current_size = len([id for id in self.identifiers_ if id is not None])
        for i, identifier in enumerate(self.identifiers_):
            if identifier is not None:
                weights[i] = (1. / float(current_size))

        self.weights_ = weights

    # TODO: Adjust this to use weights and make 
    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        return self._predict(predictions, self.weights_)

    def _predict(self, predictions, weights):
        """
        Given a list of predictions from the individual model, this method
        aggregates the predictions using a soft voting scheme with the weights
        found during training.

        Args:
            predictions (List[np.ndarray]):
                A list of predictions from the individual base models.

        Returns:
            average (np.ndarray): Soft voting predictions of ensemble models, using
                                the weights
        """

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        if len(predictions) == np.count_nonzero(weights):
            non_null_weights = [w for w in weights if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average

    # def __str__(self) -> str:
    #     return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
    #            '\n\tWeights: %s\n\tIdentifiers: %s' % \
    #            (' '.join(['%d: %5f' % (idx, performance)
    #                      for idx, performance in enumerate(self.trajectory_)]),
    #             self.indices_, self.weights_,
    #             ' '.join([str(identifier) for idx, identifier in
    #                       enumerate(self.identifiers_)
    #                       if self.weights_[idx] > 0]))

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        """
        After training of ensemble selection, not all models will be used.
        Some of them will have zero weight. This procedure filters this models
        out.

        Returns:
            output (List[Tuple[int, int, float]]):
                The models actually used by ensemble selection
        """
        return self.identifiers_

    def get_validation_performance(self) -> float:
        """
        Returns the best optimization performance seen during hill climbing

        Returns:
            (float):
                best ensemble training performance
        """
        return self.train_loss_

    def predict_with_current_pipeline(
        self,
        pipeline_predictions: np.ndarray,
    ) -> np.ndarray:
        """
        predict with ensemble by replacing model at j = self.iteration mod m,
        where m is ensemble_size.
        returns ensemble predictions
        """
        predictions = self.ensemble_predictions.copy()
        if predictions[self.ensemble_slot_j] is None:
            total_predictions = len([pred for pred in predictions if pred is not None])
            total_predictions += 1
            weights = [1/total_predictions for pred in predictions if pred is not None]
        else:
            weights = self.weights_

        predictions[self.ensemble_slot_j] = pipeline_predictions
        return self._predict(predictions, weights)
