from typing import Any, Dict, List, Optional, Tuple, Type, Union
import re

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.base_model import BaseModelComponent
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import (
    BaseTraditionalLearner, get_available_traditional_learners)
from autoPyTorch.utils.common import HyperparameterSearchSpace


# TODO: Make this a choice and individual components for each traditional classifier
class TabularTraditionalModel(BaseModelComponent):
    """
    Implementation of a dynamic model, that consists of a learner and a head
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            **kwargs: Any
    ):
        super().__init__(
            random_state=random_state,
        )
        self.config = kwargs
        self._traditional_learners = get_available_traditional_learners()

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            "shortname": "TabularTraditionalModel",
            "name": "Tabular Traditional Model",
        }

    
    def get_hyperparameter_search_space(self, dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        **kwargs: Any) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        available_traditional_learners: Dict[str, Type[BaseTraditionalLearner]] = get_available_traditional_learners()
        # Remove knn if data is all categorical

        if dataset_properties is not None:
            numerical_columns = dataset_properties['numerical_columns'] \
                if isinstance(dataset_properties['numerical_columns'], List) else []
            if len(numerical_columns) == 0:
                del available_traditional_learners['knn']

        updates = self._get_search_space_updates()

        if 'traditional_learner' in updates:
            learner_hp = CategoricalHyperparameter("traditional_learner", choices=updates['traditional_learner'].value_range)
        else:
            learner_hp = CategoricalHyperparameter("traditional_learner", choices=available_traditional_learners.keys())
        cs.add_hyperparameters([learner_hp])

        for name in learner_hp.choices:
            child_updates = self._get_child_search_space_updates(prefix=name)
            model_configuration_space = available_traditional_learners[name]. \
                get_hyperparameter_search_space(dataset_properties, **child_updates)
            parent_hyperparameter = {'parent': learner_hp, 'value': name}
            cs.add_configuration_space(
                name,
                model_configuration_space,
                parent_hyperparameter=parent_hyperparameter
            )

        return cs

    def _get_child_search_space_updates(self, prefix: Optional[str] = None) -> Dict[str, HyperparameterSearchSpace]:
        """Get the search space updates with the given prefix

        Args:
            prefix (str):
                Only return search space updates with given prefix (default: {None})

        Returns:
            Dict[str, HyperparameterSearchSpace]:
                Mapping of search space updates. Keys don't contain the prefix.
        """

        result: Dict[str, HyperparameterSearchSpace] = dict()

        # iterate over all search space updates of this node and keep the ones that have the given prefix
        for key in self._cs_updates.keys():
            if prefix is None:
                result[key] = self._cs_updates[key].get_search_space()
            elif re.search(f'^{prefix}', key) is not None:
                result[key[len(prefix) + 1:]] = self._cs_updates[key].get_search_space(remove_prefix=prefix)
        return result

    def build_model(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                    dataset_properties: Dict[str, BaseDatasetPropertiesType],
                    logger_port: int, task_type: str, output_type: str, optimize_metric: Optional[str] = None,
                    time_limit: Optional[int] = None,
                    ) -> BaseTraditionalLearner:
        """
        This method returns a traditional learner, that is dynamically
        built using a self.config that is model specific, and contains
        the additional configuration hyperparameters to build a domain
        specific model
        """
        learner_name = self.config.pop("traditional_learner")
        Learner = self._traditional_learners[learner_name]

        config = self._remove_prefix_config(learner_name=learner_name)
        learner = Learner(random_state=self.random_state, logger_port=logger_port,
                          task_type=task_type, output_type=output_type, optimize_metric=optimize_metric,
                          dataset_properties=dataset_properties, time_limit=time_limit, **config)

        return learner

    def _remove_prefix_config(self, learner_name):
        return {key.replace(f'{learner_name}:', ''): value for key, value in self.config.items()}

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        return f"TabularTraditionalModel: {self.model.name if self.model is not None else None}"
