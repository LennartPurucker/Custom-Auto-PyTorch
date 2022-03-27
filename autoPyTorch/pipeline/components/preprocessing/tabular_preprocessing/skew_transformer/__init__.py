import os
from collections import OrderedDict
from typing import Dict, List, Optional

import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.skew_transformer.base_skew_transformer import BaseSkewTransformer

skew_transforming_directory = os.path.split(__file__)[0]
_skew_transformers = find_components(__package__,
                           skew_transforming_directory,
                           BaseSkewTransformer)

_addons = ThirdPartyComponents(BaseSkewTransformer)


def add_skew_transformer(skew_transformer: BaseSkewTransformer) -> None:
    _addons.add_component(skew_transformer)


class SkewTransformerChoice(autoPyTorchChoice):
    """
    Allows for dynamically choosing skew_transforming component at runtime
    """

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available skew_transformer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all BaseSkewTransformers components available
                as choices for skew_transforming
        """
        components = OrderedDict()
        components.update(_skew_transformers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                                        default: Optional[str] = None,
                                        include: Optional[List[str]] = None,
                                        exclude: Optional[List[str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()

        dataset_properties = {**self.dataset_properties, **dataset_properties}

        available_skew_transformers = self.get_available_components(dataset_properties=dataset_properties,
                                                          include=include,
                                                          exclude=exclude)

        if len(available_skew_transformers) == 0:
            raise ValueError("no skew_transformers found, please add a skew_transformer")

        if default is None:
            defaults = [
                'PowerTransformer',
                'QuantileTransformer',
                'NoSkewTransformer'
            ]
            for default_ in defaults:
                if default_ in available_skew_transformers:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        skew_columns = dataset_properties['skew_columns']\
            if isinstance(dataset_properties['skew_columns'], List) else []
        updates = self._get_search_space_updates()
        if '__choice__' in updates.keys():
            choice_hyperparameter = updates['__choice__']
            if not set(choice_hyperparameter.value_range).issubset(available_skew_transformers):
                raise ValueError("Expected given update for {} to have "
                                 "choices in {} got {}".format(self.__class__.__name__,
                                                               available_skew_transformers,
                                                               choice_hyperparameter.value_range))
            if len(skew_columns) == 0:
                assert len(choice_hyperparameter.value_range) == 1
                if 'NoSkewTransformer' not in choice_hyperparameter.value_range:
                    raise ValueError("Provided {} in choices, however, the dataset "
                                     "is incompatible with it".format(choice_hyperparameter.value_range))

            preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                         choice_hyperparameter.value_range,
                                                         default_value=choice_hyperparameter.default_value)
        else:
            # add only no skew_transformer to choice hyperparameters in case the dataset is only categorical
            if len(skew_columns) == 0:
                default = 'NoSkewTransformer'
                if include is not None and default not in include:
                    raise ValueError("Provided {} in include, however, "
                                     "the dataset is incompatible with it".format(include))
                preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                             ['NoSkewTransformer'],
                                                             default_value=default)
            else:
                preprocessor = CSH.CategoricalHyperparameter('__choice__',
                                                             list(available_skew_transformers.keys()),
                                                             default_value=default)
        cs.add_hyperparameter(preprocessor)

        # add only child hyperparameters of preprocessor choices
        for name in preprocessor.choices:
            updates = self._get_search_space_updates(prefix=name)
            config_space = available_skew_transformers[name].get_hyperparameter_search_space(dataset_properties,  # type:ignore
                                                                                   **updates)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, config_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _check_dataset_properties(self, dataset_properties: Dict[str, BaseDatasetPropertiesType]) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.
        Args:
            dataset_properties:

        """
        super()._check_dataset_properties(dataset_properties)
        assert 'skew_columns' in dataset_properties.keys() and \
               'categorical_columns' in dataset_properties.keys(), \
            "Dataset properties must contain information about the type of columns"
