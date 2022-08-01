from typing import Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import NetworkBackboneComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter


def get_search_space_from_kwargs(original_search_Space, kwargs, hp_name):
        update = kwargs.get(hp_name, None)
        if update is not None:
            updated_search_space = HyperparameterSearchSpace(hyperparameter=hp_name,
                    value_range=update.value_range, default_value=update.default_value)
        else:
            updated_search_space = HyperparameterSearchSpace(hyperparameter=hp_name,
                                                                value_range=original_search_Space.value_range,
                                                                default_value=original_search_Space.default_value,
                                                                log=original_search_Space.log)
                                                            
        return updated_search_space


class MLPBackbone(NetworkBackboneComponent):
    """
    This component automatically creates a Multi Layer Perceptron based on a given config.

    This MLP allows for:
        - Different number of layers
        - Specifying the activation. But this activation is shared among layers
        - Using or not dropout
        - Specifying the number of units per layers
    """

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        layers: List[nn.Module] = list()
        in_features = input_shape[0]

        self._add_layer(layers, in_features, self.config['num_units_1'], 1)

        for i in range(2, self.config['num_groups'] + 1):
            self._add_layer(layers, self.config["num_units_%d" % (i - 1)],
                            self.config["num_units_%d" % i], i)
        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.config["num_units_%d" % self.config["num_groups"]],)

    def _add_layer(self, layers: List[nn.Module], in_features: int, out_features: int,
                   layer_id: int) -> None:
        """
        Dynamically add a layer given the in->out specification

        Args:
            layers (List[nn.Module]): The list where all modules are added
            in_features (int): input dimensionality of the new layer
            out_features (int): output dimensionality of the new layer

        """
        layers.append(nn.Linear(in_features, out_features))
        if self.config['use_batch_norm']:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(_activations[self.config["activation"]]())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.config["dropout_%d" % layer_id]))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MLPBackbone',
            'name': 'MLPBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_groups: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_groups",
                                                                          value_range=(1, 15),
                                                                          default_value=5,
                                                                          ),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0],
                                                                          ),
        use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_dropout",
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        use_batch_norm: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="use_batch_norm",
                                                                           value_range=(True, False),
                                                                           default_value=False,
                                                                           ),
        num_units: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_units",
                                                                         value_range=(10, 1024),
                                                                         default_value=200,
                                                                         log=True
                                                                         ),
        dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dropout",
                                                                       value_range=(0, 0.8),
                                                                       default_value=0.5,
                                                                       ),
        **kwargs
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # The number of hidden layers the network will have.
        # Layer blocks are meant to have the same architecture, differing only
        # by the number of units
        if len(num_groups.value_range) > 1:
            min_mlp_layers, max_mlp_layers = num_groups.value_range
        else:
            min_mlp_layers = num_groups.value_range[-1]
            max_mlp_layers = num_groups.value_range[-1]

        num_groups = get_hyperparameter(num_groups, UniformIntegerHyperparameter)
        add_hyperparameter(cs, activation, CategoricalHyperparameter)

        # whether to use batch normalization
        add_hyperparameter(cs, use_batch_norm, CategoricalHyperparameter)

        # We can have dropout in the network for
        # better generalization
        dropout_flag = False
        if any(use_dropout.value_range):
            dropout_flag = True

        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        cs.add_hyperparameters([num_groups, use_dropout])

        for i in range(1, int(max_mlp_layers) + 1):
            n_units_hp_name = 'num_units_%d' % i
            n_units_search_space = get_search_space_from_kwargs(num_units, kwargs, n_units_hp_name)
            n_units_hp = get_hyperparameter(n_units_search_space, UniformIntegerHyperparameter)

            cs.add_hyperparameter(n_units_hp)

            if i > int(min_mlp_layers):
                # The units of layer i should only exist
                # if there are at least i layers
                cs.add_condition(
                    CS.GreaterThanCondition(
                        n_units_hp, num_groups, i - 1
                    )
                )
            if dropout_flag:
                dropout_search_space = get_search_space_from_kwargs(dropout, kwargs, 'dropout_%d' % i)

                dropout_hp = get_hyperparameter(dropout_search_space, UniformFloatHyperparameter)
                cs.add_hyperparameter(dropout_hp)

                dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout, True)

                if i > int(min_mlp_layers):
                    dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_groups, i - 1)
                    cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
                else:
                    cs.add_condition(dropout_condition_1)

        return cs
