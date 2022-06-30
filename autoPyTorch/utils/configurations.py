from typing import Union, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner import get_available_traditional_learners
from autoPyTorch.pipeline.traditional_tabular_classification import TraditionalTabularClassificationPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


HYPERPARAMETERS = {
    "catboost": ['default'],
    "lgb": ["extra_trees", 'default'],
    "knn": ["uniform", "distance"],
    "extra_trees": ["gini", "entropy"],
    "xgboost": ['default'],
    "random_forest": ["gini", "entropy"]
}


def get_configuration(learner: str, hyperparameter, dataset_properties, random_state):
    pipeline = TraditionalTabularClassificationPipeline(dataset_properties=dataset_properties,
                                                     random_state=random_state,
                                                     search_space_updates=_get_traditional_search_space_updates(learner, hyperparameter=hyperparameter))
    return pipeline.config_space.get_default_configuration()


def get_traditional_learners_configurations(random_state, dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None):
    traditional_learners = get_available_traditional_learners(dataset_properties=dataset_properties)
    configurations = []
    for learner in traditional_learners.keys():
        for hyperparameter in HYPERPARAMETERS[learner]:
            configurations.append(get_configuration(
                learner=learner,
                hyperparameter=hyperparameter,
                dataset_properties=dataset_properties,
                random_state=random_state
            ))
    return configurations

def _get_traditional_search_space_updates(learner: str, hyperparameter):
    updates = HyperparameterSearchSpaceUpdates()
    updates.append(node_name='model_trainer', hyperparameter='traditional_learner', value_range=(learner,), default_value=learner)
    if hyperparameter == 'default':
        return updates
    elif learner == 'lgb':
        updates.append(node_name='model_trainer', hyperparameter=f'lgb:{hyperparameter}', value_range=(True,), default_value=True)
    elif learner == 'random_forest':
        updates.append(node_name='model_trainer', hyperparameter='random_forest:criterion', value_range=(hyperparameter,), default_value=hyperparameter)
    elif learner == 'extra_trees':
        updates.append(node_name='model_trainer', hyperparameter='extra_trees:criterion', value_range=(hyperparameter,), default_value=hyperparameter)
    elif learner == 'knn':
        updates.append(node_name='model_trainer', hyperparameter='knn:weights', value_range=(hyperparameter,), default_value=hyperparameter)
    return updates


def is_configuration_traditional(configuration: Union[Configuration, dict]):
    if 'model_trainer:traditional_learner' in configuration:
        return True
    else:
        return False

def get_traditional_search_space_updates(config: Union[Dict, Configuration]):
    config = config.get_dictionary() if isinstance(config, Configuration) else config
    learner = config['model_trainer:traditional_learner']
    if learner in ['catboost', 'xgboost']:
        hyperparameter = 'default'
    elif learner == 'lgb':
        if config['model_trainer:lgb:extra_trees']:
            hyperparameter = 'extra_trees'
        else:
            hyperparameter = 'default'
    elif learner == 'random_forest':
        hyperparameter = config['model_trainer:random_forest:criterion']
    elif learner == 'extra_trees':
        hyperparameter = config['model_trainer:extra_trees:criterion']
    elif learner == 'knn':
        hyperparameter = config['model_trainer:knn:weights']
    return _get_traditional_search_space_updates(learner, hyperparameter)


def get_traditional_config_space(config, dataset_properties, random_state) -> ConfigurationSpace:
    pipeline = TraditionalTabularClassificationPipeline(dataset_properties=dataset_properties,
                                                     random_state=random_state,
                                                     search_space_updates=get_traditional_search_space_updates(config))
    return pipeline.config_space
