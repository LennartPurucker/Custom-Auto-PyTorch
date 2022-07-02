from collections import OrderedDict
import json
from typing import Dict, Union, OrderedDict as OrderedDict_typing
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

from smac.runhistory.runhistory import DataOrigin, RunHistory, RunInfo, RunValue, EnumEncoder, RunKey

from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


def get_autogluon_default_nn_config(feat_types):
    has_numerical_features = "numerical" in feat_types
    has_cat_features = "categorical" in feat_types
    search_space_updates = HyperparameterSearchSpaceUpdates()


    # architecture head
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='__choice__',
        value_range=['no_head'],
        default_value='no_head',
    )
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='no_head:activation',
        value_range=['relu', 'elu'],
        default_value='relu',
    )

    # backbone architecture
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['MLPBackbone'],
        default_value='MLPBackbone',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:num_groups',
        value_range=(2, 4),
        default_value=4,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:num_units',
        value_range=[128, 512],
        default_value=128,
        log=True
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:dropout',
        value_range=(0.1, 0.5),
        default_value=0.1,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:use_dropout',
        value_range=(True, False),
        default_value=True,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:use_batch_norm',
        value_range=(True, False),
        default_value=True,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='MLPBackbone:activation',
        value_range=['relu', 'elu'],
        default_value='relu',
    )

    # training updates
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['NoScheduler'],
        default_value='NoScheduler',
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamOptimizer', 'SGDOptimizer'],
        default_value='AdamOptimizer',
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:lr',
        value_range=[1e-4, 3e-2],
        default_value=3e-4,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamOptimizer:weight_decay',
        value_range=(1E-12, 0.1),
        default_value=1e-6,
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='max_batch_size',
        value_range=[512],
        default_value=512,
    )

    # preprocessing
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='__choice__',
        value_range=['NoFeaturePreprocessor'],
        default_value='NoFeaturePreprocessor',
    )

    if has_numerical_features:
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='numerical_strategy',
            value_range=['median', 'mean', 'most_frequent'],
            default_value='median',
        )
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )
        # preprocessing
        search_space_updates.append(
            node_name='skew_transformer',
            hyperparameter='__choice__',
            value_range=['QuantileTransformer'],
            default_value='QuantileTransformer',
        )

    if has_cat_features:
        search_space_updates.append(
            node_name='encoder',
            hyperparameter='__choice__',
            value_range=['OneHotEncoder', 'NoEncoder'],
            default_value='OneHotEncoder',
        )
        search_space_updates.append(
            node_name="network_embedding",
            hyperparameter="__choice__",
            value_range=('NoEmbedding', 'LearnedEntityEmbedding'),
            default_value='LearnedEntityEmbedding'
        )

    return search_space_updates


def get_config_from_run_history(run_history: Union[RunHistory, OrderedDict], num_run: int):
    data = run_history.data if isinstance(run_history, RunHistory) else run_history

    for _, run_value in data.items():
        if run_value.additional_info.get('num_run', -1) == num_run:  # to ensure that unsuccessful configs are not returned
            return run_value.additional_info['configuration']
    
def update_run_history_with_max_config_id(run_history_data: OrderedDict_typing[RunKey, RunValue], ids_config: Dict[int, Configuration], max_run_history_config_id: int):

    updated_run_history: OrderedDict_typing[RunKey, RunValue] = OrderedDict()
    updated_ids_config: Dict[int, Configuration] = {}
    for run_key, run_value in run_history_data.items():

        configuration = ids_config[run_key.config_id]
        new_config_id = run_key.config_id+max_run_history_config_id
        configuration.config_id = new_config_id

        updated_ids_config[new_config_id] = configuration
        updated_run_history.update({
            RunKey(config_id=new_config_id, instance_id=run_key.instance_id, seed=run_key.seed, budget=run_key.budget): \
                RunValue(run_value.cost, run_value.time, run_value.status, run_value.starttime, run_value.endtime, run_value.additional_info)
        })

    return updated_run_history, updated_ids_config

def save_run_history(full_run_history: Dict, full_ids_config: Dict, path_run_history):
        data = [
            (
                [
                    int(k.config_id),
                    str(k.instance_id) if k.instance_id is not None else None,
                    int(k.seed),
                    float(k.budget) if k[3] is not None else 0,
                ],
                [v.cost, v.time, v.status, v.starttime, v.endtime, v.additional_info],
            )
            for k, v in full_run_history.items()
        ]
        config_ids_to_serialize = set([entry[0][0] for entry in data])
        configs = {
            id_: conf.get_dictionary() for id_, conf in full_ids_config.items() if id_ in config_ids_to_serialize
        }
        config_origins = {
            id_: conf.origin
            for id_, conf in full_ids_config.items()
            if (id_ in config_ids_to_serialize and conf.origin is not None)
        }

        with open(path_run_history, "w") as fp:
            json.dump(
                {"data": data, "config_origins": config_origins, "configs": configs},
                fp,
                cls=EnumEncoder,
                indent=2,
            )
