from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


def get_autogluon_default_nn_config(feat_type):
    has_numerical_features = "numerical" in feat_type
    has_cat_features = "categorical" in feat_type
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
