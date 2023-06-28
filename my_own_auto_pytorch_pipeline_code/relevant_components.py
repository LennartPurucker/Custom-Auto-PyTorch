"""Try to build a hardcoded non autopyotrch version of reg cocktails"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, List
import torchvision

from torch import nn
import torch
from torch.utils.data import Dataset, Subset

from torch.optim import AdamW, swa_utils
from my_own_auto_pytorch_pipeline_code.utils import get_output_shape, get_shaped_neuron_counts, ResBlock, custom_collate_fn
from my_own_auto_pytorch_pipeline_code.trainer_stuff.StandardTrainer import StandardTrainer

_activations = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid
}

# --- No Head
"""
Head which only adds a fully connected layer which takes the
output of the backbone as input and outputs the predictions.
Flattens any input in a array of shape [B, prod(input_shape)].

https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L193
no_head:activation is missing but does not seem to have any impact here!

"""


def build_head(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
    layers = []
    # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L187
    in_features = np.prod(input_shape).item()
    out_features = np.prod(output_shape).item()
    layers.append(nn.Linear(in_features=in_features,
                            out_features=out_features))
    return nn.Sequential(*layers)


def _add_group(config, in_features: int, out_features: int,
               blocks_per_group: int, last_block_index: int, dropout: Optional[float]
               ) -> nn.Module:
    """
    Adds a group into the main backbone.
    In the case of ResNet a group is a set of blocks_per_group
    ResBlocks

    Args:
        in_features (int): number of inputs for the current block
        out_features (int): output dimensionality for the current block
        blocks_per_group (int): Number of ResNet per group
        last_block_index (int): block index for shake regularization
        dropout (None, float): dropout value for the group. If none,
            no dropout is applied.
    """
    blocks = list()
    for i in range(blocks_per_group):
        blocks.append(
            ResBlock(
                config=config,
                in_features=in_features,
                out_features=out_features,
                blocks_per_group=blocks_per_group,
                block_index=last_block_index + i,
                dropout=dropout,
                activation=_activations[config["activation"]]
            )
        )
        in_features = out_features
    return nn.Sequential(*blocks)


def build_backbone(input_shape: Tuple[int, ...], config) -> torch.nn.Sequential:
    num_groups = config["num_groups"]
    blocks_per_group = config["blocks_per_group"]

    layers: List[torch.nn.Module] = list()
    in_features = input_shape[0]
    out_features = config["out_features"]

    # use the get_shaped_neuron_counts to update the number of units
    neuron_counts = get_shaped_neuron_counts(
        shape=config['resnet_shape'],
        in_feat=in_features,
        out_feat=out_features,
        max_neurons=config['max_units'],
        layer_count=num_groups + 2,
    )[:-1]
    num_units_config = {"num_units_%d" % (i): num for i, num in enumerate(neuron_counts)}

    if config["use_dropout"]:
        # the last dropout ("neuron") value is skipped since it will be equal
        # to output_feat, which is 0. This is also skipped when getting the
        # n_units for the architecture, since, it is mostly implemented for the
        # output layer, which is part of the head and not of the backbone.
        dropout_shape = get_shaped_neuron_counts(
            shape=config['dropout_shape'],
            in_feat=0,
            out_feat=0,
            max_neurons=config["max_dropout"],
            layer_count=num_groups + 1,
        )[:-1]
        dropout_num_units_config = {"dropout_%d" % (i + 1): dropout for i, dropout in enumerate(dropout_shape)}

    layers.append(torch.nn.Linear(in_features, num_units_config["num_units_0"]))

    # build num_groups-1 groups each consisting of blocks_per_group ResBlocks
    # the output features of each group is defined by num_units_i
    for i in range(1, num_groups + 1):
        layers.append(
            _add_group(
                config=config,
                in_features=num_units_config["num_units_%d" % (i - 1)],
                out_features=num_units_config["num_units_%d" % i],
                blocks_per_group=blocks_per_group,
                last_block_index=(i - 1) * blocks_per_group,
                dropout=dropout_num_units_config[f'dropout_{i}'] if config['use_dropout'] else None
            )
        )

    if config['use_batch_norm']:
        layers.append(torch.nn.BatchNorm1d(num_units_config["num_units_%i" % num_groups]))
    layers.append(_activations[config["activation"]]())
    backbone = torch.nn.Sequential(*layers)
    return backbone


def build_optimizer(network, config):
    # Hardcoded to AdamWOptimizer https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L263

    return AdamW(
        params=network.parameters(),
        lr=config["AdamWOptimizer:lr"],
        betas=(config["AdamWOptimizer:beta1"], config["AdamWOptimizer:beta2"]),
        weight_decay=config["AdamWOptimizer:weight_decay"] if config["AdamWOptimizer:use_weight_decay"] else 0.0,

    )


def build_lr_scheduler(optimizer, config):
    # Hardcoded to CosineAnnealingWarmRestarts - https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L251

    # initialise required attributes for the scheduler
    T_mult: int = 2
    # using Epochs = T_0 * (T_mul ** n_restarts -1) / (T_mul - 1) (Sum of GP)
    T_0: int = max((config['epochs'] * (T_mult - 1)) // (T_mult ** config["CosineAnnealingWarmRestarts:n_restarts"] - 1), 1)

    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=int(T_0),
        T_mult=int(T_mult),
    )


def build_final_activation(is_multiclass):
    if is_multiclass:
        final_activation = nn.Softmax(dim=1)
    else:
        final_activation = nn.Sigmoid()

    return final_activation


def build_data_loader(X, y, config):
    class BaseDataset(Dataset):
        def __init__(self, train_tensors):
            self.train_tensors = train_tensors
            self.input_shape: Tuple[int] = self.train_tensors[0].shape[1:]
            n_classes = len(np.unique(train_tensors[1]))
            self.output_shape = n_classes if n_classes > 2 else 1

        def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
            """
            The base dataset uses a Subset of the data. Nevertheless, the base dataset expects
            both validation and test data to be present in the same dataset, which motivates
            the need to dynamically give train/test data with the __getitem__ command.

            This method yields a datapoint of the whole data (after a Subset has selected a given
            item, based on the resampling strategy) and applies a train/testing transformation, if any.

            Args:
                index (int): what element to yield from all the train/test tensors

            Returns:
                A transformed single point prediction
            """

            X = self.train_tensors[0].iloc[[index]] if hasattr(self.train_tensors[0], "iloc") \
                else self.train_tensors[0][index]

            X = torchvision.transforms.Compose([torch.from_numpy])(X)
            # In case of prediction, the targets are not provided
            Y = self.train_tensors[1][index] if self.train_tensors[1] is not None else None
            return X, Y

        def __len__(self) -> int:
            return int(self.train_tensors[0].shape[0])
    bd = BaseDataset(train_tensors=(X, y))
    return torch.utils.data.DataLoader(
        bd,
        batch_size=min(config["batch_size"], len(X)),
        shuffle=False,
        collate_fn=custom_collate_fn,
    ), bd


def build_trainer(seed, config):
    # Hardcoded to one default trainer, and hence use the default - https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L370

    lookahead_config = dict()
    if config["StandardTrainer:use_lookahead_optimizer"]:
        from my_own_auto_pytorch_pipeline_code.trainer_stuff.utils import Lookahead
        lookahead_config[f"{Lookahead.__name__}:la_steps"] = config["lookahead_optimizer:la_steps"]
        lookahead_config[f"{Lookahead.__name__}:la_alpha"] = config["lookahead_optimizer:la_alpha"]

    return StandardTrainer(
        random_state=seed,  # TODO: transform into random state in code later
        weighted_loss=config["StandardTrainer:weighted_loss"],
        use_stochastic_weight_averaging=config["StandardTrainer:use_stochastic_weight_averaging"],
        use_snapshot_ensemble=config["StandardTrainer:use_snapshot_ensemble"],
        use_lookahead_optimizer=config["StandardTrainer:use_lookahead_optimizer"],
        **lookahead_config
    )


# --- Config parameter
def _get_config():
    return {
        # Constants
        "num_groups": [2, ],  # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L213
        "blocks_per_group": [2, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L219
        "out_features": [512, ],  # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L225
        "resnet_shape": ["brick", ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L207
        "max_units": [512, ],  # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L231
        "activation": ["relu", ],  # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L237

        "use_batch_norm": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L341 and paper

        "use_dropout": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L335 and paper
        # + on condition:  use_dropout = True
        "dropout_shape": ['funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'],
        # paper and resnet code
        "max_dropout": (0.0, 0.8),  # paper and resnet code; paper name := drop rate

        "use_skip_connection": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L347 and paper; paper name := SC
        # + on condition: use_skip_connection = True
        "multi_branch_choice": ['shake-drop', 'shake-shake', 'None'],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L354C37-L354C46, paper, and resent code
        # on condition: use_skip_connection = True and multi_branch_choice = shake-shake or shake-drop
        "shake_shake_update_func": ['shake-shake', 'shake-even', 'even-even', 'M3'],
        # unclear from code and paper; likely shake-shake default used for mb=shake-shake but unclear for mb=shake-drop; keep it as an HP like this following resnet code
        # + on condition: use_skip_connection = True and multi_branch_choice = shake-drop
        "max_shake_drop_probability": (0.0, 1.0),  # paper and reset code

        # + on condition: optimizer is AdamWOptimizer (which is hardcoded by default)
        "AdamWOptimizer:lr": [1e-3, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L269
        "AdamWOptimizer:beta1": [0.9, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L318
        "AdamWOptimizer:beta2": [0.999, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L324

        "AdamWOptimizer:use_weight_decay": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L363
        # + on condition: AdamWOptimizer:use_weight_decay = True
        "AdamWOptimizer:weight_decay": (1E-5, 0.1),  # paper and resnet code

        # + on condition: lr_scheduler is CosineAnnealingWarmRestarts (which is hardcoded by default)
        "CosineAnnealingWarmRestarts:n_restarts": [3, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L257

        # Training parameter
        "epochs": [105, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/cocktails/main_experiment.py#L69 and paper
        "batch_size": [128, ],  # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L275

        "StandardTrainer:use_snapshot_ensemble": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L397
        "StandardTrainer:weighted_loss": [1, ],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L379
        "StandardTrainer:use_stochastic_weight_averaging": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L397 and paper
        "StandardTrainer:use_lookahead_optimizer": [True, False],
        # https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L385 and paper
        # + on condition: StandardTrainer:use_lookahead_optimizer = True
        "lookahead_optimizer:la_steps": (5, 10),  # paper and lookahead code
        "lookahead_optimizer:la_alpha": (0.5, 0.8),  # paper and lookahead code
    }


# --- Build it
def _build_and_train_network(X, y, config, seed, device="cpu"):
    data_loader, base_dataset = build_data_loader(X, y, config)
    input_shape = base_dataset.input_shape
    output_shape = base_dataset.output_shape
    is_multiclass = base_dataset.output_shape  > 1

    # -- Rebuild reg cocktail from paper
    # we use no embedding - https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L179
    # we use not network init - https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L180

    backbone = build_backbone(input_shape, config)
    head = build_head(input_shape=get_output_shape(backbone, input_shape=input_shape), output_shape=output_shape, )
    network = torch.nn.Sequential(backbone, head).to(device)

    final_activation = build_final_activation(is_multiclass)
    optimizer = build_optimizer(network, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)
    trainer = build_trainer(seed, config)

    return _fit(data_loader, y, network, final_activation, trainer, optimizer, lr_scheduler, config, is_multiclass,
                device=device)  # -> must create network snapshots if needed (config["use_snapshot_ensemble"])


def _fit(data_loader, y, network, final_activation, trainer, optimizer, lr_scheduler, config, is_multiclass, device):
    from my_own_auto_pytorch_pipeline_code.trainer_stuff.base_trainer import BudgetTracker
    from my_own_auto_pytorch_pipeline_code.trainer_stuff.utils import update_model_state_dict_from_swa
    from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss
    budget_tracker = BudgetTracker(
        budget_type="epochs",
        max_runtime=None,
        max_epochs=config["epochs"],
    )
    trainer.prepare(
        model=network,
        model_final_activation=final_activation,
        device=device,
        criterion=CrossEntropyLoss if is_multiclass else BCEWithLogitsLoss,
        budget_tracker=budget_tracker,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        labels=y,
        step_interval='epoch', # epoch due to CosineAnnealingWarmRestarts
        output_type = "BINARY" if len(np.unique(y)) == 2 else "MULTICLASS",
    )

    epoch = 1
    while True:
        # No early stopping - https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/main/utilities.py#L406
        # No tracking as not needed
        train_loss = trainer.train_epoch(train_loader=data_loader)
        print(train_loss)

        if trainer.on_epoch_end(is_cyclic_scheduler=True, epoch=epoch):
            break

        if budget_tracker.is_max_epoch_reached(epoch + 1):
            break

        epoch += 1

    if trainer.use_stochastic_weight_averaging and trainer.swa_updated:
        # update batch norm statistics
        swa_model = trainer.swa_model.double()
        swa_utils.update_bn(loader=data_loader, model=swa_model.cpu())
        # change model
        update_model_state_dict_from_swa(network, trainer.swa_model.state_dict())
        if trainer.use_snapshot_ensemble:
            # we update only the last network which pertains to the stochastic weight averaging model
            snapshot_model = trainer.model_snapshots[-1].double()
            swa_utils.update_bn(data_loader, snapshot_model.cpu())

    network_snapshots = trainer.model_snapshots

    return network, network_snapshots, final_activation


def predict_with_network(X, config, network, network_snapshots, final_activation, device="cpu"):
    loader = build_data_loader(X, None, config)
    if len(network_snapshots) == 0:
        assert network is not None
        return _predict(network, loader, device, final_activation).numpy()
    else:
        # if there are network snapshots,
        # take average of predictions of all snapshots
        Y_snapshot_preds: List[torch.Tensor] = list()

        for snap_network in network_snapshots:
            Y_snapshot_preds.append(_predict(snap_network, loader, device, final_activation))
        Y_snapshot_preds_tensor = torch.stack(Y_snapshot_preds)
        return Y_snapshot_preds_tensor.mean(dim=0).numpy()


def _predict(network, loader, device, final_activation) -> torch.Tensor:
    network.to(device)
    network.float()
    network.eval()
    # Batch prediction
    Y_batch_preds = list()

    # `torch.no_grad` reduces memory usage even after `model.eval()`
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(loader):
            # Predict on batch
            X_batch = X_batch.float().to(device)
            Y_batch_pred = network(X_batch)
            Y_batch_pred = final_activation(Y_batch_pred)
            Y_batch_preds.append(Y_batch_pred.detach().cpu())

    return torch.cat(Y_batch_preds, 0)


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_magic():
    # TODO: open questions
    #   Epochs? why so low? or is this a lot??
    # TODO: what happens in case of no init and how can I seed what happens?? where bias init?


    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4202021)
    seed = 42

    config_space = _get_config()  # will crash as above is a config space not a config
    default_config = {k: v[0] for k,v in config_space.items()}

    network, network_snapshots, final_activation = _build_and_train_network(X_train, y_train, default_config, seed)
    y_pred = predict_with_network(X_test, default_config, network, network_snapshots, final_activation)

    print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    run_magic()