import warnings
from typing import Any, List, Tuple, Callable, Optional

import torch
from torch.autograd import Function
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch: List, x_collector: Callable = default_collate) -> List[Optional[torch.Tensor]]:
    """
    In the case of not providing a y tensor, in a
    dataset of form {X, y}, y would be None.

    This custom collate function allows to yield
    None data for functions that require only features,
    like predict.

    Args:
        batch (List): a batch from a dataset
        x_collector (callable): how the data is collected, e.g., when one want to pad sequences with different lengths.
            collate is only applied to X, for y, the normal default_collate is applied.

    Returns:
        List[Optional[torch.Tensor]]
    """

    items = list(zip(*batch))

    # The feature will always be available
    items[0] = x_collector(items[0])
    if None in items[1]:
        items[1] = list(items[1])
    else:
        items[1] = default_collate(items[1])
    return items


def get_output_shape(network: torch.nn.Module, input_shape: Tuple[int, ...], has_hidden_states: bool = False
                     ) -> Tuple[int, ...]:
    """
    Run a dummy forward pass to get the output shape of the backbone.
    Can and should be overridden by subclasses that know the output shape
    without running a dummy forward pass.
    :param input_shape: shape of the input
    :param has_hidden_states: bool, if the network backbone contains a hidden_states. if yes,
        the network will return a Tuple, we will then only consider the first item
    :return: output_shape
    """
    # as we are using nn embedding, 2 is a safe upper limit as 3
    # is the lowest `min_values_for_embedding` can be
    placeholder = torch.randint(high=2, size=(2, *input_shape), dtype=torch.float)
    with torch.no_grad():
        if has_hidden_states:
            output = network(placeholder)[0]
        else:
            output = network(placeholder)
    return tuple(output.shape[1:])


def get_shaped_neuron_counts(
        shape: str,
        in_feat: int,
        out_feat: int,
        max_neurons: int,
        layer_count: int
) -> List[int]:
    counts: List[int] = []

    if (layer_count <= 0):
        return counts

    if (layer_count == 1):
        counts.append(out_feat)
        return counts

    max_neurons = max(in_feat, max_neurons)
    # https://mikkokotila.github.io/slate/#shapes

    if shape == 'brick':
        #
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |___  ___|
        #
        for _ in range(layer_count - 1):
            counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'triangle':
        #
        #        /  \
        #       /    \
        #      /      \
        #     /        \
        #    /          \
        #   /_____  _____\
        #
        previous = in_feat
        step_size = int((max_neurons - previous) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous + step_size
            counts.append(previous)
        counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'funnel':
        #
        #   \            /
        #    \          /
        #     \        /
        #      \      /
        #       \    /
        #        \  /
        #
        previous = max_neurons
        counts.append(previous)

        step_size = int((previous - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous - step_size
            counts.append(previous)

        counts.append(out_feat)

    if shape == 'long_funnel':
        #
        #   |        |
        #   |        |
        #   |        |
        #    \      /
        #     \    /
        #      \  /
        #
        brick_layer = int(layer_count / 2)
        funnel_layer = layer_count - brick_layer
        counts.extend(get_shaped_neuron_counts(
            'brick', in_feat, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts(
            'funnel', in_feat, out_feat, max_neurons, funnel_layer))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: long funnel layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'diamond':
        #
        #     /  \
        #    /    \
        #   /      \
        #   \      /
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 2) + 1
        funnel_layer = layer_count - triangle_layer
        counts.extend(get_shaped_neuron_counts(
            'triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        remove_triangle_layer = len(counts) > 1
        if (remove_triangle_layer):
            # remove the last two layers since max_neurons == out_features
            # (-> two layers with the same size)
            counts = counts[0:-2]
        counts.extend(get_shaped_neuron_counts(
            'funnel',
            max_neurons,
            out_feat,
            max_neurons,
            funnel_layer + (2 if remove_triangle_layer else 0)))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: diamond layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'hexagon':
        #
        #     /  \
        #    /    \
        #   |      |
        #   |      |
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 3) + 1
        funnel_layer = triangle_layer
        brick_layer = layer_count - triangle_layer - funnel_layer
        counts.extend(get_shaped_neuron_counts(
            'triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        counts.extend(get_shaped_neuron_counts(
            'brick', max_neurons, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts(
            'funnel', max_neurons, out_feat, max_neurons, funnel_layer))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: hexagon layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'stairs':
        #
        #   |          |
        #   |_        _|
        #     |      |
        #     |_    _|
        #       |  |
        #       |  |
        #
        previous = max_neurons
        counts.append(previous)

        if layer_count % 2 == 1:
            counts.append(previous)

        step_size = 2 * int((max_neurons - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(int(layer_count / 2 - 1)):
            previous = previous - step_size
            counts.append(previous)
            counts.append(previous)

        counts.append(out_feat)

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: stairs layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    return counts


class ShakeShakeFunction(Function):
    """
    References:
        Title: Shake-Shake regularization
        Authors: Xavier Gastaldi
        URL: https://arxiv.org/pdf/1705.07485.pdf
        Github URL: https://github.com/hysts/pytorch_shake_shake/blob/master/functions/shake_shake_function.py
    """

    @staticmethod
    def forward(
            ctx: Any,  # No typing for AutogradContext
            x1: torch.Tensor,
            x2: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx: Any,
                 grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, alpha, beta = ctx.saved_tensors
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_shake = ShakeShakeFunction.apply


class ShakeDropFunction(Function):
    """
    References:
        Title: ShakeDrop Regularization for Deep Residual Learning
        Authors: Yoshihiro Yamada et. al.
        URL: https://arxiv.org/pdf/1802.02375.pdf
        Title: ShakeDrop Regularization
        Authors: Yoshihiro Yamada et. al.
        URL: https://openreview.net/pdf?id=S1NHaMW0b
        Github URL: https://github.com/owruby/shake-drop_pytorch/blob/master/models/shakedrop.py
    """

    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                bl: torch.Tensor,
                ) -> torch.Tensor:
        ctx.save_for_backward(x, alpha, beta, bl)

        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx: Any,
                 grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, alpha, beta, bl = ctx.saved_tensors
        grad_x = grad_alpha = grad_beta = grad_bl = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)

        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_get_alpha_beta(
        is_training: bool,
        device: torch.device,
        method: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The methods used in this function have been introduced in 'ShakeShake Regularisation'
    Each method name is available in the referred paper.
    Currently, this function supports `even-even`, `shake-even`, `shake-shake` and `M3`.

    Args:
        is_training (bool): Whether the computation for the training
        is_cuda (bool): Whether the tensor is on CUDA
        method (str): The shake method either `even-even`, `shake-even`, `shake-shake` or `M3`

    Returns:
        alpha, beta (Tuple[float, float]):
            alpha (in [0, 1]) is the weight coefficient  for the forward pass
            beta (in [0, 1]) is the weight coefficient for the backward pass

    Reference:
        Title: Shake-shake regularization
        Author: Xavier Gastaldi
        URL: https://arxiv.org/abs/1705.07485

    The names have been taken from the paper as well.
    Currently, this function supports `even-even`, `shake-even`, `shake-shake` and `M3`.
    """
    if not is_training:
        result = (torch.FloatTensor([0.5]).to(device), torch.FloatTensor([0.5]).to(device))
        return result

    # TODO implement other update methods
    # alpha is the weight ratio for the forward pass and beta is that for the backward pass
    alpha = torch.FloatTensor([0.5]) if method.startswith('even') else torch.rand(1)
    if method.endswith('even'):
        beta = torch.FloatTensor([0.5])
    elif method.endswith('shake'):
        beta = torch.rand(1)
    elif method == 'M3':
        # Table 4 in the paper `Shake-Shake regularization`
        rnd = torch.rand(1)
        beta = torch.FloatTensor(
            [rnd * (0.5 - alpha) + alpha if alpha < 0.5 else rnd * (alpha - 0.5) + 0.5]
        )
    else:
        raise ValueError(f"Unknown method `{method}` for ShakeShakeRegularisation in NetworkBackbone")

    alpha = alpha.to(device)
    beta = beta.to(device)

    return alpha, beta


def shake_drop_get_bl(
        block_index: int,
        min_prob_no_shake: float,
        num_blocks: int,
        is_training: bool,
        device: torch.device
) -> torch.Tensor:
    """
    The sampling of Bernoulli random variable
    based on Eq. (4) in the paper
    Args:
        block_index (int): The index of the block from the input layer
        min_prob_no_shake (float): The initial shake probability
        num_blocks (int): The total number of building blocks
        is_training (bool): Whether it is training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        bl (torch.Tensor): a Bernoulli random variable in {0, 1}
    Reference:
        ShakeDrop Regularization for Deep Residual Learning
        Yoshihiro Yamada et. al. (2020)
        paper: https://arxiv.org/pdf/1802.02375.pdf
        implementation: https://github.com/imenurok/ShakeDrop
    """
    pl = 1 - ((block_index + 1) / num_blocks) * (1 - min_prob_no_shake)

    if is_training:
        # Move to torch.randn(1) for reproducibility
        bl = torch.as_tensor(1.0) if torch.rand(1) <= pl else torch.as_tensor(0.0)
    else:
        bl = torch.as_tensor(pl)

    bl = bl.to(device)

    return bl


from torch import nn
from typing import Dict, Optional, Tuple, Union, List, Callable


class ResBlock(nn.Module):
    """
    __author__ = "Max Dippel, Michael Burkart and Matthias Urban"
    """

    def __init__(
            self,
            config: Dict[str, Any],
            in_features: int,
            out_features: int,
            blocks_per_group: int,
            block_index: int,
            dropout: Optional[float],
            activation: nn.Module
    ):
        super(ResBlock, self).__init__()
        self.config = config
        self.dropout = dropout
        self.activation = activation

        self.shortcut = None
        self.start_norm: Optional[Callable] = None

        # if in != out the shortcut needs a linear layer to match the result dimensions
        # if the shortcut needs a layer we apply batchnorm and activation to the shortcut
        # as well (start_norm)
        if in_features != out_features and self.config["use_skip_connection"]:
            self.shortcut = nn.Linear(in_features, out_features)
            initial_normalization = list()
            if self.config['use_batch_norm']:
                initial_normalization.append(
                    nn.BatchNorm1d(in_features)
                )
            initial_normalization.append(
                self.activation()
            )
            self.start_norm = nn.Sequential(
                *initial_normalization
            )

        self.block_index = block_index
        self.num_blocks = blocks_per_group * self.config["num_groups"]
        self.layers = self._build_block(in_features, out_features)

        if self.config["use_skip_connection"]:
            if config["multi_branch_choice"] == 'shake-shake':
                self.shake_shake_layers = self._build_block(in_features, out_features)

    # each block consists of two linear layers with batch norm and activation
    def _build_block(self, in_features: int, out_features: int) -> nn.Module:
        layers = list()

        if self.start_norm is None:
            if self.config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(in_features))
            layers.append(self.activation())

        layers.append(nn.Linear(in_features, out_features))

        if self.config['use_batch_norm']:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.activation())

        if self.dropout is not None:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(out_features, out_features))

        return nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        # if shortcut is not none we need a layer such that x matches the output dimension
        if self.shortcut is not None and self.start_norm is not None:
            # in this case self.start_norm is also != none
            # apply start_norm to x in order to have batchnorm+activation
            # in front of shortcut and layers. Note that in this case layers
            # does not start with batchnorm+activation but with the first linear layer
            # (see _build_block). As a result if in_features == out_features
            # -> result = x + W_2(~D(A(BN(W_1(A(BN(x))))))
            # if in_features != out_features
            # -> result = W_shortcut(A(BN(x))) + W_2(~D(A(BN(W_1(A(BN(x))))))
            x = self.start_norm(x)
            residual = self.shortcut(x)
        elif self.config["use_skip_connection"]:
            # We use a skip connection but we do not need to match dimensions
            residual = x
        else:  # Early-return because no need of skip connection
            return self.layers(x)

        if self.config["multi_branch_choice"] == 'shake-shake':
            x1 = self.layers(x)
            x2 = self.shake_shake_layers(x)
            alpha, beta = shake_get_alpha_beta(
                is_training=self.training,
                device=x.device,
                method=self.config['shake_shake_update_func'],
            )
            x = shake_shake(x1, x2, alpha, beta)
        elif self.config["multi_branch_choice"] == 'shake-drop':
            x = self.layers(x)
            alpha, beta = shake_get_alpha_beta(
                is_training=self.training,
                device=x.device,
                method=self.config['shake_shake_update_func'],
            )
            bl = shake_drop_get_bl(
                self.block_index,
                1 - self.config["max_shake_drop_probability"],
                self.num_blocks,
                self.training,
                x.device,
            )
            x = shake_drop(x, alpha, beta, bl)
        else:
            x = self.layers(x)

        return x + residual
