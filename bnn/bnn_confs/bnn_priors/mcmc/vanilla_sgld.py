import torch
import math
import numpy as np
from collections import OrderedDict
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing

from .wenzel_sgld import WenzelSGLD, dot


class VanillaSGLD(WenzelSGLD):
    """SGLD with identity preconditioning.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        num_data (int): the number of data points in this learning task
        temperature (float): Temperature for tempering the posterior.
        rmsprop_alpha: decay for the moving average of the squared gradients
        rmsprop_eps: the regularizer parameter for the RMSProp update
        raise_on_no_grad (bool): whether to complain if a parameter does not
                                 have a gradient
        raise_on_nan: whether to complain if a gradient is not all finite.
    """

    def __init__(
        self,
        params: Sequence[Union[torch.nn.Parameter, Dict]],
        lr: float,
        num_data: int,
        temperature: float = 1.0,
        raise_on_no_grad: bool = True,
        raise_on_nan: bool = False,
    ):
        assert lr >= 0 and num_data >= 0 and temperature >= 0
        defaults = dict(
            lr=lr,
            num_data=num_data,
            temperature=temperature,
        )
        super(WenzelSGLD, self).__init__(params, defaults)
        self.raise_on_no_grad = raise_on_no_grad
        self.raise_on_nan = raise_on_nan
        self._step_count = 0  # keep the `torch.optim.scheduler` happy

    def _step_fn(self, group, p, state, calc_metrics=True, is_final=False):
        """if is_final, do not change parameters"""
        if not is_final:
            # Take the gradient step
            p.add_(p.grad, alpha=-group["hn"])
            p.add_(torch.randn_like(p), alpha=group["noise_std"])
