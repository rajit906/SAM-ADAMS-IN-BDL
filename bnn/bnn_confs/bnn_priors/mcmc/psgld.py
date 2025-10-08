import torch
import math
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
from collections import OrderedDict
import typing

from .wenzel_sgld import dot, WenzelSGLD


class pSGLD(WenzelSGLD):
    """SGLD with preconditioning from Li et al. 2015.

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
        rmsprop_alpha: float = 0.99,
        rmsprop_eps: float = 1e-8,
        raise_on_no_grad: bool = True,
        raise_on_nan: bool = False,
    ):
        assert lr >= 0 and num_data >= 0 and temperature >= 0
        defaults = dict(
            lr=lr,
            num_data=num_data,
            rmsprop_alpha=rmsprop_alpha,
            rmsprop_eps=rmsprop_eps,
            temperature=temperature,
        )
        super(WenzelSGLD, self).__init__(params, defaults)
        self.raise_on_no_grad = raise_on_no_grad
        self.raise_on_nan = raise_on_nan
        self._step_count = 0  # keep the `torch.optim.scheduler` happy

    def _update_group_fn(self, g):
        g["hn"] = g["lr"]
        g["h"] = g["lr"] / g["num_data"]
        g["noise_std"] = math.sqrt(2 * g["h"] * g["temperature"])

        eps = g["rmsprop_eps"]
        for p in g["params"]:
            state = self.state[p]
            alpha = g["rmsprop_alpha"]

            if "square_avg" not in state:
                state["square_avg"] = torch.zeros_like(p)
            state["square_avg"].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)

            self.state[p]["preconditioner"] = 1 / (
                torch.sqrt(state["square_avg"]) + eps
            )

    def _step_fn(self, group, p, state, calc_metrics=True, is_final=False):
        M_r = state["preconditioner"]
        M_rsqrt = torch.sqrt(M_r)
        if not is_final:
            p.add_(p.grad * M_r, alpha=-group["hn"])
            p.add_(torch.randn_like(p) * M_rsqrt, alpha=group["noise_std"])
