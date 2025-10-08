import torch
import math
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
from collections import OrderedDict
import typing

from .wenzel_sgld import dot, WenzelSGLD


class MongeSGLD(WenzelSGLD):
    """SGLD with preconditioning based on Hartmann et al. 2022.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        num_data (int): the number of data points in this learning task
        temperature (float): Temperature for tempering the posterior.
        raise_on_no_grad (bool): whether to complain if a parameter does not
                                 have a gradient
        raise_on_nan: whether to complain if a gradient is not all finite.
        monge_lambd: decay for the moving average of the gradients
        monge_alpha_2: alpha^{2} value for Monge
    """

    def __init__(
        self,
        params: Sequence[Union[torch.nn.Parameter, Dict]],
        lr: float,
        num_data: int,
        temperature: float = 1.0,
        raise_on_no_grad: bool = True,
        raise_on_nan: bool = False,
        monge_lambd: float = 0.9,
        monge_alpha_2: float = 1.0,
    ):
        assert lr >= 0 and num_data >= 0 and temperature >= 0
        defaults = dict(
            lr=lr,
            num_data=num_data,
            monge_lambd=monge_lambd,
            monge_alpha_2=monge_alpha_2,
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

        g["exp_avg_exp_avg"] = 0.0
        g["exp_avg_grad"] = 0.0
        g["exp_avg_noise"] = 0.0

        lambd = g["monge_lambd"]
        alpha_2 = g["monge_alpha_2"]

        for p in g["params"]:
            state = self.state[p]

            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg"].mul_(lambd).add_(p.grad, alpha=1 - lambd)

            g["exp_avg_exp_avg"] += dot(state["exp_avg"], state["exp_avg"])
            g["exp_avg_grad"] += dot(state["exp_avg"], p.grad)

            state["noise"] = torch.randn_like(p.grad)
            g["exp_avg_noise"] += dot(state["exp_avg"], state["noise"])

        g["r_factors"] = -alpha_2 / (1 + alpha_2 * g["exp_avg_exp_avg"])
        g["rsqrt_factors"] = (
            1 / math.sqrt(1 + alpha_2 * g["exp_avg_exp_avg"]) - 1
        ) / g["exp_avg_exp_avg"]

    def _step_fn(self, group, p, state, calc_metrics=True, is_final=False):
        if not is_final:
            p.add_(
                p.grad + group["r_factors"] * state["exp_avg"] * group["exp_avg_grad"],
                alpha=-group["hn"],
            )
            p.add_(
                state["noise"]
                + group["rsqrt_factors"] * state["exp_avg"] * group["exp_avg_noise"],
                alpha=group["noise_std"],
            )
