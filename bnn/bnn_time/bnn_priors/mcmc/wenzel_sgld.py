import torch
import math
import numpy as np
from collections import OrderedDict
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing


def dot(a, b):
    "return (a*b).sum().item()"
    return (a.view(-1) @ b.view(-1)).item()


class WenzelSGLD(torch.optim.Optimizer):
    """SGLD with preconditioning from Wenzel et al. 2020.

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
        rmsprop_eps: float = 1e-8,  # Wenzel et al. use 1e-7
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
        self.update_preconditioner()
        self._step_count = 0  # keep the `torch.optim.scheduler` happy
        for group in self.param_groups:
            if "hn" not in group:
                group["hn"] = group["lr"]
                group["h"] = group["lr"] / group["num_data"]
                group["noise_std"] = math.sqrt(2 * group["h"] * group["temperature"])

    def _preconditioner_default(self, state, p) -> float:
        try:
            return state["preconditioner"], math.sqrt(state["preconditioner"])
        except KeyError:
            v = state["preconditioner"] = 1.0
            return v, v

    def delta_energy(self, a, b) -> float:
        return math.inf

    @torch.no_grad()
    def step(self):
        return self._step_internal(self._update_group_fn, self._step_fn)

    initial_step = step

    @torch.no_grad()
    def final_step(
        self,
        save_state=False,
    ):
        assert save_state is False
        return self._step_internal(
            self._update_group_fn,
            self._step_fn,
            is_final=True,
        )

    def _step_internal(self, update_group_fn, step_fn, **step_fn_kwargs):
        for group in self.param_groups:
            update_group_fn(group)
            for p in group["params"]:
                if p.grad is None:
                    if self.raise_on_no_grad:
                        raise RuntimeError(
                            f"No gradient for parameter with shape {p.shape}"
                        )
                    continue
                if self.raise_on_nan and not torch.isfinite(p.grad).all():
                    raise ValueError(
                        f"Gradient of shape {p.shape} is not finite: {p.grad}"
                    )
                step_fn(group, p, self.state[p], **step_fn_kwargs)

    def _update_group_fn(self, g):
        pass

    def _step_fn(self, group, p, state):
        """if is_final, do not change parameters"""
        M_r, M_rsqrt = self._preconditioner_default(state, p)
        # Take the gradient step
        p.add_(p.grad, alpha=-group["hn"] * M_r)
        p.add_(torch.randn_like(p), alpha=group["noise_std"] * M_rsqrt)

        # RMSProp moving average
        alpha = group["rmsprop_alpha"]
        state["square_avg"].mul_(alpha).addcmul_(p.grad, p.grad, value=1 - alpha)

    @torch.no_grad()
    def update_preconditioner(self):
        """Updates the preconditioner for each parameter `state['preconditioner']` using
        the estimated `state['square_avg']`.
        """
        precond = OrderedDict()
        min_s = math.inf

        for group in self.param_groups:
            eps = group["rmsprop_eps"]
            for p in group["params"]:
                state = self.state[p]
                try:
                    square_avg = state["square_avg"]
                except KeyError:
                    square_avg = state["square_avg"] = torch.ones_like(p)

                precond[p] = square_avg.mean().item() + eps
                min_s = min(min_s, precond[p])

        for p, new_M in precond.items():
            # ^(1/2) to form the preconditioner,
            # ^(-1) because we want the preconditioner's inverse.
            self.state[p]["preconditioner"] = (new_M / min_s) ** (-1 / 2)
