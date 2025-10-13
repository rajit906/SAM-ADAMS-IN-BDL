# bnn_priors/mcmc/sasgld.py
import torch
import math
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
from collections import OrderedDict
import typing

from .wenzel_sgld import dot, WenzelSGLD


def psi_fn(z: float, m: float, M: float, r: float) -> float:
    """Bounded scaling function ψ(z) in [m, M]."""
    zr = z ** r
    # safe form; m should be > 0 in normal usage
    return m * (zr + M) / (zr + m)


class SASGLD(WenzelSGLD):
    """
    SASGLD: Adaptive-step-size overdamped SGLD with per-parameter Z variables.

    - Follows the same API as pSGLD / WenzelSGLD:
        * _update_group_fn(self, g)
        * _step_fn(self, group, p, state, calc_metrics=True, is_final=False)

    - Uses one Z update per optimizer step (no half-steps).
    - Stores per-parameter z in state[p]['z'].
    - Exposes state[p]['z'], state[p]['psi'], state[p]['dt'] for metrics.
    """

    def __init__(
        self,
        params: Sequence[Union[torch.nn.Parameter, Dict]],
        lr: float,
        num_data: int,
        temperature: float = 1.0,
        alpha: float = 1.0,
        m: float = 1e-6,
        M: float = 1.0,
        r: float = 1.0,
        s: float = 1.0,
        init_z: float = 1.0,
        raise_on_no_grad: bool = True,
        raise_on_nan: bool = False,
    ):
        assert lr >= 0 and num_data >= 0 and temperature >= 0
        defaults = dict(
            lr=lr,
            num_data=num_data,
            temperature=temperature,
            alpha=alpha,
            m=m,
            M=M,
            r=r,
            s=s,
            init_z=init_z,
        )
        # call parent with defaults (we use WenzelSGLD's mechanism)
        super(WenzelSGLD, self).__init__(params, defaults)

        # expose extras on the instance for convenience (not strictly required)
        self.raise_on_no_grad = raise_on_no_grad
        self.raise_on_nan = raise_on_nan
        self._step_count = 0  # keep schedulers happy

    def _update_group_fn(self, g):
        """
        Called before iterating parameters in this group.
        We'll set basic step sizes similar to other SGLD implementations.
        Leave per-parameter z initialization to state init below.
        """
        g["hn"] = g["lr"]
        g["h"] = g["lr"] / g["num_data"]
        # noise_std here is a base placeholder; actual noise depends on adaptive dt per-parameter
        g["noise_std"] = math.sqrt(2 * g["h"] * g["temperature"])

        # ensure per-parameter state entries exist (z, psi, dt)
        for p in g["params"]:
            state = self.state[p]
            # initialize z if missing
            if "z" not in state:
                state["z"] = float(g.get("init_z", 1.0))
            # ensure psi/dt are present to avoid KeyError in metrics
            if "psi" not in state:
                state["psi"] = float(1.0)
            if "dt" not in state:
                state["dt"] = state["psi"] * g["h"]

    def _step_fn(self, group, p, state, calc_metrics=True, is_final=False):
        """
        Performs parameter update for a single parameter tensor `p`.
        - Updates per-parameter z in state['z'] (single full-step per iteration).
        - Computes psi = psi_fn(z) and dt = psi * dtau (dtau = group['h']).
        - Applies overdamped Langevin update theta <- theta - dt * grad + sqrt(2 dt T) * N(0,1)
        If is_final, parameters are left unchanged (matching other optimizers).
        """
        # If there is no grad and raisers are set, error out as other optimizers do.
        if p.grad is None:
            if self.raise_on_no_grad:
                raise RuntimeError(f"No gradient for parameter with shape {p.shape}")
            return

        if self.raise_on_nan and not torch.isfinite(p.grad).all():
            raise ValueError(f"Gradient of shape {p.shape} is not finite: {p.grad}")

        # grad norm for this parameter tensor
        # convert to Python float for z updates
        grad_sq = float((p.grad.view(-1) ** 2).sum().item())
        grad_norm = math.sqrt(grad_sq)

        # dtau is the base timescale for the group
        dtau = group["h"]
        alpha = group.get("alpha", group.get("lr", 1.0))  # alpha is available in defaults but read from group for safety
        # But our defaults did not write alpha into group; better to read from optimizer defaults
        # The parent stored alpha in defaults; access via self.param_groups? Use group-level fallback
        alpha = group.get("alpha", getattr(self, "alpha", 1.0))

        # However in our __init__ we stored alpha etc in defaults; WenzelSGLD super call placed them in param_groups' defaults.
        # Safest: read from group if present, else from param_groups[0] if available.
        if "alpha" not in group:
            if len(self.param_groups) > 0:
                alpha = self.param_groups[0].get("alpha", alpha)

        # read SASGLD hyperparams (m, M, r, s, init_z)
        m = group.get("m", getattr(self.param_groups[0], "get", lambda k, d: d)("m", 1e-6))
        M = group.get("M", getattr(self.param_groups[0], "get", lambda k, d: d)("M", 1.0))
        r = group.get("r", getattr(self.param_groups[0], "get", lambda k, d: d)("r", 1.0))
        s = group.get("s", getattr(self.param_groups[0], "get", lambda k, d: d)("s", 1.0))

        # If alpha/m/M/r/s were present in group's dict (from defaults) this picks them up; otherwise fallback reasonable values.
        # Now compute z update (single full-step)
        rho = math.exp(-alpha * dtau)
        g_val = grad_norm ** s

        # init state['z'] if missing (should be done in _update_group_fn, but double-safe)
        if "z" not in state:
            state["z"] = float(group.get("init_z", 1.0))

        # update z (per-parameter)
        state["z"] = rho * state["z"] + (1.0 - rho) * (g_val / alpha)

        # compute psi and dt for this parameter
        psi = psi_fn(state["z"], m, M, r)
        dt = psi * dtau

        # stash for metrics
        state["psi"] = float(psi)
        state["dt"] = float(dt)

        # noise std for this parameter uses group temperature
        noise_std = math.sqrt(2.0 * dt * group["temperature"])

        if not is_final:
            # Langevin update: theta <- theta - dt * grad + sqrt(2 dt T) * N(0,1)
            p.add_(p.grad, alpha=-dt)
            p.add_(torch.randn_like(p), alpha=noise_std)

        # Optionally calculate metrics (no-op here, but signature kept)
        if calc_metrics:
            # keep any metric calculations minimal; runner will fetch state[p]['z']
            pass

