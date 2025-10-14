import torch
import math
from torch.optim.optimizer import Optimizer

def psi_fn(z, m, M, r):
    zr = z ** r
    return m * (zr + M) / (zr + m)

class SGLD(Optimizer):
    def __init__(self, params, lr, num_data, temperature=1.0):
        defaults = dict(lr=lr, num_data=num_data, temperature=temperature)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            dt = group["lr"] / group["num_data"]
            noise_std = math.sqrt(max(0.0, 2.0 * dt * group["temperature"]))
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-dt)
                p.add_(torch.randn_like(p), alpha=noise_std)
        return loss

class SASGLD(Optimizer):
    """
    SASGLD: per-parameter z, psi and dt. Single Z update per step (practical).
    Hyperparams in group defaults: alpha, m, M, r, s, init_z
    """
    def __init__(self, params, lr, num_data, temperature=1.0,
                 alpha=1.0, m=1e-6, M=1.0, r=0.25, s=2., Omega=50000, init_z=1.0):
        defaults = dict(
            lr=lr, num_data=num_data, temperature=temperature,
            alpha=alpha, m=m, M=M, r=r, s=s, Omega=Omega, init_z=init_z
        )
        super().__init__(params, defaults)

        # initialize per-parameter state entries
        for g in self.param_groups:
            for p in g["params"]:
                state = self.state[p]
                if "z" not in state:
                    state["z"] = float(0.)
                if "psi" not in state:
                    state["psi"] = psi_fn(state["z"], m, M, r)
                if "dt" not in state:
                    state["dt"] = state["psi"] * lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            dtau = lr / group["num_data"]
            alpha = group["alpha"]
            m = group["m"]
            M = group["M"]
            r = group["r"]
            s = group["s"]
            T = group["temperature"]
            Omega = group["Omega"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g_sq = float((p.grad.view(-1) ** 2).sum().item())
                g_norm = math.sqrt(g_sq)
                rho = math.exp(-alpha * dtau)
                g_val = g_norm ** s / Omega
                state["z"] = rho * state["z"] + (1.0 - rho) * (g_val / alpha)
                state["psi"] = float(psi_fn(state["z"], m, M, r))
                state["dt"] = float(state["psi"] * dtau)

                if state["dt"] <= 0:
                    raise ValueError("Non-positive dt in SASGLD")

                noise_std = math.sqrt(max(0.0, 2.0 * state["dt"] * T))
                p.add_(p.grad, alpha=-state["dt"])
                p.add_(torch.randn_like(p), alpha=noise_std)
        return loss

    def get_state_samples_info(self, model):
        """
        Return dicts mapping parameter names -> (z, psi, dt) for the given model.
        Useful for logging sample-level info. model should be the same instance used with this optimizer.
        """
        info_z = {}
        info_psi = {}
        info_dt = {}
        # iterate named_parameters to get matching param objects
        for name, p in model.named_parameters():
            st = self.state.get(p, {})
            info_z[name] = st.get("z", None)
            info_psi[name] = st.get("psi", None)
            info_dt[name] = st.get("dt", None)
        return {"z": info_z, "psi": info_psi, "dt": info_dt}
