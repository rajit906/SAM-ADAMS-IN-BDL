# partly based on https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch

import torch
import math
import numpy as np
from collections import OrderedDict
from typing import Sequence, Optional, Callable, Tuple, Dict, Union
import typing
from .wenzel_sgld import WenzelSGLD, dot

import enum
import itertools

from dataclasses import dataclass
from .matrix_functions import ComputePower, MatPower
import numpy as np
import torch


@dataclass
class ShampooHyperParams:
    """Shampoo hyper parameters."""

    inverse_exponent_override: int = 0  # fixed exponent for preconditioner, if >0
    start_preconditioning_step: int = 1
    # Performance tuning params for controlling memory and compute requirements.
    # How often to compute preconditioner.
    preconditioning_compute_steps: int = 100
    # How often to compute statistics.
    statistics_compute_steps: int = 1
    # Block size for large layers (if > 0).
    # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
    # Block size should be as large as feasible under memory/time constraints.
    block_size: int = 128
    # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
    # 12 x [1024, 512] L and R statistics. Disabled by default which results in
    # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
    best_effort_shape_interpretation: bool = True


class BlockPartitioner:
    """Partitions a tensor into smaller tensors for preconditioning.
    For example, if a variable has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 variables of size
    (1024, 512) each.
    """

    def __init__(self, var, hps):
        self._shape = var.shape
        self._splits = []
        self._split_sizes = []
        split_sizes = []
        # We split var into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(var.shape):
            if hps.block_size > 0 and d > hps.block_size:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // hps.block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * hps.block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * hps.block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                self._split_sizes.append((i, sizes))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._num_splits = len(split_sizes)
        self._preconditioner_shapes = []
        for t in itertools.product(*split_sizes):
            self._preconditioner_shapes.extend([[d, d] for d in t])

    def shapes_for_preconditioners(self):
        return self._preconditioner_shapes

    def num_splits(self):
        return self._num_splits

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for (i, sizes) in self._split_sizes:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(torch.split(t, tuple(sizes), dim=i))
            tensors = tensors_local
        return tensors

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for (i, indices) in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(
                    torch.cat(partitions[ind : ind + n], axis=i)
                )
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


def _merge_small_dims(shape_to_merge, max_dim):
    """Merge small dimensions.
    If there are some small dimensions, we collapse them:
    e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
         [1, 2, 768, 1, 2048] --> [2, 768, 2048]
    Args:
      shape_to_merge: Shape to merge small dimensions.
      max_dim: Maximal dimension of output shape used in merging.
    Returns:
      Merged shape.
    """
    resulting_shape = []
    product = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d
    if product > 1:
        resulting_shape.append(product)
    return resulting_shape


class Preconditioner:
    """Compute statistics/shape from gradients for preconditioning."""

    def __init__(self, var, hps, rmsprop_alpha, rmsprop_eps):
        self._hps = hps
        self.rmsprop_alpha = rmsprop_alpha
        self.rmsprop_eps = rmsprop_eps
        self._original_shape = var.shape
        self._transformed_shape = var.shape
        if hps.best_effort_shape_interpretation:
            self._transformed_shape = _merge_small_dims(
                self._original_shape, hps.block_size
            )

        reshaped_var = torch.reshape(var, self._transformed_shape)
        self._partitioner = BlockPartitioner(reshaped_var, hps)
        shapes = self._partitioner.shapes_for_preconditioners()
        rank = len(self._transformed_shape)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if rank <= 1:
            self.statistics = []
            self.grad_preconditioners = []
            self.noise_preconditioners = []
        else:
            self.statistics = [
                self.rmsprop_eps * torch.eye(s[0], device=device) for s in shapes
            ]
            self.grad_preconditioners = [torch.eye(s[0], device=device) for s in shapes]
            self.noise_preconditioners = [
                torch.eye(s[0], device=device) for s in shapes
            ]

    def add_statistics(self, grad):
        """Compute statistics from gradients and add to the correct state entries.
        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics:
            return
        reshaped_grad = torch.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        w1 = self.rmsprop_alpha
        w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = torch.tensordot(grad, grad, [axes, axes])
                self.statistics[j * rank + i].mul_(w1).add_(stat, alpha=w2)

    def exponent_for_preconditioner(self):
        """Returns exponent to use for inverse-pth root M^{-1/p}.
        The exponent is 2 times that in Shampoo for optimization,
        because we solve for the inverse square root of the metric here.
        """
        if self._hps.inverse_exponent_override > 0:
            return self._hps.inverse_exponent_override
        return 4 * len(self._transformed_shape)

    def compute_preconditioners(self):
        """Compute L^{-1/exp} for each stats matrix L."""
        exp = self.exponent_for_preconditioner()
        eps = self.rmsprop_eps
        for i, stat in enumerate(self.statistics):
            self.noise_preconditioners[i] = ComputePower(stat, exp, ridge_epsilon=eps)
            self.grad_preconditioners[i] = torch.matmul(
                self.noise_preconditioners[i], self.noise_preconditioners[i]
            )

    def preconditioned_grad(self, grad):
        """Precondition the gradient.
        Args:
          grad: A gradient tensor to precondition.
        Returns:
          A preconditioned gradient.
        """
        if not self.grad_preconditioners:
            return grad
        reshaped_grad = torch.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        num_splits = self._partitioner.num_splits()
        for i, grad in enumerate(partitioned_grads):
            preconditioners_for_grad = self.grad_preconditioners[
                i * num_splits : (i + 1) * num_splits
            ]
            rank = len(grad.shape)
            precond_grad = grad
            for j in range(rank):
                preconditioner = preconditioners_for_grad[j]
                precond_grad = torch.tensordot(precond_grad, preconditioner, [[0], [0]])
            preconditioned_partitioned_grads.append(precond_grad)
        merged_grad = self._partitioner.merge_partitions(
            preconditioned_partitioned_grads
        )
        return torch.reshape(merged_grad, self._original_shape)

    def preconditioned_noise(self, noise):
        """Precondition the noise.
        Args:
          noise: A noise tensor to precondition.
        Returns:
          A preconditioned noise.
        """
        if not self.noise_preconditioners:
            return noise
        reshaped_noise = torch.reshape(noise, self._transformed_shape)
        partitioned_noises = self._partitioner.partition(reshaped_noise)
        preconditioned_partitioned_noises = []
        num_splits = self._partitioner.num_splits()
        for i, noise in enumerate(partitioned_noises):
            preconditioners_for_noise = self.noise_preconditioners[
                i * num_splits : (i + 1) * num_splits
            ]
            rank = len(noise.shape)
            precond_noise = noise
            for j in range(rank):
                preconditioner = preconditioners_for_noise[j]
                precond_noise = torch.tensordot(
                    precond_noise, preconditioner, [[0], [0]]
                )
            preconditioned_partitioned_noises.append(precond_noise)
        merged_noise = self._partitioner.merge_partitions(
            preconditioned_partitioned_noises
        )
        return torch.reshape(merged_noise, self._original_shape)


class ShampooSGLD(WenzelSGLD):
    """SGLD with preconditioning based on Anil et al. 2021.

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
        hps: additional hyper parameters for Shampoo
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
        hps: dataclass = ShampooHyperParams,
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
        self.hps = hps

        # initialize
        for group in self.param_groups:
            if "hn" not in group:
                group["hn"] = group["lr"]
                group["h"] = group["lr"] / group["num_data"]
                group["noise_std"] = math.sqrt(2 * group["h"] * group["temperature"])
            for p in group["params"]:
                state = self.state[p]
                state["preconditioner"] = Preconditioner(
                    p, self.hps, rmsprop_alpha, rmsprop_eps
                )

    def _step_fn(self, group, p, state):
        """if is_final, do not change parameters"""
        grad = p.grad.data

        preconditioner = state["preconditioner"]

        if self._step_count % self.hps.statistics_compute_steps == 0:
            preconditioner.add_statistics(grad)

        if self._step_count % self.hps.preconditioning_compute_steps == 0:
            preconditioner.compute_preconditioners()

        shampoo_grad = grad
        shampoo_noise = torch.randn_like(p)
        if self._step_count >= self.hps.start_preconditioning_step:
            shampoo_grad = preconditioner.preconditioned_grad(grad)
            shampoo_noise = preconditioner.preconditioned_noise(shampoo_noise)

        # Take the gradient step
        p.add_(shampoo_grad, alpha=-group["hn"])
        p.add_(shampoo_noise, alpha=group["noise_std"])
