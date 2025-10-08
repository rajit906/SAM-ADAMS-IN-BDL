"""
Training script for the BNN experiments with different data sets and priors.
"""

import os
import math
import uuid
import json
import contextlib
import pickle
import time

import numpy as np
import torch as t
from pathlib import Path
from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from bnn_priors.data import UCI, CIFAR10, Synthetic
from bnn_priors.models import RaoBDenseNet, DenseNet, PreActResNet18, PreActResNet34
from bnn_priors.prior import LogNormal
from bnn_priors import prior
import bnn_priors.inference
from bnn_priors import exp_utils
from bnn_priors.exp_utils import get_prior

import matplotlib.pyplot as plt

# Makes CUDA faster
if t.cuda.is_available():
    t.backends.cudnn.benchmark = True

TMPDIR = "/tmp"

ex = Experiment("bnn_training")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    # the dataset to be trained on, e.g., "mnist", "cifar10", "UCI_boston"
    data = "mnist"
    # the inference method to be used, defaults to GGMC from https://arxiv.org/abs/2102.01691
    inference = "VerletSGHMCReject"
    # model to be used, e.g., "classificationdensenet", "classificationconvnet", "googleresnet"
    model = "classificationconvnet"
    # width of the model (might not have an effect in some models)
    width = 50
    # depth of the model (might not have an effect in some models)
    depth = 3
    # number of learning rate cycles for the Markov chain (see https://arxiv.org/abs/1902.03932)
    cycles = 5
    # number of epochs in a cycle
    epochs_per_cycle = 20
    # weight prior, e.g., "gaussian", "laplace", "student-t"
    weight_prior = "gaussian"
    # bias prior, same as above
    bias_prior = "gaussian"
    # location parameter for the weight prior
    weight_loc = 0.0
    # scale parameter for the weight prior
    weight_scale = 2.0**0.5
    # location parameter for the bias prior
    bias_loc = 0.0
    # scale parameter for the bias prior
    bias_scale = 1.0
    # additional keyword arguments for the weight prior
    weight_prior_params = {}
    # additional keyword arguments for the bias prior
    bias_prior_params = {}
    if not isinstance(weight_prior_params, dict):
        weight_prior_params = json.loads(weight_prior_params)
    if not isinstance(bias_prior_params, dict):
        bias_prior_params = json.loads(bias_prior_params)
    # number of epochs to skip between computing metrics
    metrics_skip = 10
    # temperature of the sampler
    temperature = 1.0
    # learning rate schedule during sampling
    sampling_decay = "cosine"
    # update factor for the preconditioner, applicable only to SGLD
    precond_update = 1
    # learning rate
    lr = 5e-4
    # initialization method for the network weights
    init_method = "he"
    # previous samples to be loaded to initialize the chain
    load_samples = None
    # batch size for the training
    batch_size = 128
    # whether to use Metropolis-Hastings rejection steps (works only with some integrators)
    reject_samples = False
    # whether to use batch normalization
    batchnorm = True
    # device to use, "cpu", "cuda:0", "try_cuda"
    device = "try_cuda"
    # whether the samples should be saved
    save_samples = True
    # whether a progressbar should be plotted to stdout during the training
    progressbar = True
    # a random unique ID for the run
    run_id = uuid.uuid4().hex
    # directory where the results will be stored
    log_dir = str(Path(__file__).resolve().parent.parent / "logs")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        ex.observers.append(FileStorageObserver(log_dir))
    # whether to plot uncertainty plot, currently only works when save_samples == True
    uncertainty_plot = False
    # maximum absolute value of gradient
    grad_max = 1e6
    # EMA factor for Monge
    monge_lambd = 0.9
    # value of alpha^{2} for Monge
    monge_alpha_2 = 1.0
    # EMA factor for RMSprop-like preconditioners
    rmsprop_alpha = 0.99
    # epsilon for RMSprop-like preconditioners
    rmsprop_eps = 1e-8
    # number of batches to wait before taking a new sample
    thinning = 100
    # number of batches to use as burn-in
    burnin_batches = 300
    # whether to calculate curvature of samples during training
    calc_curvatures = True


device = ex.capture(exp_utils.device)
get_model = ex.capture(exp_utils.get_model)


@ex.capture
def get_data(data, batch_size, _run):
    if data == "empty":
        dataset = exp_utils.get_data("UCI_boston", device())
        dataset.norm.train = [(None, None)]
        dataset.norm.test = [(None, None)]
        dataset.unnorm.train = [(None, None)]
        dataset.unnorm.test = [(None, None)]
        return dataset

    if data[:9] == "synthetic":
        _, data, prior = data.split(".")
        dataset = get_data(data)
        x_train = dataset.norm.train_X
        y_train = dataset.norm.train_y
        model = get_model(
            x_train=x_train, y_train=y_train, weight_prior=prior, weight_prior_params={}
        )
        model.sample_all_priors()
        data = Synthetic(
            dataset=dataset, model=model, batch_size=batch_size, device=device()
        )
        t.save(data, exp_utils.sneaky_artifact(_run, "synthetic_data.pt"))
        t.save(model, exp_utils.sneaky_artifact(_run, "true_model.pt"))
        return data
    else:
        return exp_utils.get_data(data, device())


@ex.capture
def evaluate_model(model, dataloader_test, samples, uncertainty_plot, dataloader=None):
    return exp_utils.evaluate_model(
        model=model,
        dataloader_test=dataloader_test,
        samples=samples,
        likelihood_eval=True,
        accuracy_eval=True,
        calibration_eval=False,
        uncertainty_plot=uncertainty_plot,
        dataloader=dataloader,
    )


@ex.automain
def main(
    inference,
    model,
    width,
    init_method,
    metrics_skip,
    cycles,
    epochs_per_cycle,
    temperature,
    precond_update,
    lr,
    batch_size,
    load_samples,
    save_samples,
    reject_samples,
    run_id,
    log_dir,
    sampling_decay,
    progressbar,
    _run,
    _log,
    uncertainty_plot,
    grad_max,
    monge_lambd,
    monge_alpha_2,
    rmsprop_alpha,
    rmsprop_eps,
    thinning,
    burnin_batches,
    calc_curvatures,
):

    assert width > 0
    assert cycles > 0
    assert temperature >= 0

    data = get_data()

    x_train = data.norm.train_X
    y_train = data.norm.train_y

    total = x_train.shape[0]
    num_batches = math.ceil(total / batch_size)
    samples_per_epoch = math.ceil(num_batches / thinning)

    # number of first samples to skip
    skip_first = math.ceil(burnin_batches / thinning)

    # num_samples = cycles * epochs_per_cycle * samples_per_epoch

    model = get_model(x_train=x_train, y_train=y_train)

    if load_samples is None:
        if init_method == "he":
            exp_utils.he_initialize(model)
        elif init_method == "he_uniform":
            exp_utils.he_uniform_initialize(model)
        elif init_method == "he_zerobias":
            exp_utils.he_zerobias_initialize(model)
        elif init_method == "prior":
            pass
        else:
            raise ValueError(f"unknown init_method={init_method}")
    else:
        state_dict = exp_utils.load_samples(load_samples, idx=-1, keep_steps=False)
        model_sd = model.state_dict()
        for k in state_dict.keys():
            if k not in model_sd:
                _log.warning(f"key {k} not in model, ignoring")
                del state_dict[k]
            elif model_sd[k].size() != state_dict[k].size():
                _log.warning(
                    f"key {k} size mismatch, model={model_sd[k].size()}, loaded={state_dict[k].size()}"
                )
                state_dict[k] = model_sd[k]

        missing_keys = set(model_sd.keys()) - set(state_dict.keys())
        _log.warning(
            f"The following keys were not found in loaded state dict: {missing_keys}"
        )
        model_sd.update(state_dict)
        model.load_state_dict(model_sd)
        del state_dict
        del model_sd

    if save_samples:
        model_saver_fn = lambda: exp_utils.HDF5ModelSaver(
            exp_utils.sneaky_artifact(_run, "samples.pt"), "w"
        )
    else:

        @contextlib.contextmanager
        def model_saver_fn():
            yield None

    with exp_utils.HDF5Metrics(
        exp_utils.sneaky_artifact(_run, "metrics.h5"), "w"
    ) as metrics_saver, model_saver_fn() as model_saver:
        # if inference == "HMC":
        #     # TODO Fix this
        #     _potential_fn = model.get_potential(
        #         x_train, y_train, eff_num_data=len(x_train)
        #     )
        #     kernel = HMC(
        #         potential_fn=_potential_fn,
        #         adapt_step_size=False,
        #         adapt_mass_matrix=False,
        #         step_size=1e-3,
        #         num_steps=32,
        #     )
        #     mcmc = MCMC(
        #         kernel, num_samples=num_samples, initial_params=model.params_dict()
        #     )
        # else:
        if inference == "WenzelSGLD":
            runner_class = bnn_priors.inference.WenzelSGLDRunner
        elif inference == "VanillaSGLD":
            runner_class = bnn_priors.inference.VanillaSGLDRunner
        elif inference == "MongeSGLD":
            runner_class = bnn_priors.inference.MongeSGLDRunner
        elif inference == "pSGLD":
            runner_class = bnn_priors.inference.pSGLDRunner
        elif inference == "ShampooSGLD":
            runner_class = bnn_priors.inference.ShampooSGLDRunner
        else:
            raise Exception

        if batch_size is None:
            batch_size = len(data.norm.train)
        # Disable parallel loading for `TensorDataset`s.
        num_workers = (
            0 if isinstance(data.norm.train, t.utils.data.TensorDataset) else 2
        )
        dataloader = t.utils.data.DataLoader(
            data.norm.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )
        dataloader_val = t.utils.data.DataLoader(
            data.norm.val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        dataloader_test = t.utils.data.DataLoader(
            data.norm.test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        mcmc = runner_class(
            model=model,
            dataloader=dataloader,
            dataloader_val=dataloader_val,
            dataloader_test=dataloader_test,
            epochs_per_cycle=epochs_per_cycle,
            samples_per_epoch=samples_per_epoch,
            thinning=thinning,
            learning_rate=lr,
            sampling_decay=sampling_decay,
            cycles=cycles,
            temperature=temperature,
            precond_update=precond_update,
            metrics_saver=metrics_saver,
            model_saver=model_saver,
            metrics_skip=metrics_skip,
            reject_samples=reject_samples,
            grad_max=grad_max,
            monge_lambd=monge_lambd,
            monge_alpha_2=monge_alpha_2,
            rmsprop_alpha=rmsprop_alpha,
            rmsprop_eps=rmsprop_eps,
            skip_first=skip_first,
            burnin_batches=burnin_batches,
            calc_curvatures=calc_curvatures,
            batch_size=batch_size,
        )

        if calc_curvatures:
            evaluations, timestamps, curvatures = mcmc.run(progressbar=progressbar)
        else:
            evaluations, timestamps = mcmc.run(progressbar=progressbar)
    model.eval()

    batch_size = min(batch_size, len(data.norm.test))
    dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size)

    if uncertainty_plot:
        samples = mcmc.get_samples()
        dataloader = t.utils.data.DataLoader(data.norm.train, batch_size=batch_size)
        _, fig = evaluate_model(
            model, dataloader_test, samples, uncertainty_plot, dataloader=dataloader
        )
        fig.savefig(exp_utils.sneaky_artifact(_run, "uncertainty_plot.png"))

    final_evaluation = {
        "val": evaluations["val"][-1],
        "test": evaluations["test"][-1],
    }

    with open(exp_utils.sneaky_artifact(_run, "evaluations.pkl"), "wb") as f:
        pickle.dump(evaluations, f)
        pickle.dump(timestamps, f)
        if calc_curvatures:
            pickle.dump(curvatures, f)

    return final_evaluation
