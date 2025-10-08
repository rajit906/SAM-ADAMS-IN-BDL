from tqdm import tqdm
import torch
from .utils import get_cosine_schedule
from . import mcmc
import math
from .exp_utils import evaluate_model, const_evaluate, get_curvature
import time


class WenzelSGLDRunner:
    def __init__(
        self,
        model,
        dataloader,
        dataloader_val,
        dataloader_test,
        epochs_per_cycle,
        samples_per_epoch,
        thinning,
        learning_rate=1e-2,
        temperature=1.0,
        data_mult=1.0,
        sampling_decay=True,
        grad_max=1e6,
        cycles=1,
        precond_update=None,
        metrics_saver=None,
        model_saver=None,
        reject_samples=False,
        metrics_skip=1,
        monge_lambd=None,
        monge_alpha_2=None,
        rmsprop_alpha=None,
        rmsprop_eps=None,
        skip_first=0,
        burnin_batches=300,
        calc_curvatures=True,
        batch_size=128,
    ):
        """
        Args:
            model (torch.Module, PriorMixin): BNN model to sample from
            num_data (int): Number of datapoints in training sest
            learning_rate (float): Initial learning rate
            temperature (float): Temperature for tempering the posterior
            data_mult (float): Effective replication of each datapoint (which is the usual approach to tempering in VI).
            sampling_decay (bool): Flag to control whether the learning rate should decay during sampling
            grad_max (float): maximum absolute magnitude of an element of the gradient
            cycles (int): Number of warmup and sampling cycles to perform
            precond_update (int): Number of steps after which the preconditioner should be updated. None disables the preconditioner.
            metrics_saver : HDF5Metrics to log metric with a certain name and value
        """
        self.model = model
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val
        self.dataloader_test = dataloader_test
        self.samples_per_epoch = samples_per_epoch
        self.samples_per_cycle = epochs_per_cycle * samples_per_epoch
        self.thinning = thinning
        self.epochs_per_cycle = epochs_per_cycle

        self.learning_rate = learning_rate
        self.temperature = temperature
        self.eff_num_data = len(dataloader.dataset) * data_mult
        self.sampling_decay = sampling_decay
        self.cycles = cycles
        self.precond_update = precond_update
        self.metrics_saver = metrics_saver
        self.model_saver = model_saver

        self.param_names, self._params = zip(*model.named_parameters())
        self.reject_samples = reject_samples
        self.grad_max = grad_max
        self.monge_lambd = monge_lambd
        self.monge_alpha_2 = monge_alpha_2
        self.rmsprop_alpha = rmsprop_alpha
        self.rmsprop_eps = rmsprop_eps
        self.skip_first = skip_first
        self.metrics_skip = metrics_skip
        self.burnin_batches = burnin_batches
        self.calc_curvatures = calc_curvatures
        self.batch_size = batch_size

        self.n_samples = self.cycles * self.samples_per_cycle - self.skip_first

        self.statistics = dict()
        self.statistics["val"] = dict()
        self.statistics["test"] = dict()

        # type of classification, "Categorical", "Normal" or "Bernoulli"
        self.preds_class = "None"
        # start index of the samples of the current cycle
        self.sample_index = 0
        # cache for samples of the current index
        self.samples_in_cycle = {
            name: torch.zeros(
                torch.Size([self.samples_per_cycle]) + p_or_b.shape,
                dtype=p_or_b.dtype,
            )
            for name, p_or_b in self.model.state_dict().items()
        }

    def _make_optimizer(self, params):
        return mcmc.WenzelSGLD(
            params=params,
            lr=self.learning_rate,
            num_data=self.eff_num_data,
            temperature=self.temperature,
            rmsprop_alpha=self.rmsprop_alpha,
            rmsprop_eps=self.rmsprop_eps,
        )

    def _make_scheduler(self, optimizer):
        if self.sampling_decay is True or self.sampling_decay == "cosine":
            schedule = get_cosine_schedule(len(self.dataloader) * self.epochs_per_cycle)
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=schedule
            )
        elif self.sampling_decay is False or self.sampling_decay == "stairs":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, 150 * len(self.dataloader), gamma=0.1
            )
        elif self.sampling_decay == "flat":
            # No-op scheduler
            return torch.optim.lr_scheduler.StepLR(optimizer, 2**30, gamma=1.0)
        elif self.sampling_decay == "decay_0.8":
            # decay by 0.8 every cycle
            return torch.optim.lr_scheduler.StepLR(
                optimizer, self.epochs_per_cycle * len(self.dataloader), gamma=0.8
            )
        elif self.sampling_decay == "decay_0.5":
            # decay by 0.5 every cycle
            return torch.optim.lr_scheduler.StepLR(
                optimizer, self.epochs_per_cycle * len(self.dataloader), gamma=0.5
            )
        raise ValueError(f"self.sampling_decay={self.sampling_decay}")

    def run(self, progressbar=False):
        """
        Runs the sampling on the model.

        Args:
            x (torch.tensor): Training input data
            y (torch.tensor): Training labels
            progressbar (bool): Flag that controls whether a progressbar is printed
        """
        if self.calc_curvatures:
            curvatures = []
        self.optimizer = self._make_optimizer(self._params)
        self.scheduler = self._make_scheduler(self.optimizer)

        self.metrics_saver.add_scalar("val/log_prob", math.nan, step=-1)
        self.metrics_saver.add_scalar("val/acc", math.nan, step=-1)

        step = (
            -1
        )  # used for `self.metrics_saver.add_scalar`, must start at 0 and never reset
        postfix = {}
        for cycle in range(self.cycles):
            self.n_samples_in_cycle = 0

            if progressbar:
                epochs = tqdm(
                    range(self.epochs_per_cycle),
                    position=0,
                    leave=True,
                    desc=f"Cycle {cycle}, Sampling",
                    mininterval=2.0,
                )
            else:
                epochs = range(self.epochs_per_cycle)

            for epoch in epochs:
                for i, (x, y) in enumerate(self.dataloader):
                    step += 1
                    store_metrics = (
                        i == 0 or step % self.metrics_skip == 0  # The start of an epoch
                    )

                    loss, acc, _ = self.step(
                        step,
                        x.to(self._params[0].device).detach(),
                        y.to(self._params[0].device).detach(),
                        store_metrics=store_metrics,
                    )

                    if progressbar and store_metrics:
                        postfix["train/loss"] = loss.item()
                        postfix["train/acc"] = acc.item()
                        epochs.set_postfix(postfix, refresh=False)

                    if (i + 1) % self.thinning == 0 and step > self.burnin_batches:
                        state_dict = self.model.state_dict()
                        self._save_sample(state_dict, step)

                if self.precond_update is not None and epoch % self.precond_update == 0:
                    self.optimizer.update_preconditioner()

                state_dict = self.model.state_dict()
                results = self._evaluate_model(state_dict, step)
                if progressbar:
                    postfix.update(results)
                    epochs.set_postfix(postfix, refresh=False)

                # Important to put here because no new metrics are added
                # Write metrics to disk every 10 seconds
                self.metrics_saver.flush(every_s=10)

            # evaluate the samples collected in the current cycle
            self.model.eval()
            device = self._params[0].device
            for idx_in_cache in range(self.n_samples_in_cycle):
                sample = dict(
                    (k, v[idx_in_cache]) for k, v in self.samples_in_cycle.items()
                )
                self.model.load_state_dict(sample)

                if self.calc_curvatures:
                    curvatures.append(
                        get_curvature(
                            model=self.model,
                            dataloader_test=self.dataloader_test,
                            data_fraction=0.1,
                            batch_size=self.batch_size,
                        )
                    )

                for (name, loader) in zip(
                    ["val", "test"], [self.dataloader_val, self.dataloader_test]
                ):
                    if hasattr(loader.dataset, "tensors"):
                        labels = loader.dataset.tensors[1].cpu()
                    elif hasattr(loader.dataset, "targets"):
                        labels = torch.tensor(loader.dataset.targets).cpu()
                    else:
                        raise ValueError("I cannot find the labels in the dataloader.")
                    N, *possibly_D = labels.shape

                    i = 0

                    for batch_i, (batch_x, batch_y) in enumerate(loader):
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        preds = self.model(batch_x)
                        if cycle == 0 and idx_in_cache == 0 and batch_i == 0:
                            if isinstance(preds, torch.distributions.Categorical):
                                if len(possibly_D) == 0:
                                    n_classes_or_funs = labels.max().item() + 1
                                else:
                                    (n_classes_or_funs,) = possibly_D
                                self.preds_class = "Categorical"
                            elif isinstance(preds, torch.distributions.Normal):
                                if len(possibly_D) == 0:
                                    n_classes_or_funs = 1
                                else:
                                    (n_classes_or_funs,) = possibly_D
                                self.preds_class = "Normal"
                            elif isinstance(preds, torch.distributions.Bernoulli):
                                if len(possibly_D) == 0:
                                    n_classes_or_funs = labels.max().item() + 1
                                else:
                                    (n_classes_or_funs,) = possibly_D
                                self.preds_class = "Bernoulli"
                            else:
                                raise ValueError(f"unknown likelihood {type(preds)}")

                            self.statistics[name]["lps"] = torch.zeros(
                                (self.n_samples, N), dtype=torch.float64, device="cpu"
                            )
                            self.statistics[name]["acc_data"] = torch.zeros(
                                (self.n_samples, N, n_classes_or_funs),
                                dtype=torch.float64,
                                device="cpu",
                            )

                        if isinstance(preds, torch.distributions.Categorical):
                            acc_data_batch = preds.logits  # B,n_classes
                            lps_batch = preds.log_prob(batch_y)
                        elif isinstance(preds, torch.distributions.Normal):
                            acc_data_batch = preds.mean  # B,n_latent
                            lps_batch = preds.log_prob(batch_y).sum(-1)
                        else:
                            acc_data_batch = preds.logits
                            lps_batch = preds.log_prob(batch_y.float()).flatten()

                        next_i = i + len(batch_x)
                        self.statistics[name]["lps"][
                            self.sample_index + idx_in_cache, i:next_i
                        ] = lps_batch.detach()
                        self.statistics[name]["acc_data"][
                            self.sample_index + idx_in_cache, i:next_i, :
                        ] = acc_data_batch.detach()
                        i = next_i

            self.sample_index += self.n_samples_in_cycle
            self.model.train()

        # Save metrics for the last sample
        (x, y) = next(iter(self.dataloader))
        self.step(
            step + 1,
            x.to(self._params[0].device),
            y.to(self._params[0].device),
            store_metrics=True,
        )

        evaluations = const_evaluate(
            self.model,
            self.dataloader_val,
            self.dataloader_test,
            self.statistics,
            self.preds_class,
            skip_first=self.skip_first,
            likelihood_eval=True,
            accuracy_eval=True,
            calibration_eval=True,
            samples_per_epoch=self.samples_per_epoch,
            num_epochs=self.cycles * self.epochs_per_cycle,
        )

        if self.calc_curvatures:
            return evaluations, curvatures
        else:
            return evaluations

    def _save_sample(self, state_dict, step):
        # save sample to model_saver if save_samples, and save sample to cache of the current cycle
        if self.model_saver is not None:
            self.model_saver.add_state_dict(state_dict, step)
            self.model_saver.flush()
        for name, param in state_dict.items():
            self.samples_in_cycle[name][self.n_samples_in_cycle] = param
        self.n_samples_in_cycle += 1

    def _evaluate_model(self, state_dict, step):
        if len(self.dataloader_val) == 0:
            return {}
        self.model.eval()
        state_dict = {k: v.unsqueeze(0) for k, v in state_dict.items()}
        results = evaluate_model(
            self.model,
            self.dataloader_val,
            state_dict,
            likelihood_eval=True,
            accuracy_eval=True,
            calibration_eval=False,
            uncertainty_plot=False,
        )
        self.model.train()

        results = {"val/loss": -results["lp_last"], "val/acc": results["acc_last"]}
        for k, v in results.items():
            self.metrics_saver.add_scalar(k, v, step)
        return results

    def _model_potential_and_grad(self, x, y):
        self.optimizer.zero_grad()
        loss, log_prior, potential, accs_batch, _ = self.model.split_potential_and_acc(
            x, y, self.eff_num_data
        )
        potential.backward()
        for p in self.optimizer.param_groups[0]["params"]:
            p.grad.clamp_(min=-self.grad_max, max=self.grad_max)
        if torch.isnan(potential).item():
            raise ValueError("Potential is NaN")
        return loss, log_prior, potential, accs_batch.mean()

    def step(self, i, x, y, store_metrics, lr_decay=True, initial_step=False):
        """
        Perform one step of SGHMC on the model.

        Args:
            x (torch.Tensor): Training input data
            y (torch.Tensor): Training labels
            lr_decay (bool): Flag that controls whether the learning rate should decay after this step

        Returns:
            loss (float): The current loss of the model for x and y
        """
        loss, log_prior, potential, acc = self._model_potential_and_grad(x, y)
        self.optimizer.step(calc_metrics=store_metrics)

        lr = self.optimizer.param_groups[0]["lr"]
        if lr_decay:
            self.scheduler.step()

        if store_metrics:
            # The metrics are valid for the previous step.
            # TODO: fix corresponds_to_sample
            self.store_metrics(
                i=i - 1,
                loss=loss.item(),
                log_prior=log_prior.item(),
                potential=potential.item(),
                acc=acc.item(),
                lr=lr,
                corresponds_to_sample=initial_step,
            )
        return loss, acc, None

    def get_samples(self):
        """
        Returns the acquired SGLD samples from the last run.

        Returns:
            samples (dict): Dictionary of torch.tensors with samples_per_cycle*cycles samples for each parameter of the model
        """
        if self.model_saver is None:
            raise Exception
        return self.model_saver.load_samples(keep_steps=False)

    def store_metrics(
        self,
        i,
        loss,
        log_prior,
        potential,
        acc,
        lr,
        corresponds_to_sample: bool,
        delta_energy=None,
        total_energy=None,
        rejected=None,
    ):
        add_scalar = self.metrics_saver.add_scalar
        for n, p in zip(self.param_names, self.optimizer.param_groups[0]["params"]):
            state = self.optimizer.state[p]
            try:
                add_scalar("preconditioner/" + n, state["preconditioner"], i)
            except:
                pass

        temperature = self.optimizer.param_groups[0]["temperature"]
        add_scalar("temperature", temperature, i)
        add_scalar("loss", loss, i)
        add_scalar("acc", acc, i)
        add_scalar("log_prior", log_prior, i)
        add_scalar("potential", potential, i)
        add_scalar("lr", lr, i)
        add_scalar("acceptance/is_sample", int(corresponds_to_sample), i)

        if delta_energy is not None:
            add_scalar("delta_energy", delta_energy, i)
            add_scalar("total_energy", total_energy, i)
        if rejected is not None:
            add_scalar("acceptance/rejected", int(rejected), i)


class VanillaSGLDRunner(WenzelSGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.VanillaSGLD(
            params=params,
            lr=self.learning_rate,
            num_data=self.eff_num_data,
            temperature=self.temperature,
        )

    def run(self, progressbar=False):
        """
        Runs the sampling on the model.

        Args:
            x (torch.tensor): Training input data
            y (torch.tensor): Training labels
            progressbar (bool): Flag that controls whether a progressbar is printed
        """
        if self.calc_curvatures:
            curvatures = []
        self.optimizer = self._make_optimizer(self._params)
        self.scheduler = self._make_scheduler(self.optimizer)

        self.metrics_saver.add_scalar("val/log_prob", math.nan, step=-1)
        self.metrics_saver.add_scalar("val/acc", math.nan, step=-1)

        step = (
            -1
        )  # used for `self.metrics_saver.add_scalar`, must start at 0 and never reset
        postfix = {}
        for cycle in range(self.cycles):
            self.n_samples_in_cycle = 0

            if progressbar:
                epochs = tqdm(
                    range(self.epochs_per_cycle),
                    position=0,
                    leave=True,
                    desc=f"Cycle {cycle}, Sampling",
                    mininterval=2.0,
                )
            else:
                epochs = range(self.epochs_per_cycle)

            for epoch in epochs:
                for i, (x, y) in enumerate(self.dataloader):
                    step += 1
                    store_metrics = (
                        i == 0 or step % self.metrics_skip == 0  # The start of an epoch
                    )

                    loss, acc, _ = self.step(
                        step,
                        x.to(self._params[0].device).detach(),
                        y.to(self._params[0].device).detach(),
                        store_metrics=store_metrics,
                    )

                    if progressbar and store_metrics:
                        postfix["train/loss"] = loss.item()
                        postfix["train/acc"] = acc.item()
                        epochs.set_postfix(postfix, refresh=False)

                    if (i + 1) % self.thinning == 0 and step > self.burnin_batches:
                        state_dict = self.model.state_dict()
                        self._save_sample(state_dict, step)

                state_dict = self.model.state_dict()
                results = self._evaluate_model(state_dict, step)
                if progressbar:
                    postfix.update(results)
                    epochs.set_postfix(postfix, refresh=False)

                # Important to put here because no new metrics are added
                # Write metrics to disk every 10 seconds
                self.metrics_saver.flush(every_s=10)

            self.model.eval()
            device = self._params[0].device
            for idx_in_cache in range(self.n_samples_in_cycle):
                sample = dict(
                    (k, v[idx_in_cache]) for k, v in self.samples_in_cycle.items()
                )
                self.model.load_state_dict(sample)

                if self.calc_curvatures:
                    curvatures.append(
                        get_curvature(
                            model=self.model,
                            dataloader_test=self.dataloader_test,
                            data_fraction=0.1,
                            batch_size=self.batch_size,
                        )
                    )

                for (name, loader) in zip(
                    ["val", "test"], [self.dataloader_val, self.dataloader_test]
                ):
                    if hasattr(loader.dataset, "tensors"):
                        labels = loader.dataset.tensors[1].cpu()
                    elif hasattr(loader.dataset, "targets"):
                        labels = torch.tensor(loader.dataset.targets).cpu()
                    else:
                        raise ValueError("I cannot find the labels in the dataloader.")
                    N, *possibly_D = labels.shape

                    i = 0

                    for batch_i, (batch_x, batch_y) in enumerate(loader):
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        preds = self.model(batch_x)
                        if cycle == 0 and idx_in_cache == 0 and batch_i == 0:
                            if isinstance(preds, torch.distributions.Categorical):
                                if len(possibly_D) == 0:
                                    n_classes_or_funs = labels.max().item() + 1
                                else:
                                    (n_classes_or_funs,) = possibly_D
                                self.preds_class = "Categorical"
                            elif isinstance(preds, torch.distributions.Normal):
                                if len(possibly_D) == 0:
                                    n_classes_or_funs = 1
                                else:
                                    (n_classes_or_funs,) = possibly_D
                                self.preds_class = "Normal"
                            elif isinstance(preds, torch.distributions.Bernoulli):
                                if len(possibly_D) == 0:
                                    n_classes_or_funs = 1
                                else:
                                    (n_classes_or_funs,) = possibly_D
                                self.preds_class = "Bernoulli"
                            else:
                                raise ValueError(f"unknown likelihood {type(preds)}")

                            self.statistics[name]["lps"] = torch.zeros(
                                (self.n_samples, N), dtype=torch.float64, device="cpu"
                            )
                            self.statistics[name]["acc_data"] = torch.zeros(
                                (self.n_samples, N, n_classes_or_funs),
                                dtype=torch.float64,
                                device="cpu",
                            )

                        if isinstance(preds, torch.distributions.Categorical):
                            acc_data_batch = preds.logits  # B,n_classes
                            lps_batch = preds.log_prob(batch_y)
                        elif isinstance(preds, torch.distributions.Normal):
                            acc_data_batch = preds.mean  # B,n_latent
                            lps_batch = preds.log_prob(batch_y).sum(-1)
                        else:
                            acc_data_batch = preds.logits
                            lps_batch = preds.log_prob(batch_y.float()).flatten()

                        next_i = i + len(batch_x)
                        self.statistics[name]["lps"][
                            self.sample_index + idx_in_cache, i:next_i
                        ] = lps_batch.detach()
                        self.statistics[name]["acc_data"][
                            self.sample_index + idx_in_cache, i:next_i, :
                        ] = acc_data_batch.detach()
                        i = next_i

            self.sample_index += self.n_samples_in_cycle
            self.model.train()

        # Save metrics for the last sample
        (x, y) = next(iter(self.dataloader))
        self.step(
            step + 1,
            x.to(self._params[0].device),
            y.to(self._params[0].device),
            store_metrics=True,
        )

        evaluations = const_evaluate(
            self.model,
            self.dataloader_val,
            self.dataloader_test,
            self.statistics,
            self.preds_class,
            skip_first=self.skip_first,
            likelihood_eval=True,
            accuracy_eval=True,
            calibration_eval=True,
            samples_per_epoch=self.samples_per_epoch,
            num_epochs=self.cycles * self.epochs_per_cycle,
        )

        if self.calc_curvatures:
            return evaluations, curvatures
        else:
            return evaluations

    def store_metrics(
        self,
        i,
        loss,
        log_prior,
        potential,
        acc,
        lr,
        corresponds_to_sample: bool,
        delta_energy=None,
        total_energy=None,
        rejected=None,
    ):
        add_scalar = self.metrics_saver.add_scalar

        temperature = self.optimizer.param_groups[0]["temperature"]
        add_scalar("temperature", temperature, i)
        add_scalar("loss", loss, i)
        add_scalar("acc", acc, i)
        add_scalar("log_prior", log_prior, i)
        add_scalar("potential", potential, i)
        add_scalar("lr", lr, i)
        add_scalar("acceptance/is_sample", int(corresponds_to_sample), i)

        if delta_energy is not None:
            add_scalar("delta_energy", delta_energy, i)
            add_scalar("total_energy", total_energy, i)
        if rejected is not None:
            add_scalar("acceptance/rejected", int(rejected), i)


class pSGLDRunner(VanillaSGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.pSGLD(
            params=params,
            lr=self.learning_rate,
            num_data=self.eff_num_data,
            temperature=self.temperature,
            rmsprop_alpha=self.rmsprop_alpha,
            rmsprop_eps=self.rmsprop_eps,
        )


class MongeSGLDRunner(VanillaSGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.MongeSGLD(
            params=params,
            lr=self.learning_rate,
            num_data=self.eff_num_data,
            temperature=self.temperature,
            monge_lambd=self.monge_lambd,
            monge_alpha_2=self.monge_alpha_2,
        )


class ShampooSGLDRunner(VanillaSGLDRunner):
    def _make_optimizer(self, params):
        return mcmc.ShampooSGLD(
            params=params,
            lr=self.learning_rate,
            num_data=self.eff_num_data,
            temperature=self.temperature,
            rmsprop_alpha=self.rmsprop_alpha,
            rmsprop_eps=self.rmsprop_eps,
        )
