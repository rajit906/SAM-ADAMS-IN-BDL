# Code for the paper Scalable Stochastic Gradient Riemannian Langevin Dynamics in Non-Diagonal Metrics

This repository contains code that can be used to reproduce the experiments as presented in the paper [Scalable Stochastic Gradient Riemannian Langevin Dynamics in Non-Diagonal Metrics](https://openreview.net/pdf?id=dXAuvo6CGI), published in Transactions on Machine Learning Research. The detailed descriptions for each part, along with notes on implementation, can be found in this README.md.

## Geodesics induced by Monge

Plot the geodesics induced by Monge metric on funnel (see Figure 1). Gradients are exact, and we set $\alpha^{2}=1.0$.

See directory **geodesics**.

Python version is `3.10.11`. The main dependencies are:

```
scipy==1.10.1
matplotlib==3.7.1
numpy==1.24.3
```

Create the folder **geodesics/figs**. Run

```
python sampling.py
```

in directory **geodesics**, and the resulting figures can be found in **geodesics/figs**.

## Sampling from funnel

Plot the results of sampling from funnel with identity metric, RMSprop metric, Monge metric and Shampoo metric (see Figure 2). Gradients are corrupted by noise.

See directory **sgmcmc**.

Python version is `3.10.11`. The main dependencies are:

```
scipy==1.10.1
matplotlib==3.7.1
numpy==1.24.3
```

Create the folder **sgmcmc/figs**. Run 

```
python two_d.py
```

in directory **sgmcmc**, and the resulting figures can be found in **sgmcmc/figs**.

## Note on code for neural network experiments

The provided code may not work (correctly) for scenarios not considered in our paper, e.g. different datasets, different thinning intervals, etc. In order to extend the code to other datasets, different from the original bnn_priors, you need to specify a validation set.

## Neural network experiments

Our code is based on [bnn_priors](https://github.com/ratschlab/bnn_priors). Following their practice, we used [sacred](https://github.com/IDSIA/sacred) to record the experiments. The code for ShampooSGLD is partly based on [here](https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch). The code for computing the curvature is partly based on [here](https://github.com/kylematoba/lcnn/blob/main/estimate_curvature.py).

Here we provide the raw scripts for running experiments.

Concerning the environment used to run the experiments, Python version is `3.9.13`.

Dependencies as recorded by sacred are

```
"bnn-priors==0.1.0",
"matplotlib==3.6.2",
"numpy==1.23.5",
"pyro-ppl==1.8.2",
"sacred==0.8.2",
"torch==1.13.0+cu116"
```

Here `bnn_priors` refers to our customized version. Some options they provide may not work in our codebase, and we have some custom options.

Some options used in the below instructions are

* `model_name` is `classificationdensenet` for MNIST experiments and `googleresnet` for CIFAR10 Gaussian prior experiments or `correlatedgoogleresnet` for CIFAR10 correlated  Normal prior experiments,
* `data_name` is `mnist` for MNIST experiments and `cifar10` for CIFAR10 experiments,
* available `inference_method_name` are `VanillaSGLD` (identity metric), `WenzelSGLD` (Wenzel metric), `pSGLD` (RMSprop metric), `MongeSGLD` (Monge metric) and `ShampooSGLD` (Shampoo metric),
* `lrs` are learning rates, given as numbers, e.g. `0.1`,
* `num_trials` is number of repeated trials to run,
* `prior_name` is name of prior, e.g. `gaussian`, `horseshoe`, `convcorrnormal`
* `width_for_mnist` is width of the network for MNIST experiments and can be specified to an arbitrary number (e.g. 50) for CIFAR10,
* `other_args` are other arguments, specifically for `MongeSGLD` is `monge_alpha_2={number}`, where `number` is the value for ``\alpha^2``.

### Evaluating performances

For evaluating the performances of the samplers, install the `bnn_priors` version as provided in `bnn_performance`.

Run

```
python bnn_performance/experiments/train_experiments.py --model model_name --data dataset_name --inference inference_method_name --lrs lrs --trials num_trials --prior prior_name --temperature 1.0 --sampling_decay flat --batch_size 100 --width width_for_mnist --save_samples False --cycles 20 --burnin_batches 1000 (--other_args other_args)
```

in the current directory, where `()` denote optional.

### Evaluating running times

For evaluating the running times of the samplers, install the `bnn_priors` version as provided in `bnn_time`.

Run

```
python bnn_time/experiments/train_experiments.py --model model_name --data dataset_name --inference inference_method_name --lrs lrs --trials num_trials --prior prior_name --temperature 1.0 --sampling_decay flat --batch_size 100 --width width_for_mnist --save_samples False --cycles 1 --burnin_batches 1000 (--other_args other_args)
```

in the current directory, where `()` denote optional.

### Evaluating average confidences

For evaluating the average confidences of the samplers with MNIST, hidden unit size 400, horseshoe prior, install the `bnn_priors` version as provided in `bnn_time`.

Run

```
python bnn_confs/experiments/train_experiments.py --model classificationdensenet --data MNIST --inference inference_method_name --lrs lrs --trials 3 --prior horseshoe --temperature 1.0 --sampling_decay flat --batch_size 100 --width 400 --save_samples False --cycles 20 --burnin_batches 1000 (--other_args other_args)
```

in the current directory, where `()` denote optional.

### Viewing obtained results

The scripts that can be used to view the results as reported in the paper are `plot_evaluations.ipynb`, `plot_experiments_results.ipynb`, `compare_time.ipynb` and `get_confs.ipynb` in directory `final_results`.

* `plot_evaluations.ipynb` generates Figure 3
* `plot_experiments_results.ipynb` generates results of MNIST and CIFAR10 experiments and Figure 4
* `compare_time.ipynb` generates results as presented in Table 6 and Table 7
* `get_confs.ipynb` generates results as presented in Table 10

The environment used to view the results is slightly different from the environment used to obtain the results.

## Note on practical implementation of the inverse square root of Monge metric

In the main paper and the experiments, the inverse square root of Monge metric contains 1 divided by the norm of the moving average of the gradients, which can possibly cause numerical issues. We note that, based on previous experiences, one option to make the inverse square root of Monge metric more numerically stable is to set $f_{-\frac{1}{2}}(x) = -\frac{\alpha^{2}}{2}$ when $\Vert x \Vert^{2}$ is smaller than 1e-12.

However, we did not observe runs that failed due to division by zero when running the experiments. For sampling from funnel, with the specific random seed, $\Vert x \Vert^{2}$ never goes below 1e-7. For neural networks, we repeat the experiments on settings (a setting refers to a specific combination of network structure and prior) where MongeSGLD improves upon VanillaSGLD, using the same learning rates and $\alpha^{2}$ that resulted in the best performances. For each setting, with the specific 10 random seeds of the 10 independent runs as generated and recorded by sacred, whose results are reported in Table 1 and Table 2 of the paper, $\Vert x \Vert^{2}$ never goes below 1e-3. We therefore conclude that generally there are no noticable numerical issues with the cases considered in the paper.
