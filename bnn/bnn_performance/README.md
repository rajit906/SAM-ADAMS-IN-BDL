This is based on [bnn_priors](https://github.com/ratschlab/bnn_priors). For further information see README.md in the parent folder.

The following are parts from [the original README.md](https://github.com/ratschlab/bnn_priors/blob/main/README.md).

# Bayesian Neural Network Priors Revisited

This repository contains the code for the paper [Bayesian Neural Network Priors Revisited](https://arxiv.org/abs/2102.06571), as described in the accompanying paper [BNNpriors: A library for Bayesian neural network inference with different prior distributions](https://www.sciencedirect.com/science/article/pii/S2665963821000270).
It allows to perform SG-MCMC inference in BNNs with different architectures and priors on a range of tasks.

## Installation

After cloning the repository, the package can be installed from inside the main directory with

```sh
pip install -e .
```

The `-e` makes the installation be in "development mode", so any changes you
make to the code in the repository will be reflected in the `bnn_priors` package
you can import.

The code has run at some point with Python 3.6, 3.7 and 3.8.


## Running experiments

We are using `sacred` (https://github.com/IDSIA/sacred) to manage the experiments.

## Cite this work

If you are using this codebase in your work, please cite it as

```
@article{fortuin2021bnnpriors,
  title={{BNNpriors}: A library for {B}ayesian neural network inference with different prior distributions},
  author={Fortuin, Vincent and Garriga-Alonso, Adri{\`a} and van der Wilk, Mark and Aitchison, Laurence},
  journal={Software Impacts},
  volume={9},
  pages={100079},
  year={2021},
  publisher={Elsevier}
}
```

If you would also like to cite our results regarding different BNN priors, please cite

```
@article{fortuin2021bayesian,
  title={{B}ayesian neural network priors revisited},
  author={Fortuin, Vincent and Garriga-Alonso, Adri{\`a} and Wenzel, Florian and R{\"a}tsch, Gunnar and Turner, Richard and van der Wilk, Mark and Aitchison, Laurence},
  journal={arXiv preprint arXiv:2102.06571},
  year={2021}
}
```

Finally, if you would like to cite the GG-MC inference algorithm used in this package, please cite

```
@article{garriga2021exact,
  title={Exact langevin dynamics with stochastic gradients},
  author={Garriga-Alonso, Adri{\`a} and Fortuin, Vincent},
  journal={arXiv preprint arXiv:2102.01691},
  year={2021}
}
```
