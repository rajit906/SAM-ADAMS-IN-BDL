import os
import torch as t
import numpy as np
from torch.utils.data import TensorDataset

from bnn_priors.data import Dataset
from sklearn.datasets import make_moons, make_regression

__all__ = ('RandomData', 'Synthetic', 'RandomOODTestData', 'Custom', 'TwoMoons', 'Regression')


class RandomData:
    """
    The usage is:
    ```
    data = RandomData(dim=64, n_points=2000)
    ```
    e.g. normalized training dataset:
    ```
    data.norm.train
    ```
    """

    def __init__(self, dim=20, n_points=2000, dtype='float32', device="cpu"):
        X_unnorm = t.from_numpy(np.random.uniform(
            low=-1., high=1., size=[n_points, dim]).astype(dtype))
        y_unnorm = t.from_numpy(np.random.uniform(
            low=-1., high=1., size=[n_points, 1]).astype(dtype))

        # split into train and test
        index_train = np.arange(n_points//2)
        index_test = np.arange(n_points//2, n_points)

        # record unnormalized dataset
        self.unnorm = Dataset(X_unnorm, y_unnorm,
                              index_train, index_test, device)

        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, 0)
        self.X_std[self.X_std == 0] = 1.  # ensure we don't divide by zero
        self.X_mean = t.mean(self.unnorm.train_X, 0)

        self.y_mean = t.mean(self.unnorm.train_y)
        self.y_std = t.std(self.unnorm.train_y)

        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        y_norm = (self.unnorm.y - self.y_mean)/self.y_std

        self.norm = Dataset(X_norm, y_norm, index_train, index_test, device)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]

    def denormalize_y(self, y):
        return self.y_std * y + self.y_mean


class RandomOODTestData(RandomData):
    def __init__(self, dim=20, n_points=2000, dtype='float32', device="cpu"):
        len_train = n_points//2
        X_unnorm = t.from_numpy(np.random.uniform(
            low=-1., high=1., size=[len_train, dim]).astype(dtype))
        y_unnorm = t.from_numpy(np.random.uniform(
            low=-1., high=1., size=[len_train, 1]).astype(dtype))

        X_test_unnorm = t.from_numpy(np.random.uniform(
            low=1., high=2., size=[n_points-len_train, dim]).astype(dtype))
        y_test_unnorm = t.from_numpy(np.random.uniform(
            low=1., high=2., size=[n_points-len_train, 1]).astype(dtype))

        X_unnorm = t.cat([X_unnorm, X_test_unnorm])
        y_unnorm = t.cat([y_unnorm, y_test_unnorm])

        index_train = np.arange(len_train)
        index_test = np.arange(len_train, n_points)

        # record unnormalized dataset
        self.unnorm = Dataset(X_unnorm, y_unnorm,
                              index_train, index_test, device)

        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, 0)
        self.X_std[self.X_std == 0] = 1.  # ensure we don't divide by zero
        self.X_mean = t.mean(self.unnorm.train_X, 0)

        self.y_mean = t.mean(self.unnorm.train_y)
        self.y_std = t.std(self.unnorm.train_y)

        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        y_norm = (self.unnorm.y - self.y_mean)/self.y_std

        self.norm = Dataset(X_norm, y_norm, index_train, index_test, device)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]


class Synthetic:
    """
    The usage is:
    ```
    synth_data = Synthetic(dataset=data, model=net)
    ```
    e.g. normalized training dataset:
    ```
    synth_data.norm.train
    ```
    """

    def __init__(self, dataset, model, batch_size=None, dtype='float32', device="cpu"):
        if batch_size is None:
            new_y = model(dataset.norm.X).sample()
        else:
            dataloader_train = t.utils.data.DataLoader(
                dataset.norm.train, batch_size=batch_size)
            dataloader_test = t.utils.data.DataLoader(
                dataset.norm.test, batch_size=batch_size)
            batch_preds = []
            for dataloader in [dataloader_train, dataloader_test]:
                for batch_x, _ in dataloader:
                    batch_preds.append(model(batch_x).sample())
            new_y = t.cat(batch_preds)

        # split into train and test
        index_train = np.arange(len(dataset.norm.train_X))
        index_test = np.arange(len(dataset.norm.train_X), len(dataset.norm.X))

        # record unnormalized dataset
        self.unnorm = Dataset(dataset.unnorm.X, new_y,
                              index_train, index_test, device)
        self.norm = Dataset(dataset.norm.X, new_y,
                            index_train, index_test, device)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]


class Custom:
    def __init__(self, n_points=20, dtype='float32', device="cpu"):
        np.random.seed(1)
        xs = np.random.random(n_points*2)
        xs = np.where(xs < 0.6, xs * 10 - 10, xs * 10)
        def f(x): return x ** 3 + x ** 2 - x + 10
        ys = f(xs)

        X_unnorm = t.from_numpy(xs.astype(dtype)).unsqueeze(1)
        y_unnorm = t.from_numpy(ys.astype(dtype)).unsqueeze(1)

        index_train = np.arange(n_points)
        index_test = np.arange(n_points, n_points*2)

        # record unnormalized dataset
        self.unnorm = Dataset(X_unnorm, y_unnorm,
                              index_train, index_test, device)

        # compute normalization constants based on training set
        self.X_std = t.std(self.unnorm.train_X, 0)
        self.X_std[self.X_std == 0] = 1.  # ensure we don't divide by zero
        self.X_mean = t.mean(self.unnorm.train_X, 0)

        self.y_mean = t.mean(self.unnorm.train_y)
        self.y_std = t.std(self.unnorm.train_y)

        X_norm = (self.unnorm.X - self.X_mean)/self.X_std
        y_norm = (self.unnorm.y - self.y_mean)/self.y_std

        self.norm = Dataset(X_norm, y_norm, index_train, index_test, device)

        self.num_train_set = self.unnorm.X.shape[0]
        self.in_shape = self.unnorm.X.shape[1:]
        self.out_shape = self.unnorm.y.shape[1:]


class TwoMoons:
    def __init__(self, n_points=300, dtype_x='float32', dtype_y='int64', device="cpu"):
        np.random.seed(1)
        Xs, ys = make_moons(n_samples=n_points*2, random_state=1)

        index_train = np.arange(n_points)
        index_test = np.arange(n_points, n_points*2)

        X_norm = t.from_numpy(Xs.astype(dtype_x))
        y_norm = t.from_numpy(ys.astype(dtype_y))

        self.norm = Dataset(X_norm, y_norm, index_train, index_test, device)

        self.num_train_set = n_points
        self.in_shape = 2
        self.out_shape = 2


class Regression:
    def __init__(self, n_points=2000, dtype_x='float32', dtype_y='float32', device='cpu'):
        seed = 1
        bias = 1.
        prior_sigma = 1.
        noise_sigma = 1.
        X, y, _ = make_regression(n_samples=2000, n_features=200, n_informative=100, bias=bias, coef=True, random_state=seed, noise=noise_sigma)
        X = np.c_[X, np.ones(X.shape[0])]

        index_train = np.arange(n_points)
        index_test = np.arange(n_points)

        X_norm = t.from_numpy(X.astype(dtype_x))
        y_norm = t.from_numpy(y.astype(dtype_y))

        self.norm = Dataset(X_norm, y_norm, index_train, index_test, device)

        self.num_train_set = n_points
        self.in_shape = X.shape[1]
        self.out_shape = 1
