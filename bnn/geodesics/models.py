import numpy as np
from numpy.linalg import norm
from scipy import stats


def under_over_flow(sp, v, cut_off=np.finfo(float).eps, min_val=np.finfo(float).eps):
    # function checks for under and over flow for nonzero, non infinity variables
    # Note: This function is correct only for sp>=0, since sp<np.finfo(float).eps only for non negative values
    # min_val = 1e-13
    if np.isscalar(sp):
        if np.isinf(sp):
            sp = v
        if sp < cut_off:
            sp = min_val
    else:
        if np.isinf(sp).any():
            sp[np.isinf(sp)] = v[np.isinf(sp)]
        if (sp < cut_off).any():
            sp[sp < cut_off] = min_val
    return sp


def clip_derivative(u, threshold=10e10):
    mg = norm(u)  # Use the norm of Gradient or Hessian
    if (np.isfinite(mg)) and (mg > threshold):
        u = threshold * u / mg  # Gradient clipping-by-norm
    else:
        u[np.isnan(u)] = 1.0
    return u


def softplus(x):
    # numerical stable softplus
    sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    return sp


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def dsoftplus(x):
    dsp = sigmoid(x)
    return dsp


class funnel:
    def __init__(self, D=2, mmean=2.0, sig2v=15.0):
        self.sig2v = sig2v
        self.mmean = mmean
        self.D = D
        self.name = "funnel"
        self.alpha = 1.0

    def logp(self, x):
        """Funnel log density (vectorized)
        Parameters
        ----------
        M : class
            funnel class object
        x : numpy float
            first columns are points at which to evaluate
            last column is funnel parameter v at point
        Returns
        -------
        numpy float
            a numpy array of the log density at given points
        """
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        v = x[-1]
        sig2th = softplus(v)
        lp = (
            -(D1 * 0.5) * (np.log(2.0 * np.pi) + np.log(sig2th))
            - 0.5 * norm(x[:-1] - muth) ** 2.0 / sig2th
            - 0.5 * (np.log(2.0 * np.pi) + np.log(s2) + v**2.0 / s2)
        )
        return lp

    def dlogp(self, x):
        """Funnel partial derivatives log density
        Parameters
        ----------
        M : class
            funnel class object
        x : numpy float
            first columns are points at which to evaluate
            last column is funnel parameter v at point
        Returns
        -------
        numpy float
            a numpy array of the partial derivatives log density at given points
        """
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        v = x[-1]
        # u is a N, D matrix, in each row the gradient [d x, d a]
        u = np.zeros(self.D)
        sig2th = softplus(v)
        dsig2th = dsoftplus(v)
        u[0:D1] = -(x[0:D1] - muth) / sig2th
        u[-1] = (
            -D1 / 2.0 * dsig2th / sig2th
            + 0.5 * norm(x[0:D1] - muth) ** 2.0 / sig2th**2.0 * dsig2th
            - v / s2
        )
        u = clip_derivative(u)
        return u

    def d2logp(self, x):
        """Funnel Hessian of log density
        Parameters
        ----------
        M : class
            funnel class object
        x : numpy float
            first entry is points at which to evaluate
            second entry is funnel parameter v
        Returns
        -------
        numpy float
            a numpy array of the Hessian log density
        """
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        v = x[-1]
        U = np.zeros((self.D, self.D))
        sig2th = softplus(v)
        dsig2th = dsoftplus(v)
        d2sig2th = dsig2th - dsig2th**2.0
        d1 = d2sig2th / sig2th - dsig2th**2.0 / sig2th**2.0
        d2 = d2sig2th / sig2th**2.0 - 2.0 * dsig2th**2.0 / sig2th**3.0
        # Diagonal elements (Second derivatives)
        # First 0:D1 elements
        U[range(D1), range(D1)] = -np.ones((D1)) / sig2th
        # Last D1+1 element
        U[D1, D1] = -D1 / 2.0 * d1 + d2 * 0.5 * norm(x[0:D1] - muth) ** 2.0 - 1.0 / s2
        # Non diagonal entries (Cross derivatives)
        U[-1, range(D1)] = U[range(D1), -1] = (
            (x[0:D1] - muth) * dsig2th / (sig2th**2.0)
        )
        # Clip values
        U = clip_derivative(U)
        return U

    def hvp_logp(self, x, v):
        # Efficient Hessian vector product
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        y = x[-1]
        hpv = np.zeros(self.D)
        sig2th = softplus(y)
        dsig2th = dsoftplus(y)
        d2sig2th = dsig2th - dsig2th**2.0
        d1 = d2sig2th / sig2th - dsig2th**2.0 / sig2th**2.0
        d2 = d2sig2th / sig2th**2.0 - 2.0 * dsig2th**2.0 / sig2th**3.0
        hpv[0:D1] = (
            -1 / sig2th * v[0:D1] + (x[0:D1] - muth) * dsig2th / (sig2th**2.0) * v[-1]
        )
        hpv[D1] = np.dot((x[0:D1] - muth), v[0:D1]) * dsig2th / (sig2th**2.0)
        hpv[D1] += (
            -D1 / 2.0 * d1 + d2 * 0.5 * norm(x[0:D1] - muth) ** 2.0 - 1.0 / s2
        ) * v[-1]
        hpv = clip_derivative(hpv)  # Clip values
        return hpv

    def densities(self):
        D = self.D
        s2 = self.sig2v
        muth = self.mmean
        density1 = lambda x: stats.norm.pdf(x, 0, np.sqrt(s2))
        x = np.random.normal(loc=0, scale=np.sqrt(s2), size=10000)
        a = [
            np.random.normal(loc=muth, scale=np.sqrt(np.log(1 + np.exp(xi))))
            for xi in x
        ]
        kde = stats.gaussian_kde(a)
        density2 = lambda y: kde(y)
        return density2, density1
