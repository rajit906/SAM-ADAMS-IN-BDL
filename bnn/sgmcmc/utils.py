import numpy as np


def kl_divergence(p_probs, q_probs):
    assert p_probs.shape == q_probs.shape
    temp = p_probs * np.log(p_probs / q_probs)
    return np.sum(temp[~np.isnan(temp)])


if __name__ == "__main__":
    test1 = np.array([1 / 6, 0, 1 / 3, 1 / 3, 0, 1 / 6])
    test2 = np.array([1 / 3, 0, 2 / 3, 2 / 3, 0, 1 / 3])
    print(kl_divergence(test1, test2))
