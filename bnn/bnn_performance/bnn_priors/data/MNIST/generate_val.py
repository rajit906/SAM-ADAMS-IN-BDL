import numpy as np
import pickle
import os


np.random.seed(1)
_ROOT = os.path.abspath(os.path.dirname(__file__))
if os.path.isfile(f"{_ROOT}/index_val.pkl"):
    os.remove(f"{_ROOT}/index_val.pkl")

index_val = np.random.choice(np.arange(60000), size=10000, replace=False)

with open(f"{_ROOT}/index_val.pkl", "wb") as f:
    pickle.dump(index_val, f)
