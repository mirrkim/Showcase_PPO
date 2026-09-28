"""Runtime-only loader for the trained RSSI distance estimator."""

import numpy as np


MODEL_FILE = 'ann_model.npz'


class MLP6to3:
    """Small NumPy MLP retained in the format used by the trained model."""

    def __init__(self, hidden, rng):
        self.H = hidden
        self.D = 12
        self.dim = hidden * 12 + hidden + 3 * hidden + 3
        self.w = 0.05 * rng.standard_normal(self.dim)

    def unpack(self, weights):
        hidden = self.H
        index = 0
        w1 = weights[index:index + hidden * 12].reshape(hidden, 12)
        index += hidden * 12
        b1 = weights[index:index + hidden]
        index += hidden
        w2 = weights[index:index + 3 * hidden].reshape(3, hidden)
        index += 3 * hidden
        b2 = weights[index:index + 3]
        return w1, b1, w2, b2

    def forward(self, features, weights=None):
        if weights is None:
            weights = self.w
        w1, b1, w2, b2 = self.unpack(weights)
        hidden = np.maximum(features @ w1.T + b1, 0.0)
        return hidden @ w2.T + b2, None


def load_model(path):
    data = np.load(path)
    hidden = int(data['hidden'][0])
    model = MLP6to3(hidden=hidden, rng=np.random.default_rng(0))
    model.w = data['weights']
    return model, data['mu'], data['sd']
