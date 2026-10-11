import numpy as np


class GMM():

    k_curves: int
    tolerance: float
    means_: np.ndarray
    stds_: np.ndarray
    weights_: np.ndarray
    seed: int
    max_rounds: int


    def __init__(self, k_curves, tolerance, seed, max_rounds):
        self.k_curves = k_curves
        self.tolerance = tolerance
        self.seed = seed
        self.max_rounds = max_rounds

    def fit(self, datapoints):
        pass
    def predict(self, datapoints):
        pass
    def predict_proba(self, datapoints):
        pass