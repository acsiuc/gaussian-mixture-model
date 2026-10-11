import numpy as np


class GMM:

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
        rng = np.random.default_rng(seed = self.seed)
        shuffle_order = rng.permutation(len(datapoints))
        randomized_datapoints = datapoints[shuffle_order]

        self.means_ = randomized_datapoints[:self.k_curves]
        self.weights_ = np.ones(self.k_curves) / self.k_curves
        standard_deviation = np.std(datapoints)
        self.stds_ = np.ones(self.k_curves)*standard_deviation



    def predict(self, datapoints):
        pass
    def predict_proba(self, datapoints):
        pass

def bell_curve_height(x, mean, std):

    z = (x - mean)/std

    fx = 1 / (std * np.sqrt(2*np.pi)) * np.exp(-z**2/2)

    return fx