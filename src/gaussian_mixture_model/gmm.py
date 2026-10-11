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

        for x in range(self.max_rounds):
            current_values = self.predict_proba(datapoints=datapoints)
            old_curves = self.means_.copy()

            for curve in range(self.k_curves):
                shares = current_values[:, curve]
                self.means_[curve] = (shares*datapoints).sum() / shares.sum()
                self.weights_[curve] = shares.sum()/len(datapoints)
                self.stds_[curve] = np.sqrt((shares * (datapoints - self.means_[curve])**2).sum() / shares.sum())

            if np.abs(self.means_ - old_curves).max() < self.tolerance:
                break

    def predict(self, datapoints):
        return np.argmax(self.predict_proba(datapoints), axis = 1)

    def predict_proba(self, datapoints):
        table = np.zeros((len(datapoints), self.k_curves))

        for curve in range(self.k_curves):
            
            table[:, curve] = self.weights_[curve] * bell_curve_height(datapoints, self.means_[curve], self.stds_[curve])

        row_sums = table.sum(axis=1, keepdims = True)

        return table/row_sums

    
            


        

def bell_curve_height(x, mean, std):

    z = (x - mean)/std

    fx = 1 / (std * np.sqrt(2*np.pi)) * np.exp(-z**2/2)

    return fx