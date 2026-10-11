import numpy as np

def make_two_groups(mean1, mean2, std, n_per_group, seed):
    """Make shuffled 1D data from two bell curves.

    Returns (heights, labels): label 0 = group 1, label 1 = group 2.
    """
    rng = np.random.default_rng(seed=seed)
    group1 = rng.normal(loc=mean1, scale=std, size=n_per_group)
    group2 = rng.normal(loc=mean2, scale=std, size=n_per_group)

    group1_labels = np.zeros(n_per_group)
    group2_labels = np.ones(n_per_group)

    unshuffled_labels = np.concatenate([group1_labels, group2_labels])
    unshuffled_heights = np.concatenate([group1, group2])

    shuffle_order = rng.permutation(len(unshuffled_heights))

    heights = unshuffled_heights[shuffle_order]
    labels = unshuffled_labels[shuffle_order]

    return heights, labels