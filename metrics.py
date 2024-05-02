import scipy.stats as stats
from partitioning import flatten


def wasserstein(p, q):
    p, q = flatten(p), flatten(q)
    return stats.wasserstein_distance(p, q)
