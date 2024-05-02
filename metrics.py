import scipy.stats as stats
from partitioning import flatten


def wasserstein(p, q):
    p, q = flatten(p), flatten(q)
    return stats.wasserstein_distance(p, q)

# TODO: Implement way to estimate KL based on samples from continuous distributions
#       e.g. using KDE, fitting densities (check empirical distribution shapes - normal?)
#       or directly using ECDF https://www.semanticscholar.org/paper/Kullback-Leibler-divergence-estimation-of-Pérez-Cruz/310974d3c141589a7800d737e5859b76676dcb5d?p2df
#       WHY? -> KL is undefined if the support of the two distributions is not the same, so we can't just use the discrete values of the histograms
#       (or would need to define the bins in a way that the support is the same, but this would underestimate the KL)

# TODO (Peer): use the wasserstein metric to compute differences between order book volume samples (copy from LOBS5 project)
