from typing import Callable, Optional,Union
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from partitioning import flatten
import warnings

import scipy, time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats, integrate

import scipy.stats
from sklearn.neighbors import KernelDensity

from scipy.spatial import cKDTree


# --- JAX auto-detection for GPU-accelerated bootstrap ---
_USE_JAX = False
try:
    import jax
    import jax.numpy as jnp
    _USE_JAX = True
    # Log device at import time so we can verify GPU is used
    _jax_devices = jax.devices()
    print(f"[metrics] JAX available — devices: {_jax_devices}")
except ImportError:
    pass

# --- Bootstrap subsampling for large score_dfs ---
_MAX_BOOTSTRAP_SAMPLES = int(os.environ.get('BENCH_MAX_BOOTSTRAP_SAMPLES', '50000'))


def _subsample_for_bootstrap(score_df: pd.DataFrame, max_samples: int = None) -> pd.DataFrame:
    """Stratified subsample of score_df for bootstrap CI estimation.

    When score_df has millions of rows (e.g. 3.1M for spread), sorting 1.568M
    floats per bootstrap iteration dominates wall time. Subsampling to 50K
    preserves CI accuracy (bootstrap precision depends on n_bootstrap, not
    sample size) while enabling JAX GPU and reducing scipy sort cost 62x.
    """
    if max_samples is None:
        max_samples = _MAX_BOOTSTRAP_SAMPLES
    if max_samples <= 0 or len(score_df) <= max_samples:
        return score_df
    n_per_type = max_samples // 2
    sampled = score_df.groupby('type', group_keys=False).apply(
        lambda g: g.sample(n=min(n_per_type, len(g)), random_state=42)
    )
    print(f"[metrics] Bootstrap subsample: {len(score_df)} → {len(sampled)} samples", flush=True)
    return sampled.reset_index(drop=True)


def _use_jax_bootstrap(score_df: pd.DataFrame, n_bootstrap: int = 100) -> bool:
    """Check if JAX bootstrap is feasible for this score_df size.

    vmap materializes (n_bootstrap × n_samples) arrays for indices, sorted
    values, and intermediates.  XLA pre-allocates the entire computation graph
    including sort workspace (~2-4× data), staging buffers, and searchsorted
    intermediates.  We use ~500 bytes per (bootstrap, sample) to account for
    this — 10× more conservative than raw array sizes alone.
    """
    if not _USE_JAX:
        return False
    n = len(score_df)
    # XLA allocates ~500 bytes per (bootstrap, sample) including sort workspace,
    # staging buffers, searchsorted intermediates, and compilation overhead
    est_bytes = n_bootstrap * n * 500
    try:
        dev = jax.devices()[0]
        if hasattr(dev, 'memory_stats'):
            mem_stats = dev.memory_stats()
            if mem_stats and 'bytes_limit' in mem_stats:
                avail = int(mem_stats['bytes_limit'] * 0.80)
                use_jax = est_bytes < avail
                if not use_jax:
                    print(f"[metrics] JAX bootstrap skipped: {n} samples, "
                          f"est {est_bytes/1e9:.1f}GB > {avail/1e9:.1f}GB avail",
                          flush=True)
                return use_jax
        # Fallback: assume 90 GB (GH200 minus overhead)
        use_jax = est_bytes < 90_000_000_000
        if not use_jax:
            print(f"[metrics] JAX bootstrap skipped: {n} samples, "
                  f"est {est_bytes/1e9:.1f}GB > 90GB limit", flush=True)
        return use_jax
    except Exception:
        return est_bytes < 90_000_000_000


# --- JAX vmap bootstrap implementations ---

def _bootstrap_l1_jax(
    score_df: pd.DataFrame,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: float = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    """L1 bootstrap via jax.vmap — bincount over resampled group indices."""
    type_vals = score_df['type'].values
    real_mask = type_vals == 'real'
    groups = score_df['group'].values
    real_groups_np = groups[real_mask].astype(np.intp)
    gen_groups_np = groups[~real_mask].astype(np.intp)
    n_real, n_gen = len(real_groups_np), len(gen_groups_np)
    n_bins = int(max(real_groups_np.max(), gen_groups_np.max())) + 1 if (n_real > 0 and n_gen > 0) else 1

    # Generate indices with numpy RNG (for reproducibility with non-JAX path)
    real_idx_np = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_idx_np = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))

    # Transfer to JAX
    real_groups_j = jnp.array(real_groups_np, dtype=jnp.int32)
    gen_groups_j = jnp.array(gen_groups_np, dtype=jnp.int32)
    real_idx_j = jnp.array(real_idx_np, dtype=jnp.int32)
    gen_idx_j = jnp.array(gen_idx_np, dtype=jnp.int32)

    @jax.vmap
    def _one_bootstrap(r_idx, g_idx):
        r_counts = jnp.bincount(real_groups_j[r_idx], length=n_bins).astype(jnp.float32)
        g_counts = jnp.bincount(gen_groups_j[g_idx], length=n_bins).astype(jnp.float32)
        r_total = r_counts.sum()
        g_total = g_counts.sum()
        # Normalize (avoid division by zero — shouldn't happen with valid data)
        r_counts = jnp.where(r_total > 0, r_counts / r_total, r_counts)
        g_counts = jnp.where(g_total > 0, g_counts / g_total, g_counts)
        return jnp.abs(r_counts - g_counts).sum() / 2.0

    losses_j = _one_bootstrap(real_idx_j, gen_idx_j)  # (n_bootstrap,)

    if whole_data_loss is not None:
        losses_j = jnp.concatenate([jnp.array([whole_data_loss], dtype=jnp.float32), losses_j])

    ci = jnp.percentile(losses_j, jnp.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100]))
    return np.asarray(ci, dtype=np.float64), np.asarray(losses_j, dtype=np.float64)


def _bootstrap_wasserstein_jax(
    score_df: pd.DataFrame,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: float = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    """Wasserstein bootstrap via jax.vmap — sort-based 1D earth mover's distance."""
    real_scores_np = score_df.loc[score_df['type'] == 'real', 'score'].values.astype(np.float32)
    gen_scores_np = score_df.loc[score_df['type'] == 'generated', 'score'].values.astype(np.float32)
    n_real, n_gen = len(real_scores_np), len(gen_scores_np)

    # Same RNG call order as numpy path
    real_idx_np = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_idx_np = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))

    real_scores_j = jnp.array(real_scores_np)
    gen_scores_j = jnp.array(gen_scores_np)
    real_idx_j = jnp.array(real_idx_np, dtype=jnp.int32)
    gen_idx_j = jnp.array(gen_idx_np, dtype=jnp.int32)

    if n_real == n_gen:
        # Equal-size: Wasserstein = mean |sorted_r - sorted_g|
        @jax.vmap
        def _wasserstein_one(r_idx, g_idx):
            r = jnp.sort(real_scores_j[r_idx])
            g = jnp.sort(gen_scores_j[g_idx])
            return jnp.mean(jnp.abs(r - g))
    else:
        # Unequal-size: quantile-function approach on a fixed grid
        n_grid = max(n_real, n_gen)
        grid = jnp.linspace(0, 1, n_grid, endpoint=False) + 0.5 / n_grid

        @jax.vmap
        def _wasserstein_one(r_idx, g_idx):
            r = jnp.sort(real_scores_j[r_idx])
            g = jnp.sort(gen_scores_j[g_idx])
            # Quantile indices (floor-based lookup into sorted arrays)
            r_qi = jnp.clip((grid * n_real).astype(jnp.int32), 0, n_real - 1)
            g_qi = jnp.clip((grid * n_gen).astype(jnp.int32), 0, n_gen - 1)
            return jnp.mean(jnp.abs(r[r_qi] - g[g_qi]))

    losses_j = _wasserstein_one(real_idx_j, gen_idx_j)

    if whole_data_loss is not None:
        losses_j = jnp.concatenate([jnp.array([whole_data_loss], dtype=jnp.float32), losses_j])

    ci = jnp.percentile(losses_j, jnp.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100]))
    return np.asarray(ci, dtype=np.float64), np.asarray(losses_j, dtype=np.float64)


def _bootstrap_ks_jax(
    score_df: pd.DataFrame,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: float = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    """KS bootstrap via jax.vmap — CDF comparison using searchsorted."""
    real_scores_np = score_df.loc[score_df['type'] == 'real', 'score'].values.astype(np.float32)
    gen_scores_np = score_df.loc[score_df['type'] == 'generated', 'score'].values.astype(np.float32)
    n_real, n_gen = len(real_scores_np), len(gen_scores_np)

    # Same RNG call order
    real_idx_np = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_idx_np = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))

    real_scores_j = jnp.array(real_scores_np)
    gen_scores_j = jnp.array(gen_scores_np)
    real_idx_j = jnp.array(real_idx_np, dtype=jnp.int32)
    gen_idx_j = jnp.array(gen_idx_np, dtype=jnp.int32)

    @jax.vmap
    def _ks_one(r_idx, g_idx):
        r = jnp.sort(real_scores_j[r_idx])
        g = jnp.sort(gen_scores_j[g_idx])
        all_pts = jnp.concatenate([r, g])
        cdf_r = jnp.searchsorted(r, all_pts, side='right').astype(jnp.float32) / n_real
        cdf_g = jnp.searchsorted(g, all_pts, side='right').astype(jnp.float32) / n_gen
        return jnp.max(jnp.abs(cdf_r - cdf_g))

    losses_j = _ks_one(real_idx_j, gen_idx_j)

    if whole_data_loss is not None:
        losses_j = jnp.concatenate([jnp.array([whole_data_loss], dtype=jnp.float32), losses_j])

    ci = jnp.percentile(losses_j, jnp.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100]))
    return np.asarray(ci, dtype=np.float64), np.asarray(losses_j, dtype=np.float64)


# TODO: optimise this performance-wise:
def _bootstrap(
    score_df: pd.DataFrame,
    loss_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: Optional[float] = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    if whole_data_loss is not None:
        losses = [whole_data_loss]
    else:
        losses = []

    # get all real data
    real_idx = score_df.loc[score_df['type'] == 'real'].index
    n_real = len(real_idx)
    real_all = score_df.loc[real_idx]
    # get all generated data
    generated_idx = score_df.loc[score_df['type'] == 'generated'].index
    n_gen = len(generated_idx)
    gen_all = score_df.loc[generated_idx]
    # get indices for bootstrap samples
    real_btstr_idcx = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_btstr_idcx = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))
    # create individual samples
    for i in range(n_bootstrap):
        # draw bootstrap samples
        real_sample = real_all.iloc[real_btstr_idcx[i]]
        gen_sample = gen_all.iloc[gen_btstr_idcx[i]]
        score_df_sampled = pd.concat([real_sample, gen_sample], axis=0)

        # losses.append(loss_fn(real_sample.score, gen_sample.score))
        losses.append(loss_fn(score_df_sampled))
    losses = np.array(losses)

    # get the percentiles of the bootstrapped loss values
    q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
    ci = np.percentile(losses, q=q)

    return ci, losses

def _bootstrap_wasserstein(
    score_df: pd.DataFrame,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: Optional[float] = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    """Fast-path bootstrap for Wasserstein: numpy indexing, no DataFrame per iteration."""
    losses = [whole_data_loss] if whole_data_loss is not None else []
    real_scores = score_df.loc[score_df['type'] == 'real', 'score'].values.astype(float)
    gen_scores = score_df.loc[score_df['type'] == 'generated', 'score'].values.astype(float)
    n_real, n_gen = len(real_scores), len(gen_scores)
    # Same RNG call order as _bootstrap() for reproducibility
    real_idx = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_idx = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))
    for i in range(n_bootstrap):
        losses.append(stats.wasserstein_distance(
            real_scores[real_idx[i]], gen_scores[gen_idx[i]]))
    losses = np.array(losses)
    q = np.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100])
    ci = np.percentile(losses, q=q)
    return ci, losses


def _bootstrap_ks(
    score_df: pd.DataFrame,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: Optional[float] = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    """Fast-path bootstrap for KS: numpy indexing, no DataFrame per iteration."""
    losses = [whole_data_loss] if whole_data_loss is not None else []
    real_scores = score_df.loc[score_df['type'] == 'real', 'score'].values.astype(float)
    gen_scores = score_df.loc[score_df['type'] == 'generated', 'score'].values.astype(float)
    n_real, n_gen = len(real_scores), len(gen_scores)
    real_idx = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_idx = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))
    for i in range(n_bootstrap):
        losses.append(stats.ks_2samp(
            real_scores[real_idx[i]], gen_scores[gen_idx[i]]).statistic)
    losses = np.array(losses)
    q = np.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100])
    ci = np.percentile(losses, q=q)
    return ci, losses


def _bootstrap_l1(
    score_df: pd.DataFrame,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    whole_data_loss: Optional[float] = None,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> tuple[np.ndarray, np.ndarray]:
    """Fast-path bootstrap for L1: numpy bincount instead of pandas groupby."""
    losses = [whole_data_loss] if whole_data_loss is not None else []
    type_vals = score_df['type'].values
    real_mask = type_vals == 'real'
    groups = score_df['group'].values
    real_groups = groups[real_mask].astype(np.intp)
    gen_groups = groups[~real_mask].astype(np.intp)
    n_real, n_gen = len(real_groups), len(gen_groups)
    n_bins = int(max(real_groups.max(), gen_groups.max())) + 1 if (n_real > 0 and n_gen > 0) else 1
    # Same RNG call order as _bootstrap()
    real_idx = rng_np.integers(0, n_real, size=(n_bootstrap, n_real))
    gen_idx = rng_np.integers(0, n_gen, size=(n_bootstrap, n_gen))
    for i in range(n_bootstrap):
        r_counts = np.bincount(real_groups[real_idx[i]], minlength=n_bins).astype(float)
        g_counts = np.bincount(gen_groups[gen_idx[i]], minlength=n_bins).astype(float)
        r_sum, g_sum = r_counts.sum(), g_counts.sum()
        if r_sum > 0:
            r_counts /= r_sum
        if g_sum > 0:
            g_counts /= g_sum
        losses.append(np.abs(r_counts - g_counts).sum() / 2.0)
    losses = np.array(losses)
    q = np.array([ci_alpha / 2 * 100, 100 - ci_alpha / 2 * 100])
    ci = np.percentile(losses, q=q)
    return ci, losses


def wasserstein(
    score_df: pd.DataFrame,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    rng_np: np.random.Generator = np.random.default_rng(12345),
):
    def _wasserstein(score_df: pd.DataFrame):
        # p, q = flatten(p), flatten(q)
        p = score_df.loc[score_df['type'] == 'real', 'score']
        q = score_df.loc[score_df['type'] == 'generated', 'score']
        return stats.wasserstein_distance(p, q)

    if ('generated' not in score_df['type'].values) or ('real' not in score_df['type'].values):
        if bootstrap_ci:
            return np.nan, np.ones(2) * np.nan, np.ones(n_bootstrap+1) * np.nan
        return np.nan

    score_df = score_df.copy()
    score_df.score = (score_df.score - score_df.score.mean()) / score_df.score.std()

    w = _wasserstein(score_df)

    if bootstrap_ci:
        # Subsample for bootstrap CI (point estimate w already computed on full data)
        boot_df = _subsample_for_bootstrap(score_df)
        if _use_jax_bootstrap(boot_df):
            rng_state = rng_np.bit_generator.state
            try:
                ci, losses = _bootstrap_wasserstein_jax(boot_df, n_bootstrap, ci_alpha, w, rng_np)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                    print(f"[metrics] JAX OOM in wasserstein, falling back to numpy "
                          f"({len(boot_df)} samples)", flush=True)
                    rng_np.bit_generator.state = rng_state
                    ci, losses = _bootstrap_wasserstein(boot_df, n_bootstrap, ci_alpha, w, rng_np)
                else:
                    raise
        else:
            ci, losses = _bootstrap_wasserstein(boot_df, n_bootstrap, ci_alpha, w, rng_np)
        return w, ci, losses
    else:
        return w

def ks_distance(
    score_df: pd.DataFrame,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    rng_np: np.random.Generator = np.random.default_rng(12345),
):
    def _ks(score_df: pd.DataFrame):
        p = score_df.loc[score_df['type'] == 'real', 'score']
        q = score_df.loc[score_df['type'] == 'generated', 'score']
        return stats.ks_2samp(p, q).statistic

    if ('generated' not in score_df['type'].values) or ('real' not in score_df['type'].values):
        if bootstrap_ci:
            return np.nan, np.ones(2) * np.nan, np.ones(n_bootstrap+1) * np.nan
        return np.nan

    score_df = score_df.copy()
    score_df.score = (score_df.score - score_df.score.mean()) / score_df.score.std()

    ks = _ks(score_df)

    if bootstrap_ci:
        # Subsample for bootstrap CI (point estimate ks already computed on full data)
        boot_df = _subsample_for_bootstrap(score_df)
        if _use_jax_bootstrap(boot_df):
            rng_state = rng_np.bit_generator.state
            try:
                ci, losses = _bootstrap_ks_jax(boot_df, n_bootstrap, ci_alpha, ks, rng_np)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                    print(f"[metrics] JAX OOM in ks_distance, falling back to numpy "
                          f"({len(boot_df)} samples)", flush=True)
                    rng_np.bit_generator.state = rng_state
                    ci, losses = _bootstrap_ks(boot_df, n_bootstrap, ci_alpha, ks, rng_np)
                else:
                    raise
        else:
            ci, losses = _bootstrap_ks(boot_df, n_bootstrap, ci_alpha, ks, rng_np)
        return ks, ci, losses
    else:
        return ks

def l1_by_group(
    score_df: pd.DataFrame,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 100,
    ci_alpha: float = 0.01,
    rng_np: np.random.Generator = np.random.default_rng(12345),
) -> Union[float , tuple[float, np.ndarray, np.ndarray]]:
    """
    Takes a "score dataframe" with columns "score" (real numbers),
    "group" (+int), "type" ("real" or "generated")
    Returns the mean L1 distance between the number of scores in
    each group for the real and generated data.
    """
    def _calc_l1(score_df: pd.DataFrame):
        group_counts = score_df.groupby(['type', 'group']).count()
        group_counts = pd.merge(
            group_counts.loc['real'],
            group_counts.loc['generated'],
            on='group',
            how='outer'
        ).fillna(0)
        group_counts = group_counts[['score_x', 'score_y']]
        group_counts /= group_counts.sum(axis=0)
        return (group_counts.score_x - group_counts.score_y).abs().sum() / 2.

    if ('generated' not in score_df['type'].values) or ('real' not in score_df['type'].values):
        if bootstrap_ci:
            return 1, np.ones(2), np.ones(n_bootstrap+1)
        return 1.

    l1 = _calc_l1(score_df)

    if bootstrap_ci:
        # Subsample for bootstrap CI (point estimate l1 already computed on full data)
        boot_df = _subsample_for_bootstrap(score_df)
        if _use_jax_bootstrap(boot_df):
            rng_state = rng_np.bit_generator.state
            try:
                l1_ci, l1s = _bootstrap_l1_jax(boot_df, n_bootstrap, ci_alpha, l1, rng_np)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                    print(f"[metrics] JAX OOM in l1_by_group, falling back to numpy "
                          f"({len(boot_df)} samples)", flush=True)
                    rng_np.bit_generator.state = rng_state
                    l1_ci, l1s = _bootstrap_l1(boot_df, n_bootstrap, ci_alpha, l1, rng_np)
                else:
                    raise
        else:
            l1_ci, l1s = _bootstrap_l1(boot_df, n_bootstrap, ci_alpha, l1, rng_np)
        return l1, l1_ci, l1s
    else:
        return l1


def ob3Drepr(orderbook, row_index):
    '''
    Input is the orderbook, for example, `b_real`, `b_gene`, and the corresponding `row_index`.
    Output is the 3D reconstruction of the row in the orderbook corresponding to `row_index`.
    delta_mid_price: delta mid-price
    index:           relative price level where change occurs,
    quant:           change in quantity at that level
    '''
    price_level = 10
    delta_mid_price = orderbook.iloc[row_index,-1]
    lb_array=orderbook.iloc[row_index-1,:4*price_level].values.reshape(2*price_level,2)
    b_array=orderbook.iloc[row_index,:4*price_level].values.reshape(2*price_level,2)
    lb_unique = np.array([x for x in lb_array if not any(np.array_equal(x, y) for y in b_array)])
    b_unique = np.array([x for x in b_array if not any(np.array_equal(x, y) for y in lb_array)])
    first_lb_unique=lb_unique[0];first_b_unique=b_unique[0]
    index = np.where(b_array[:, 0] == first_b_unique[0])[0][0]
    quant = np.where(first_b_unique[0] == first_lb_unique[0], first_b_unique[1] - first_lb_unique[1], first_b_unique[1])
    result = np.array([delta_mid_price,index,quant])
    return result

# TODO: Implement way to estimate KL based on samples from continuous distributions
#       e.g. using KDE, fitting densities (check empirical distribution shapes - normal?)
#       or directly using ECDF https://www.semanticscholar.org/paper/Kullback-Leibler-divergence-estimation-of-Pérez-Cruz/310974d3c141589a7800d737e5859b76676dcb5d?p2df
#       WHY? -> KL is undefined if the support of the two distributions is not the same, so we can't just use the discrete values of the histograms
#       (or would need to define the bins in a way that the support is the same, but this would underestimate the KL)



def kl_divergence_kde(a, b):
    """
    Calculate the Kullback-Leibler (KL) divergence between the KDEs of two datasets.

    Args:
        a (np.ndarray): First dataset of shape (N, K), where N is the number of samples and K is the number of features.
        b (np.ndarray): Second dataset of shape (M, K), where M is the number of samples and K is the number of features.

    Returns:
        float: The KL divergence between the two datasets.
    """

    import scipy

    N = a.shape[0]
    M = b.shape[0]
    K = a.shape[1]
    assert a.shape[1] == b.shape[1]

    # Fit KDE models with adjusted bandwidth
    kde_a = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(a)
    kde_b = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(b)

    # Generate evaluation points based on the average number of samples
    num_points = int(np.mean([N, M]) * 50)  # Increased the number of points
    x_min = min(a.min(), b.min()) - 1
    x_max = max(a.max(), b.max()) + 1
    x = np.linspace(x_min, x_max, num_points)

    # Generate a grid of K-dimensional points for PDF evaluation
    grid = np.array(np.meshgrid(*([x] * K))).T.reshape(-1, K)

    log_pdf_a = kde_a.score_samples(grid)
    log_pdf_b = kde_b.score_samples(grid)

    pdf_a = np.exp(log_pdf_a)
    pdf_b = np.exp(log_pdf_b)

    # Add a small value to PDFs to avoid log(0)
    epsilon = 1e-10
    pdf_a += epsilon
    pdf_b += epsilon

    # Normalize PDFs
    pdf_a /= np.sum(pdf_a)
    pdf_b /= np.sum(pdf_b)

    # Calculate KL divergence
    kl_div = scipy.stats.entropy(pdf_a, pdf_b)

    return kl_div




def kl_divergence_PerezCruz(P, Q, eps=1e-11):
    '''
    Only work for one dimensional, e.g. P(N*1) and Q(M*1) as input

    Kullback-Leibler divergence estimation of continuous distributions
    Published in IEEE International Symposium… 6 July 2008 Mathematics
    F. Pérez-Cruz, Pedro E. Harunari, Ariel Yssou

    Codes based on Section II of Fernando Pérez-Cruz's paper
    "Kullback-Leibler Divergence Estimation of Continuous Distributions".
    From two independent datasets of continuous variables,
    the KLD (aka relative entropy) is estimated by the construction of
    cumulative probability distributions and the comparison between their slopes at specific points.
    Estimating the probability distributions and directly evaluating KLD's definition
    leads to a biased estimation, whereas the present method leads to an unbiased estimation.
    This is particularly important in practical applications due to the finitude of collected statistics.

    Func takes two datasets to estimate the relative entropy between their PDFs
    we use eps=10^-11, but it could be defined as < the minimal interval between data points.
    '''
    def cumcount_reduced(arr):
        '''Returns the step function value at each increment of the CDF'''
        sorted_arr = np.array(sorted(arr))
        counts = np.zeros(len(arr))

        rolling_count = 0
        for idx, elem in enumerate(sorted_arr):
            rolling_count += 1
            counts[idx] = rolling_count

        counts /= len(counts)
        counts -= (1 / (2 * len(counts)))

        return (sorted_arr, counts)
    P = sorted(P)
    Q = sorted(Q)

    P_positions, P_counts = cumcount_reduced(P)
    Q_positions, Q_counts = cumcount_reduced(Q)

    #definition of x_0 and x_{n+1}
    x_0 = np.min([P_positions[0], Q_positions[0]]) - 1
    P_positions = np.insert(P_positions, 0, [x_0])
    P_counts = np.insert(P_counts, 0, [0])
    Q_positions = np.insert(Q_positions, 0, [x_0])
    Q_counts = np.insert(Q_counts, 0, [0])

    x_np1 = np.max([P_positions[-1], Q_positions[-1]]) + 1
    P_positions = np.append(P_positions, [x_np1])
    P_counts = np.append(P_counts, [1])
    Q_positions = np.append(Q_positions, [x_np1])
    Q_counts = np.append(Q_counts, [1])

    f_P = interp1d(P_positions, P_counts)
    f_Q = interp1d(Q_positions, Q_counts)

    X = P_positions[1:-2]
    values = (f_P(X) - f_P(X - eps)) / (f_Q(X) - f_Q(X - eps))
    filt = ((values != 0.) & ~(np.isinf(values)) & ~(np.isnan(values)))
    values_filter = values[filt]
    out = (np.sum(np.log(values_filter)) / len(values_filter)) - 1.

    return out



def kl_divergence_knn(X, Y):
    """
    Estimate the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    X : 2D numpy array (n, d)
        Samples from distribution P, which typically represents the true distribution.
    Y : 2D numpy array (m, d)
        Samples from distribution Q, which typically represents the approximate distribution.

    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    """
    # Get important dimensions
    d = X.shape[1]  # number of dimensions, must be the same in X and Y
    n = X.shape[0]  # number of samples in X
    m = Y.shape[0]  # number of samples in Y

    # Get distances to nearest neighbors using cKDTree
    kdtree_X = cKDTree(X)
    kdtree_Y = cKDTree(Y)

    # Distances to the nearest neighbors within the same dataset
    r = kdtree_X.query(X, k=2, eps=0.01)[0][:, 1]  # distances to the 2nd closest neighbor
    s = kdtree_Y.query(X, k=1, eps=0.01)[0]  # distances to the closest neighbor

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign on the first term of the right hand side.
    return - np.sum(np.log(r / s)) * d / n + np.log(m / (n - 1))

# TODO (Peer): use the wasserstein metric to compute differences between order book volume samples (copy from LOBS5 project)

if __name__=="__main__":
    # Example usage
    N = 1011
    M = 1234
    # M = 211
    K = 1
    a = np.random.rand(N, K)
    b = np.random.rand(M, K)
    kl_divergence_kde(a, b)
    kl_divergence_PerezCruz(a, b)
