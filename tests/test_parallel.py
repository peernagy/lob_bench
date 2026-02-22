"""
Quick validation of the parallelization changes.
Tests:
  1. Vectorized bootstrap fast paths match original _bootstrap()
  2. Module imports work
  3. (Optional) Small end-to-end with real data
"""
import sys, os, time
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Test 1: Bootstrap fast-path correctness ----------
print("=" * 60)
print("TEST 1: Vectorized bootstrap vs original _bootstrap()")
print("=" * 60)

import metrics
from scipy import stats

# Create synthetic score_df
rng = np.random.default_rng(42)
n_real, n_gen = 500, 500
scores_real = rng.normal(0, 1, n_real)
scores_gen = rng.normal(0.2, 1.1, n_gen)
groups_real = np.searchsorted(np.linspace(-3, 3, 20), scores_real, side='right')
groups_gen = np.searchsorted(np.linspace(-3, 3, 20), scores_gen, side='right')

score_df = pd.DataFrame({
    'score': np.concatenate([scores_real, scores_gen]),
    'group': np.concatenate([groups_real, groups_gen]),
    'type': ['real'] * n_real + ['generated'] * n_gen,
})

# --- Wasserstein ---
print("\n[Wasserstein]")
rng1 = np.random.default_rng(12345)
ci_old, losses_old = metrics._bootstrap(
    score_df,
    lambda df: stats.wasserstein_distance(
        df.loc[df['type'] == 'real', 'score'],
        df.loc[df['type'] == 'generated', 'score']),
    n_bootstrap=50, ci_alpha=0.01, whole_data_loss=0.123, rng_np=rng1)

rng2 = np.random.default_rng(12345)
ci_new, losses_new = metrics._bootstrap_wasserstein(
    score_df, n_bootstrap=50, ci_alpha=0.01, whole_data_loss=0.123, rng_np=rng2)

match_w = np.allclose(losses_old, losses_new, atol=1e-12)
print(f"  Losses match: {match_w}")
print(f"  CI match:     {np.allclose(ci_old, ci_new, atol=1e-12)}")
assert match_w, f"Wasserstein mismatch! max diff={np.max(np.abs(losses_old - losses_new))}"

# --- KS ---
print("\n[KS distance]")
rng1 = np.random.default_rng(12345)
ci_old, losses_old = metrics._bootstrap(
    score_df,
    lambda df: stats.ks_2samp(
        df.loc[df['type'] == 'real', 'score'],
        df.loc[df['type'] == 'generated', 'score']).statistic,
    n_bootstrap=50, ci_alpha=0.01, whole_data_loss=0.456, rng_np=rng1)

rng2 = np.random.default_rng(12345)
ci_new, losses_new = metrics._bootstrap_ks(
    score_df, n_bootstrap=50, ci_alpha=0.01, whole_data_loss=0.456, rng_np=rng2)

match_ks = np.allclose(losses_old, losses_new, atol=1e-12)
print(f"  Losses match: {match_ks}")
print(f"  CI match:     {np.allclose(ci_old, ci_new, atol=1e-12)}")
assert match_ks, f"KS mismatch! max diff={np.max(np.abs(losses_old - losses_new))}"

# --- L1 ---
print("\n[L1 by group]")

def _calc_l1_ref(score_df):
    group_counts = score_df.groupby(['type', 'group']).count()
    group_counts = pd.merge(
        group_counts.loc['real'],
        group_counts.loc['generated'],
        on='group', how='outer'
    ).fillna(0)
    group_counts = group_counts[['score_x', 'score_y']]
    group_counts /= group_counts.sum(axis=0)
    return (group_counts.score_x - group_counts.score_y).abs().sum() / 2.

rng1 = np.random.default_rng(12345)
ci_old, losses_old = metrics._bootstrap(
    score_df, _calc_l1_ref,
    n_bootstrap=50, ci_alpha=0.01, whole_data_loss=0.789, rng_np=rng1)

rng2 = np.random.default_rng(12345)
ci_new, losses_new = metrics._bootstrap_l1(
    score_df, n_bootstrap=50, ci_alpha=0.01, whole_data_loss=0.789, rng_np=rng2)

match_l1 = np.allclose(losses_old, losses_new, atol=1e-12)
print(f"  Losses match: {match_l1}")
print(f"  CI match:     {np.allclose(ci_old, ci_new, atol=1e-12)}")
assert match_l1, f"L1 mismatch! max diff={np.max(np.abs(losses_old - losses_new))}"


# ---------- Test 2: Bootstrap speed comparison ----------
print("\n" + "=" * 60)
print("TEST 2: Bootstrap speed (100 iterations, n=2000)")
print("=" * 60)

n_big = 2000
scores_r = rng.normal(0, 1, n_big)
scores_g = rng.normal(0.1, 1, n_big)
groups_r = np.searchsorted(np.linspace(-3, 3, 30), scores_r, side='right')
groups_g = np.searchsorted(np.linspace(-3, 3, 30), scores_g, side='right')
big_df = pd.DataFrame({
    'score': np.concatenate([scores_r, scores_g]),
    'group': np.concatenate([groups_r, groups_g]),
    'type': ['real'] * n_big + ['generated'] * n_big,
})

for name, old_fn, new_fn in [
    ("Wasserstein",
     lambda: metrics._bootstrap(big_df,
         lambda df: stats.wasserstein_distance(
             df.loc[df['type']=='real','score'], df.loc[df['type']=='generated','score']),
         n_bootstrap=100, rng_np=np.random.default_rng(1)),
     lambda: metrics._bootstrap_wasserstein(big_df, n_bootstrap=100, rng_np=np.random.default_rng(1))),
    ("KS",
     lambda: metrics._bootstrap(big_df,
         lambda df: stats.ks_2samp(
             df.loc[df['type']=='real','score'], df.loc[df['type']=='generated','score']).statistic,
         n_bootstrap=100, rng_np=np.random.default_rng(1)),
     lambda: metrics._bootstrap_ks(big_df, n_bootstrap=100, rng_np=np.random.default_rng(1))),
    ("L1",
     lambda: metrics._bootstrap(big_df, _calc_l1_ref, n_bootstrap=100, rng_np=np.random.default_rng(1)),
     lambda: metrics._bootstrap_l1(big_df, n_bootstrap=100, rng_np=np.random.default_rng(1))),
]:
    t0 = time.perf_counter()
    old_fn()
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    new_fn()
    t_new = time.perf_counter() - t0
    speedup = t_old / t_new if t_new > 0 else float('inf')
    print(f"  {name:12s}: old={t_old:.3f}s  new={t_new:.3f}s  speedup={speedup:.1f}x")


# ---------- Test 3: partitioning parallel import ----------
print("\n" + "=" * 60)
print("TEST 3: Parallel infrastructure imports")
print("=" * 60)

import partitioning
partitioning.set_n_workers(4)
print(f"  partitioning._N_WORKERS = {partitioning._N_WORKERS}")
partitioning.set_n_workers(1)
print(f"  Reset to {partitioning._N_WORKERS}")

import scoring
print(f"  scoring.run_benchmark signature has n_workers: "
      f"{'n_workers' in scoring.run_benchmark.__code__.co_varnames}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
