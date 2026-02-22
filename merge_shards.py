#!/usr/bin/env python3
"""Merge LOBbench metric shards into a single results file.

Each shard .pkl contains (scores_dict, score_dfs_dict) keyed by metric name.
Since shards score disjoint metric subsets, merging is dict.update() across shards.

Usage:
    python merge_shards.py "results/scores/scores_uncond_*_shard*_12345.pkl" -o results/scores/scores_uncond_GOOG_merged.pkl
"""
import argparse
import gzip
import pickle
from glob import glob


def load_results(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def save_results(scores, score_dfs, path, protocol=-1):
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump((scores, score_dfs), f, protocol)


def main():
    parser = argparse.ArgumentParser(description="Merge LOBbench shard .pkl files")
    parser.add_argument("pattern", help="Glob pattern for shard files (e.g. 'results/scores/scores_uncond_*_shard*_12345.pkl')")
    parser.add_argument("--output", "-o", required=True, help="Output merged .pkl path")
    args = parser.parse_args()

    shard_files = sorted(glob(args.pattern))
    if not shard_files:
        print(f"[!] No files matched pattern: {args.pattern}")
        return 1

    print(f"[*] Merging {len(shard_files)} shards:")
    merged_scores = {}
    merged_dfs = {}

    for path in shard_files:
        print(f"  - {path}")
        scores, score_dfs = load_results(path)
        merged_scores.update(scores)
        if score_dfs is not None:
            merged_dfs.update(score_dfs)

    print(f"[*] Merged {len(merged_scores)} metrics: {sorted(merged_scores.keys())}")
    save_results(merged_scores, merged_dfs, args.output)
    print(f"[*] Saved to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
