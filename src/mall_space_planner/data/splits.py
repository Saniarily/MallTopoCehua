"""Leakage-safe dataset splitting.

The legacy repo split *by mall_id* (all floors of a mall stay in one split), which is the
correct grouping because floors of the same mall share every planning-condition feature
(they are city/mall level). We keep that protocol and additionally support stratification
on a categorical column (e.g. ``city_cluster``) so that small clusters are represented in
val/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    group_col: str

    def assert_no_leakage(self) -> None:
        g = self.group_col
        tr, va, te = set(self.train[g]), set(self.val[g]), set(self.test[g])
        assert not (tr & va), f"group leakage train/val: {list(tr & va)[:5]}"
        assert not (tr & te), f"group leakage train/test: {list(tr & te)[:5]}"
        assert not (va & te), f"group leakage val/test: {list(va & te)[:5]}"

    def summary(self) -> dict[str, int]:
        return {
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "test_rows": len(self.test),
            "train_groups": self.train[self.group_col].nunique(),
            "val_groups": self.val[self.group_col].nunique(),
            "test_groups": self.test[self.group_col].nunique(),
        }


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_col: str | None = None,
) -> SplitResult:
    """Split rows so that every group (e.g. mall) lands in exactly one partition.

    Args:
        df: Input frame.
        group_col: Column identifying the leakage group (``mall_id``).
        test_ratio: Fraction of *groups* for test.
        val_ratio: Fraction of *groups* for validation.
        seed: RNG seed.
        stratify_col: Optional categorical column; splitting is done within each stratum
            (the group's stratum is taken from its first row).
    """
    if test_ratio + val_ratio >= 1.0:
        raise ValueError("test_ratio + val_ratio must be < 1")
    rng = np.random.RandomState(seed)

    groups = df[[group_col]].drop_duplicates()
    if stratify_col is not None:
        first = df.groupby(group_col, sort=False)[stratify_col].first()
        groups = groups.assign(_stratum=groups[group_col].map(first).fillna("NA").astype(str))
    else:
        groups = groups.assign(_stratum="all")

    test_groups: list = []
    val_groups: list = []
    for _, sub in groups.groupby("_stratum", sort=True):
        ids = sub[group_col].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test_groups += ids[:n_test]
        val_groups += ids[n_test : n_test + n_val]

    test_set, val_set = set(test_groups), set(val_groups)
    is_test = df[group_col].isin(test_set)
    is_val = df[group_col].isin(val_set)
    res = SplitResult(
        train=df[~is_test & ~is_val].reset_index(drop=True),
        val=df[is_val].reset_index(drop=True),
        test=df[is_test].reset_index(drop=True),
        group_col=group_col,
    )
    res.assert_no_leakage()
    return res
