"""Attack results schema and alignment validation.

Defines the expected schema for attack_results.csv and provides
validation functions to ensure alignment with raw_results.csv.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


ATTACK_SCHEMA = [
    "source_run_id",
    "method_name",
    "image_id",
    "budget_ratio",
    "seed",
    "attack_name",
    "leakage_score",
    "attack_success_rate",
    "status",
    "error_message",
]

JOIN_KEYS = {"method_name", "image_id", "budget_ratio", "seed"}


def validate_attack_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that attack results DataFrame has required columns."""
    missing = [col for col in ATTACK_SCHEMA if col not in df.columns]
    return {
        "valid": len(missing) == 0,
        "missing_columns": missing,
        "total_columns": len(df.columns),
        "total_rows": len(df),
    }


def check_attack_raw_alignment(
    attack_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Check that attack results can be joined with raw results."""
    available_join_keys = list(JOIN_KEYS & set(attack_df.columns) & set(raw_df.columns))

    if not available_join_keys:
        return {
            "aligned": False,
            "error_message": "No common join keys found",
            "available_join_keys": [],
        }

    attack_keys = set(attack_df[available_join_keys].drop_duplicates().itertuples(index=False, name=None))
    raw_keys = set(raw_df[available_join_keys].drop_duplicates().itertuples(index=False, name=None))

    missing_in_raw = attack_keys - raw_keys
    extra_in_raw = raw_keys - attack_keys

    return {
        "aligned": len(missing_in_raw) == 0,
        "available_join_keys": sorted(available_join_keys),
        "attack_unique_keys": len(attack_keys),
        "raw_unique_keys": len(raw_keys),
        "missing_in_raw": len(missing_in_raw),
        "extra_in_raw": len(extra_in_raw),
    }


def add_attack_columns_to_raw_results(
    raw_df: pd.DataFrame,
    attack_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge attack metrics into raw results DataFrame."""
    available_join_keys = JOIN_KEYS & set(attack_df.columns) & set(raw_df.columns)
    if not available_join_keys:
        return raw_df

    attack_agg = attack_df.groupby(list(available_join_keys)).agg(
        mean_leakage_score=("leakage_score", "mean"),
        mean_attack_success_rate=("attack_success_rate", "mean"),
    ).reset_index()

    return raw_df.merge(attack_agg, on=list(available_join_keys), how="left")
