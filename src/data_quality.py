import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


def single_dataset_quality_report(df: pd.DataFrame, key_columns=None) -> dict:
    report = {
        "row_count": df.shape[0],
        "column_count": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
        "unique_values_per_column": df.nunique().to_dict(),
    }

    anomalies = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            neg_vals = int((df[col] < 0).sum())
            if neg_vals > 0:
                anomalies[col] = neg_vals
    report["anomalies_negative_values"] = anomalies

    if key_columns:
        duplicates_on_key = int(df.duplicated(subset=key_columns).sum())
        report["duplicates_on_key_columns"] = duplicates_on_key
        report["duplicates_on_key_columns_pct"] = (
            duplicates_on_key / len(df) * 100 if len(df) > 0 else 0
        )
        for col in key_columns:
            report[f"unique_in_{col}"] = bool(df[col].is_unique) if col in df.columns else None

    return report


def data_quality_and_compare(df1: pd.DataFrame, df2: pd.DataFrame, key_columns=None) -> dict:
    def quality_report(df):
        return {
            "row_count": df.shape[0],
            "column_count": df.shape[1],
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
        }

    report1 = quality_report(df1)
    report2 = quality_report(df2)

    compare = {
        "row_count_match": report1["row_count"] == report2["row_count"],
        "column_count_match": report1["column_count"] == report2["column_count"],
        "common_columns": list(set(df1.columns).intersection(set(df2.columns))),
        "columns_in_df1_not_in_df2": list(set(df1.columns) - set(df2.columns)),
        "columns_in_df2_not_in_df1": list(set(df2.columns) - set(df1.columns)),
    }

    dtype_mismatches = {}
    for col in compare["common_columns"]:
        if report1["data_types"].get(col) != report2["data_types"].get(col):
            dtype_mismatches[col] = (
                report1["data_types"].get(col),
                report2["data_types"].get(col),
            )
    compare["dtype_mismatches"] = dtype_mismatches

    missing_diff = {}
    for col in compare["common_columns"]:
        mv1 = report1["missing_values"].get(col, 0)
        mv2 = report2["missing_values"].get(col, 0)
        if mv1 != mv2:
            missing_diff[col] = (mv1, mv2)
    compare["missing_value_diff"] = missing_diff
    compare["duplicate_rows_diff"] = abs(report1["duplicate_rows"] - report2["duplicate_rows"])

    if key_columns:
        compare["duplicates_on_key_columns_df1"] = int(df1.duplicated(subset=key_columns).sum())
        compare["duplicates_on_key_columns_df2"] = int(df2.duplicated(subset=key_columns).sum())
        compare["duplicates_on_key_columns_pct_df1"] = (
            compare["duplicates_on_key_columns_df1"] / len(df1) * 100 if len(df1) > 0 else 0
        )
        compare["duplicates_on_key_columns_pct_df2"] = (
            compare["duplicates_on_key_columns_df2"] / len(df2) * 100 if len(df2) > 0 else 0
        )
        for col in key_columns:
            compare[f"unique_in_df1_{col}"] = bool(df1[col].is_unique)
            compare[f"unique_in_df2_{col}"] = bool(df2[col].is_unique)
        merged = pd.merge(df1, df2, on=key_columns, how="outer", indicator=True)
        compare["rows_missing_in_df1"] = merged[merged["_merge"] == "right_only"]
        compare["rows_missing_in_df2"] = merged[merged["_merge"] == "left_only"]
    else:
        compare["duplicates_on_key_columns_df1"] = None
        compare["duplicates_on_key_columns_df2"] = None
        compare["duplicates_on_key_columns_pct_df1"] = None
        compare["duplicates_on_key_columns_pct_df2"] = None
        compare["rows_missing_in_df1"] = None
        compare["rows_missing_in_df2"] = None

    anomalies = {}
    for col in compare["common_columns"]:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            neg_df1 = int((df1[col] < 0).sum())
            neg_df2 = int((df2[col] < 0).sum())
            if neg_df1 > 0 or neg_df2 > 0:
                anomalies[col] = {
                    "negative_values_in_df1": neg_df1,
                    "negative_values_in_df2": neg_df2,
                }
    compare["anomalies"] = anomalies

    return {"report_df1": report1, "report_df2": report2, "comparison": compare}


def ai_agent_data_quality(df1: pd.DataFrame, df2: pd.DataFrame = None, key_columns=None) -> dict:
    if df2 is None:
        return {
            "report_df1": single_dataset_quality_report(df1, key_columns),
            "report_df2": None,
            "comparison": None,
        }
    return data_quality_and_compare(df1, df2, key_columns)


def fuzzy_match_columns(cols1, cols2, threshold=60):
    matches = {}
    for col1 in cols1:
        best_match, best_score = None, 0
        for col2 in cols2:
            score = fuzz.ratio(col1.lower(), col2.lower())
            if score > best_score:
                best_score, best_match = score, col2
        if best_score > threshold:
            matches[col1] = best_match
    return matches


def summarize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if null_counts.empty:
        return pd.DataFrame(columns=["column_name", "null_count"])
    summary_df = null_counts.reset_index()
    summary_df.columns = ["column_name", "null_count"]
    return summary_df


def find_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(keep=False)
    return df[dup_mask].copy()
