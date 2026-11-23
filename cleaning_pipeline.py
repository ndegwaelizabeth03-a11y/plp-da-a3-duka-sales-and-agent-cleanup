import numpy as np, pandas as pd


def clean_agent_data(txns, agents):
    """Clean raw transactions and merge the agent register.

    Returns (clean_df, log); log is a DataFrame of rule, rows affected.
    """
    log = []
    note = lambda rule, n: log.append({"rule": rule, "rows": int(n)})
    df = txns.copy()
    n = len(df)
    df = df.drop_duplicates(subset="txn_id")
    note("duplicate txn_id removed", n - len(df))

    t = df["txn_type"].str.strip().str.capitalize()
    note("txn_type standardised", (t != df["txn_type"]).sum())
    df["txn_type"] = t

    amt = pd.to_numeric(df["amount"], errors="coerce")
    df["amount_imputed"] = amt.isna()
    note("amount missing/invalid (N/A, NaN)", amt.isna().sum())
    q1, q3 = amt.quantile([.25, .75])
    hi = q3 + 1.5 * (q3 - q1)
    df["is_outlier"] = amt > hi
    note(f"amount > IQR fence {hi:.0f} flagged, kept", df["is_outlier"].sum())
    # median per type keeps valid rows; flagged so analysts can exclude
    df["amount"] = amt.fillna(amt.groupby(df["txn_type"]).transform("median"))

    raw = df["txn_date"]
    df["txn_date"] = pd.to_datetime(raw.str.replace("/", "-"), format="%Y-%m-%d")
    note("txn_date with '/' fixed", raw.str.contains("/").sum())

    df = df.merge(agents[["agent_id", "county"]], on="agent_id", how="left")
    unk = df["county"].isna()
    df["county"] = df["county"].fillna("Unknown")
    note("unknown agents (county=Unknown, kept)", unk.sum())
    return df, pd.DataFrame(log)
