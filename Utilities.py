import pandas as pd
from pathlib import Path
from Constants import UNDeathProbability

def tsv_to_qx_csv(filepath: str, output_name: str | None = None) -> str:
    """
    Read a UN Model Life Table TSV and write a CSV of qx1 by age for all E0
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, sep="\t")

    mask = (df["Family"] == "General") & (df["Sex"].isin(["Male", "Female"]))
    df = df.loc[mask, ["E0", "Sex", "age", "qx1"]]

    # Combine sexes by averaging
    combined = df.groupby(["E0", "age"], as_index=False)["qx1"].mean()

    pivoted = combined.pivot(index="age", columns="E0", values="qx1")
    pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)
    pivoted.columns = [f"{c}" for c in pivoted.columns]

    out_path = Path.cwd() / (output_name or f"{filepath.stem}_qx_by_E0.csv")
    pivoted.to_csv(out_path)
    return str(out_path)

# print(tsv_to_qx_csv("LifeExpectancy.tsv", UNDeathProbability))

def qx_to_nyear(filepath: str = UNDeathProbability,
                output_name: str = "FiveYearSurvivalProbability.csv",
                interval: int = 5) -> str:
    """
    Collapse single-year qx1 into n-year survival probabilities per E0 column.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    age_col = df.columns[0]
    df = df.sort_values(age_col).reset_index(drop=True)

    e0_cols = [c for c in df.columns if c != age_col]

    df["_bin"] = (df[age_col] // interval) * interval

    bin_sizes = df.groupby("_bin").size()
    complete_bins = bin_sizes[bin_sizes == interval].index
    df = df[df["_bin"].isin(complete_bins)]

    survival = (1 - df[e0_cols]).assign(_bin=df["_bin"].values)
    n_p_x = survival.groupby("_bin").prod()
    n_p_x.index.name = "age"
    n_p_x.loc[130] = 0.0

    out_path = Path.cwd() / (output_name)
    n_p_x.to_csv(out_path)
    return str(out_path)

print(qx_to_nyear())