import os
import json

import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_WINDOW = 1000
RESULTS_ROOT = "MalariaFreeResult"

# Base everything on the AS phenotype frequency given that we assume SS (Heterozygous recessive) does not surive to adulthood
PLOT_LABELS = {
    "initial_pop": "Initial Population Size",
    "as_fraction": "Initial AS Phenotype Proportion",
    "life_expectancy": "Baseline Life Expectancy",
    "as_subtract": "Relative Fatality for Children Under 5 with the AS Phenotype",
    "aa_subtract": "Absolute Malarial Fatality for Children Under 5 with the AA Phenotype",
}

MEAN_Y_LABEL = "Average Steady State Proportion of AS Phenotype"
SD_Y_LABEL = "Standard Deviation of Steady State AS Phenotype Proportion"


def get_xlabel(variable_name):
    return PLOT_LABELS.get(variable_name, variable_name)


def plot_ablation_summary(results_df, variable_name, output_dir):
    x = results_df["value"]
    xlabel = get_xlabel(variable_name)

    # Mean AS proportion
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        results_df["mean_average_as_prop"],
        yerr=results_df["sd_average_as_prop"],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(MEAN_Y_LABEL)
    if variable_name == "life_expectancy":
        ax.set_xticks(x[::2])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variable_name}_mean_as.png"), dpi=150)
    plt.close()

    # SD of AS proportion when stabilized (measures random genetic drift)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        results_df["mean_sd_as_prop"],
        yerr=results_df["sd_sd_as_prop"],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(SD_Y_LABEL)
    if variable_name == "life_expectancy":
        ax.set_xticks(x[::2])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variable_name}_genotypic_stability.png"), dpi=150)
    plt.close()


def regenerate_plots(results_root=RESULTS_ROOT):
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results folder not found: {results_root}")

    for entry in os.listdir(results_root):
        ablation_dir = os.path.join(results_root, entry)
        if not os.path.isdir(ablation_dir):
            continue
        if entry == "baseline":
            continue

        summary_json = os.path.join(ablation_dir, f"{entry}_summary.json")
        if not os.path.exists(summary_json):
            print(f"Skipping {entry}: no summary json found")
            continue

        with open(summary_json, "r") as f:
            summary_rows = json.load(f)

        if not summary_rows:
            print(f"Skipping {entry}: summary json is empty")
            continue

        df = pd.DataFrame(summary_rows)

        required_cols = {
            "value",
            "mean_average_as_prop",
            "sd_average_as_prop",
            "mean_sd_as_prop",
            "sd_sd_as_prop",
        }
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Skipping {entry}: missing columns {sorted(missing)}")
            continue

        plot_ablation_summary(df, entry, ablation_dir)
        print(f"Regenerated plots for {entry}")


if __name__ == "__main__":
    regenerate_plots()