import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_AGES = 27
MAX_STEPS = 2000
SUMMARY_WINDOW = 1000
CSV_PATH = "FiveYearSurvivalProbability.csv"
RNG_SEED = 0


def load_survivals(csv_path, column, as_subtract, aa_subtract):
    df = pd.read_csv(csv_path)
    base = df[column].to_numpy(dtype=float)

    if len(base) != N_AGES:
        raise ValueError(f"Expected {N_AGES} survival values, got {len(base)}")

    survival_AS = base.copy()
    survival_AA = base.copy()

    # subtract from 0-5 survival only
    survival_AS[0] = max(0.0, survival_AS[0] - as_subtract)
    survival_AA[0] = max(0.0, survival_AA[0] - aa_subtract)

    return survival_AA, survival_AS


def initial_population(survival, total_pop, as_fraction):
    pop = np.zeros((2, N_AGES), dtype=int)

    as_total = int(round(total_pop * as_fraction))
    aa_total = total_pop - as_total

    weights = np.ones(N_AGES, dtype=float)
    for k in range(1, N_AGES):
        weights[k] = weights[k - 1] * survival[k - 1]

    weights /= weights.sum()

    aa_counts = np.floor(aa_total * weights).astype(int)
    aa_counts[0] += aa_total - aa_counts.sum()

    as_counts = np.floor(as_total * weights).astype(int)
    as_counts[0] += as_total - as_counts.sum()

    pop[0, :] = aa_counts
    pop[1, :] = as_counts

    return pop


def fertility_weights():
    w = np.zeros(N_AGES, dtype=float)
    w[3:11] = 1.0  # ages 15-50 inclusive
    return w


def make_run_name(params):
    return (
        f"pop{params['initial_pop']}"
        f"_asfrac{params['as_fraction']}"
        f"_lifeexp{params['life_expectancy']}"
        f"_assub{params['as_subtract']}"
        f"_aasub{params['aa_subtract']}"
    )


def run_simulation(params, output_dir, csv_path=CSV_PATH, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    os.makedirs(output_dir, exist_ok=True)

    run_name = make_run_name(params)
    output_csv = os.path.join(output_dir, f"{run_name}.csv")
    output_png = os.path.join(output_dir, f"{run_name}.png")

    initial_pop = params["initial_pop"]
    as_fraction = params["as_fraction"]
    life_expectancy = params["life_expectancy"]
    as_subtract = params["as_subtract"]
    aa_subtract = params["aa_subtract"]

    survival_AA, survival_AS = load_survivals(
        csv_path,
        life_expectancy,
        as_subtract,
        aa_subtract,
    )

    fert_w = fertility_weights()
    carrying_capacity = initial_pop

    pop = initial_population(survival_AS, initial_pop, as_fraction)

    rows = []
    as_prop_history = []

    for step_idx in range(1, MAX_STEPS + 1):
        new_pop = np.zeros_like(pop)

        # deterministic survival + aging, rounded to nearest int
        new_pop[0, 1:] = np.round(pop[0, :-1] * survival_AA[:-1]).astype(int)
        new_pop[1, 1:] = np.round(pop[1, :-1] * survival_AS[:-1]).astype(int)

        # reproductive population
        aa_repro = float(np.sum(pop[0, :] * fert_w))
        as_repro = float(np.sum(pop[1, :] * fert_w))
        total_repro = aa_repro + as_repro

        if total_repro > 0:
            p_AA = aa_repro / total_repro
            p_AS = as_repro / total_repro
        else:
            p_AA = 0.0
            p_AS = 0.0

        # expected births with smooth fertility-only density dependence
        current_pop = float(pop.sum())
        density_factor = carrying_capacity / (carrying_capacity + current_pop)

        births_expected = total_repro * density_factor
        n_children = max(0, int(round(births_expected)))

        # newborn genotype probabilities from Punnett square
        prob_AA = p_AA * p_AA + 2.0 * p_AA * p_AS * 0.5 + p_AS * p_AS * 0.25
        prob_AS = 2.0 * p_AA * p_AS * 0.5 + p_AS * p_AS * 0.5
        prob_SS = p_AS * p_AS * 0.25

        probs = np.array([prob_AA, prob_AS, prob_SS], dtype=float)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.array([1.0, 0.0, 0.0])

        newborn_AA, newborn_AS, newborn_SS = rng.multinomial(n_children, probs)

        # SS not tracked in living population matrix
        new_pop[0, 0] = newborn_AA
        new_pop[1, 0] = newborn_AS

        # realized newborn proportions
        total_new = newborn_AA + newborn_AS + newborn_SS
        if total_new > 0:
            prop_newborn_AA = newborn_AA / total_new
            prop_newborn_AS = newborn_AS / total_new
            prop_newborn_SS = newborn_SS / total_new
        else:
            prop_newborn_AA = 0.0
            prop_newborn_AS = 0.0
            prop_newborn_SS = 0.0

        # realized reproducing proportions
        if total_repro > 0:
            prop_repro_AA = aa_repro / total_repro
            prop_repro_AS = as_repro / total_repro
        else:
            prop_repro_AA = 0.0
            prop_repro_AS = 0.0

        total_pop_now = int(new_pop.sum())
        if total_pop_now > 0:
            prop_AS_population = new_pop[1].sum() / total_pop_now
        else:
            prop_AS_population = 0.0

        rows.append(
            {
                "step": step_idx,
                "population_size": total_pop_now,
                "births": n_children,
                "newborn_AA": int(newborn_AA),
                "newborn_AS": int(newborn_AS),
                "newborn_SS": int(newborn_SS),
                "prop_newborn_AA": prop_newborn_AA,
                "prop_newborn_AS": prop_newborn_AS,
                "prop_newborn_SS": prop_newborn_SS,
                "prop_repro_AA": prop_repro_AA,
                "prop_repro_AS": prop_repro_AS,
                "prop_AS_population": prop_AS_population,
            }
        )

        as_prop_history.append(prop_AS_population)
        pop = new_pop

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    tail = np.array(as_prop_history[-SUMMARY_WINDOW:])
    average_as_prop = float(np.mean(tail))
    sd_as_prop = float(np.std(tail))
    final_population = int(pop.sum())

    param_text = (
        f"initial_pop = {initial_pop}\n"
        f"as_fraction = {as_fraction}\n"
        f"life_expectancy = {life_expectancy}\n"
        f"as_subtract = {as_subtract}\n"
        f"aa_subtract = {aa_subtract}\n"
        f"K = {carrying_capacity}\n"
        f"density = K / (K + N)\n"
        f"max_steps = {MAX_STEPS}\n"
        f"summary_window = {SUMMARY_WINDOW}\n"
        f"avg_last_window = {average_as_prop:.6f}\n"
        f"sd_last_window = {sd_as_prop:.6f}\n"
        f"final_pop = {final_population}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["prop_AS_population"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Proportion AS")
    ax.set_title("AS Proportion Over Time")

    fig.subplots_adjust(right=0.72)
    fig.text(
        0.76,
        0.5,
        param_text,
        va="center",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"{run_name}: avg_last_{SUMMARY_WINDOW}={average_as_prop}, "
        f"sd_last_{SUMMARY_WINDOW}={sd_as_prop}, final_pop={final_population}"
    )

    return {
        "run_name": run_name,
        "average_as_prop": average_as_prop,
        "sd_as_prop": sd_as_prop,
        "final_population": final_population,
        "csv_path": output_csv,
        "png_path": output_png,
    }


def plot_ablation_summary(results_df, variable_name, output_dir):
    x = results_df["value"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, results_df["average_as_prop"], marker="o")
    ax.set_xlabel(variable_name)
    ax.set_ylabel(f"Mean AS Proportion (last {SUMMARY_WINDOW})")
    ax.set_title(f"{variable_name}: Mean AS Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variable_name}_mean_as.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, results_df["sd_as_prop"], marker="o")
    ax.set_xlabel(variable_name)
    ax.set_ylabel(f"SD of AS Proportion (last {SUMMARY_WINDOW})")
    ax.set_title(f"{variable_name}: Genotypic Stability")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variable_name}_genotypic_stability.png"), dpi=150)
    plt.close()


def run_ablation(variable_name, values, baseline_params, results_root):
    ablation_dir = os.path.join(results_root, variable_name)
    os.makedirs(ablation_dir, exist_ok=True)

    summary_rows = []

    for value in values:
        params = baseline_params.copy()
        params[variable_name] = value
        if variable_name == "aa_subtract":
            params["as_subtract"] = value / 10

        result = run_simulation(params, ablation_dir)

        summary_rows.append(
            {
                "value": value,
                "average_as_prop": result["average_as_prop"],
                "sd_as_prop": result["sd_as_prop"],
                "final_population": result["final_population"],
                "run_name": result["run_name"],
            }
        )

    summary_json = os.path.join(ablation_dir, f"{variable_name}_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_ablation_summary(pd.DataFrame(summary_rows), variable_name, ablation_dir)


def main():
    os.makedirs("Results", exist_ok=True)

    baseline_params = {
        "initial_pop": 1000,
        "as_fraction": 0.05,
        "life_expectancy": "25.0",
        "as_subtract": 0.01,
        "aa_subtract": 0.1,
    }

    # Fill these with whatever values you want to test.
    initial_pop_ablation = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
    as_fraction_ablation = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    life_expectancy_ablation = ["20.0", "25.0", "30.0", "35.0", "40.0", "45.0", "50.0", "55.0", "60.0"]
    as_subtract_ablation = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    aa_subtract_ablation = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Baseline run
    baseline_dir = os.path.join("Results", "baseline")
    baseline_result = run_simulation(baseline_params, baseline_dir)

    baseline_summary = [
        {
            "average_as_prop": baseline_result["average_as_prop"],
            "sd_as_prop": baseline_result["sd_as_prop"],
            "final_population": baseline_result["final_population"],
            "run_name": baseline_result["run_name"],
        }
    ]
    with open(os.path.join(baseline_dir, "baseline_summary.json"), "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # One-parameter-at-a-time ablations
    run_ablation("initial_pop", initial_pop_ablation, baseline_params, "Results")
    run_ablation("as_fraction", as_fraction_ablation, baseline_params, "Results")
    run_ablation("life_expectancy", life_expectancy_ablation, baseline_params, "Results")
    run_ablation("as_subtract", as_subtract_ablation, baseline_params, "Results")
    run_ablation("aa_subtract", aa_subtract_ablation, baseline_params, "Results")


if __name__ == "__main__":
    main()