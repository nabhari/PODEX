# podex/diversity.py
"""
PODEX — diversity utilities module

Contains:
- data loading helper
- deterministic and Monte-Carlo pooled heterozygosity (Het) functions
- SSD pooling score and Monte-Carlo variant
- exact and Monte-Carlo Shapley value calculators
- extinction-aware Shapley values and expected-diversity-gain ranking (baseline + iterative)
- plotting helper for cumulative iHED curve

This file is safe to import — the runnable demo is guarded under
`if __name__ == "__main__":`.
"""

import sys
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
import os
import matplotlib.cm as cm
import random
import seaborn as sns
from tqdm import tqdm
import shap


#################################################
# READING DATA
#################################################

def load_frequencies(d_path='test_example.csv', pops_col_name='Unnamed: 0'):
    """
    Load a CSV of allele frequencies and set the populations column as index.

    Parameters
    ----------
    d_path : str
        Path to CSV file. Default 'test_example.csv'.
    pops_col_name : str
        Column name that contains population identifiers (will become the index).

    Returns
    -------
    pandas.DataFrame
        DataFrame with populations as the index and loci as columns.
    """
    data0 = pd.read_csv(d_path)
    data = data0.set_index(pops_col_name)
    print("Input data:\n", data.head())
    return data


#################################################
# DIVERSITY MEASURES: POOLING
#################################################

def Het_Pooling_with_pop_size(freqs, pop_size):
    """
    Deterministic pooled heterozygosity (no extinction).

    This computes the pooled heterozygosity H = sum_j 2 * p_bar_j * (1 - p_bar_j),
    where p_bar_j is the size-weighted mean allele (or presence) frequency across populations.

    Parameters
    ----------
    freqs : pandas.DataFrame or array-like
        Shape: (n_pops, n_loci). Rows correspond to populations; columns to loci.
        Values are allele frequencies (or 0/1 presence-absence).
    pop_size : array-like
        Length n_pops. Population sizes (weights).

    Returns
    -------
    float
        Pooled heterozygosity.
    """
    freqs_np = np.asarray(freqs)
    pop_size = np.array(pop_size)
    total_n = pop_size.sum()
    if total_n == 0:
        return 0.0
    pbar = (freqs_np.T @ pop_size) / total_n
    return np.sum(2 * pbar * (1 - pbar))


def Het_Pooling_mc_with_extinction(freqs, pop_size, ext_probs, n_samples=10000, random_state=None):
    """
    Monte Carlo estimate of expected pooled heterozygosity under extinction.

    For each Monte Carlo trial:
      - draw a survival vector from Bernoulli(1 - ext_probs)
      - compute the size-weighted pooled allele frequencies using surviving populations
      - compute heterozygosity for that trial
    The function returns the mean heterozygosity across trials.

    Parameters
    ----------
    freqs : pandas.DataFrame
        (n_pops × n_loci) allele frequencies or presence/absence values.
    pop_size : array-like
        Population sizes (length n_pops).
    ext_probs : array-like
        Extinction probabilities per population (length n_pops).
    n_samples : int
        Number of Monte Carlo trials.
    random_state : int or None
        Seed or NumPy Generator-compatible seed.

    Returns
    -------
    float
        Monte Carlo estimate of expected pooled heterozygosity.
    """
    rng = np.random.default_rng(random_state)
    n_pops, n_loci = freqs.shape
    freqs_np = freqs.to_numpy()
    pop_size = np.array(pop_size)

    het_values = []
    for _ in range(n_samples):
        survival = rng.random(n_pops) > ext_probs
        if survival.sum() == 0:
            het_values.append(0.0)
            continue
        surv_sizes = pop_size * survival
        total_n = surv_sizes.sum()
        pbar = (freqs_np.T @ surv_sizes) / total_n
        het = np.sum(2 * pbar * (1 - pbar))
        het_values.append(het)

    return np.mean(het_values)


def SSD_Pooling_with_pop_size(freqs, pop_size):
    """
    Weighted SSD_Pooling score based on population sizes.

    SSD_Pooling (weighted) counts the number of loci that are polymorphic
    (i.e., have a size-weighted mean frequency 0 < p_bar < 1).

    Parameters
    ----------
    freqs : pandas.DataFrame
        Each column is a locus (split), each row is a population.
    pop_size : array-like
        Population sizes, same length as number of rows in freqs.

    Returns
    -------
    float
        Count of loci with 0 < p_bar < 1 (size-weighted).
    """
    pop_size = np.array(pop_size)
    total_n = pop_size.sum()
    if total_n == 0:
        return 0.0

    score = 0
    for col in freqs.columns:
        weighted_avg = (freqs[col] * pop_size).sum() / total_n
        if 0 < weighted_avg < 1:
            score += 1

    return score


def SSD_Pooling_mc_with_extinction(freqs, pop_size, ext_probs, n_samples=10000, random_state=None):
    """
    Monte Carlo estimate of expected SSD_Pooling under extinction.

    This vectorized implementation simulates survival for n_samples trials
    and computes per-trial size-weighted allele frequencies. For each trial,
    a locus is polymorphic if 0 < p_bar < 1. The function returns the
    mean number of polymorphic loci across simulated trials.

    Parameters
    ----------
    freqs : pandas.DataFrame or array-like
        Shape (n_pops, n_loci).
    pop_size : array-like
        Length n_pops.
    ext_probs : array-like
        Length n_pops.
    n_samples : int
    random_state : int or None

    Returns
    -------
    float
        Expected number of polymorphic loci (mean over trials).
    """
    rng = np.random.default_rng(random_state)
    freqs_np = np.asarray(freqs)
    pop_size = np.asarray(pop_size, dtype=float)
    ext_probs = np.asarray(ext_probs, dtype=float)
    n_pops, n_loci = freqs_np.shape

    # Deterministic shortcut: if no extinctions expected, compute once
    if np.all(ext_probs == 0):
        total_n = pop_size.sum()
        pbar = (freqs_np.T @ pop_size) / total_n
        return float(np.sum((pbar > 0) & (pbar < 1)))

    # Monte Carlo sampling
    survival = rng.random((n_samples, n_pops)) > ext_probs
    weighted_sizes = survival * pop_size  # (n_samples, n_pops)
    total_sizes = weighted_sizes.sum(axis=1)  # (n_samples,)

    # avoid divide-by-zero (all extinct in a trial)
    valid = total_sizes > 0
    total_sizes_safe = np.where(valid, total_sizes, 1)  # safe denominator

    # pbar: (n_samples, n_loci)
    pbar = (weighted_sizes @ freqs_np) / total_sizes_safe[:, None]
    pbar[~valid, :] = np.nan  # mark trials with no survivors as NaN

    poly_counts = np.sum((pbar > 0) & (pbar < 1), axis=1)

    return float(np.nanmean(poly_counts))


#################################################
# SHAPLEY VALUE FUNCTIONS
#################################################

def shapley_values_exact(freqs, pop_size, score_func):
    """
    Compute exact Shapley values by enumerating all permutations.

    WARNING: factorial complexity — feasible only for small n (<= ~6).

    Parameters
    ----------
    freqs : pandas.DataFrame
        Rows = populations, columns = loci.
    pop_size : list-like
        Population sizes.
    score_func : callable
        Function that takes (freqs_subset, pop_size_subset) and returns a float.
        Use the deterministic version Het_Pooling_with_pop_size or SSD_Pooling_with_pop_size.

    Returns
    -------
    dict
        Mapping from population name (freqs.index) to its Shapley value.
    """
    # Use for up to ~5 populations
    n = len(pop_size)
    players = list(range(n))
    shap_values = np.zeros(n)

    # Loop over all permutations
    for perm in itertools.permutations(players):
        coalition = []
        for i in perm:
            if coalition:
                v_without = score_func(freqs.iloc[coalition], [pop_size[j] for j in coalition])
            else:
                v_without = 0.0

            v_with = score_func(freqs.iloc[coalition + [i]], [pop_size[j] for j in coalition + [i]])
            shap_values[i] += v_with - v_without
            coalition.append(i)

    # Average over permutations
    shap_values /= np.math.factorial(n)
    shap_dict = {freqs.index[i]: shap_values[i] for i in range(n)}
    return shap_dict


def shapley_values_mc(freqs, pop_size, score_func, n_permutations=1000, random_state=None):
    """
    Monte Carlo approximation of Shapley values.

    Parameters
    ----------
    freqs : pandas.DataFrame
    pop_size : list-like
    score_func : callable (Use the deterministic version Het_Pooling_with_pop_size or SSD_Pooling_with_pop_size.)
    n_permutations : int
    random_state : int or None

    Returns
    -------
    dict
        Mapping population name -> estimated Shapley value.
    """
    rng = np.random.default_rng(random_state)
    freqs_np = freqs.to_numpy()  # (n_pops × n_loci)
    pop_size = np.asarray(pop_size)
    n = len(pop_size)
    players = np.arange(n)

    shap_values = np.zeros(n)

    for _ in range(n_permutations):
        perm = rng.permutation(players)
        coalition = []
        v_without = 0.0

        for i in perm:
            coalition.append(i)
            v_with = score_func(freqs_np[coalition, :], pop_size[coalition])
            shap_values[i] += v_with - v_without
            v_without = v_with

    shap_values /= n_permutations
    shap_dict = {freqs.index[i]: shap_values[i] for i in range(n)}
    return shap_dict


def shapley_values_mc_with_extinction(
    freqs,
    pop_size,
    score_func,
    ext_probs,
    n_permutations=1000,
    random_state=None,
    return_conditional=False,
    show_progress=True
):
    """
    Monte Carlo Shapley values taking into account independent extinction probabilities.

    The algorithm:
      - for each permutation sample:
          * sample which populations survive (Bernoulli per population)
          * compute marginal contributions for survivors in a random order
      - average contributions to obtain expected Shapley values
    Optionally returns conditional-on-survival Shapley values.

    Parameters
    ----------
    freqs : pandas.DataFrame
    pop_size : array-like
    score_func : callable
    ext_probs : array-like
    n_permutations : int
    random_state : int or None
    return_conditional : bool
    show_progress : bool

    Returns
    -------
    dict or (dict, dict)
        If return_conditional is False, returns expected Shapley dict only.
        If True, returns (expected_shap_dict, conditional_shap_dict).
    """
    rng = np.random.default_rng(random_state)
    freqs_np = freqs.to_numpy()
    pop_size = np.asarray(pop_size, dtype=float)
    ext_probs = np.asarray(ext_probs, dtype=float)

    n = len(pop_size)
    shap_values = np.zeros(n, dtype=float)
    times_survived = np.zeros(n, dtype=int)

    # Pre-generate draws for survival and random orders
    survival_draws = rng.random((n_permutations, n))
    order_draws = [rng.permutation(n) for _ in range(n_permutations)]

    iterator = tqdm(range(n_permutations), disable=not show_progress, desc="Shapley MC Extinction")
    for k in iterator:
        survival = survival_draws[k] > ext_probs
        survivors = np.flatnonzero(survival)
        if survivors.size == 0:
            continue

        times_survived[survivors] += 1
        perm = [i for i in order_draws[k] if i in survivors]
        coalition = []
        v_without = 0.0

        for i in perm:
            if coalition:
                v_without = score_func(freqs_np[coalition, :], pop_size[coalition])
            else:
                v_without = 0.0

            coalition_with_i = coalition + [i]
            v_with = score_func(freqs_np[coalition_with_i, :], pop_size[coalition_with_i])

            shap_values[i] += v_with - v_without
            coalition.append(i)

    expected_shap = shap_values / n_permutations
    ids = list(freqs.index)
    shap_dict = {ids[i]: float(expected_shap[i]) for i in range(n)}

    if not return_conditional:
        print("Shapley values:",shap_dict)
        return shap_dict

    # Conditional-on-survival values
    conditional = np.full(n, np.nan)
    nonzero = times_survived > 0
    conditional[nonzero] = shap_values[nonzero] / times_survived[nonzero]
    conditional_dict = {
        ids[i]: (float(conditional[i]) if not np.isnan(conditional[i]) else np.nan)
        for i in range(n)
    }
    print("Shap values conditional:",conditional_dict)

    return shap_dict, conditional_dict


#################################################
# EXPECTED DIVERSITY GAIN RANKING (baseline + iterative)
#################################################

def expected_diversity_gain_ranking(freqs,
                                    pop_size,
                                    ext_probs,
                                    mode="het_mc",
                                    n_samples=5000,
                                    random_state=None,
                                    only_baseline=False):
    """
    Compute baseline W(i) gains and the iterative (iHED-style) ranking.

    The function returns:
      - baseline_df: ranking of populations by their baseline gain W(i),
        computed by forcing each population to survive (setting its extinction
        prob to 0) while leaving others unchanged.
      - iterative_df: stepwise ranking produced by iteratively "protecting"
        the chosen population (setting its extinction probability to 0) and
        recomputing gains for the remainder.

    Parameters
    ----------
    freqs : pandas.DataFrame
    pop_size : array-like
    ext_probs : array-like
    mode : {"het_det", "het_mc", "ssd_det", "ssd_mc"}
    n_samples : int
    random_state : int or None
    only_baseline : bool
        If True, the function returns only baseline_df and skips the iterative
        protection procedure (early return).

    Returns
    -------
    baseline_df, iterative_df
        If only_baseline is True, only baseline_df is returned.
    """
    rng = np.random.default_rng(random_state)
    n_pops = len(pop_size)
    ext_probs = np.array(ext_probs, float)

    # Scoring dispatcher: selects the appropriate scoring routine
    def compute_score(freqs_sub, pop_size_sub, ext_probs_sub):
        if mode == "het_det":
            return Het_Pooling_with_pop_size(freqs_sub, pop_size_sub)

        elif mode == "het_mc":
            return Het_Pooling_mc_with_extinction(
                freqs_sub, pop_size_sub, ext_probs_sub,
                n_samples=n_samples, random_state=rng
            )

        elif mode == "ssd_det":
            return SSD_Pooling_with_pop_size(freqs_sub, pop_size_sub)

        elif mode == "ssd_mc":
            return SSD_Pooling_mc_with_extiction(
                freqs_sub, pop_size_sub, ext_probs_sub,
                n_samples=n_samples, random_state=rng
            )

        else:
            raise ValueError(f"Unknown mode: {mode}. Mode must be het_det, het_mc, ssd_det, or ssd_mc.")

    # Helper that uses deterministic computation when ext_probs_sub == 0 vector
    def deterministic_if_no_extinction(freqs_sub, pop_size_sub, ext_probs_sub):
        if np.all(ext_probs_sub == 0):
            if mode in ["het_mc", "het_det"]:
                return Het_Pooling_with_pop_size(freqs_sub, pop_size_sub)
            if mode in ["ssd_mc", "ssd_det"]:
                return SSD_Pooling_with_pop_size(freqs_sub, pop_size_sub)
        return compute_score(freqs_sub, pop_size_sub, ext_probs_sub)

    # --- PART 1: Baseline W(i) ranking ---
    baseline_H = deterministic_if_no_extinction(freqs, pop_size, ext_probs)
    baseline_gains = []
    print("Baseline gain and ranking with no protection are calculated as:")

    for i in range(n_pops):
        e_hat = ext_probs.copy()
        e_hat[i] = 0.0   # guarantee survival of pop i
        H_hat = deterministic_if_no_extinction(freqs, pop_size, e_hat)
        baseline_gains.append(H_hat - baseline_H)

    baseline_df = pd.DataFrame({
        "Rank": range(1, n_pops + 1),
        "Population": freqs.index,
        "Gain": baseline_gains
    }).sort_values("Gain", ascending=False)

    print(baseline_df)

    # === EARLY RETURN WHEN USER WANTS ONLY BASELINE ===
    if only_baseline:
        return baseline_df

    # --- PART 2: Iterative iHED-style ranking ---
    saved = np.zeros(n_pops, dtype=bool)
    ranking, gains = [], []

    current_ext = ext_probs.copy()

    print("Iterative (iHED-style) gain and ranking are calculated as:")
    for step in range(n_pops):
        H_e = deterministic_if_no_extinction(freqs, pop_size, current_ext)

        W = []
        for i in range(n_pops):
            if saved[i]:
                W.append(-np.inf)
                continue

            e_hat = current_ext.copy()
            e_hat[i] = 0.0
            H_hat = deterministic_if_no_extinction(freqs, pop_size, e_hat)
            W.append(H_hat - H_e)

        best_idx = np.argmax(W)
        gains.append(W[best_idx])
        ranking.append(freqs.index[best_idx])

        saved[best_idx] = True
        current_ext[best_idx] = 0.0

    iterative_df = pd.DataFrame({
        "Rank": range(1, n_pops + 1),
        "Population": ranking,
        "Gain": gains
    })
    print(iterative_df)
    return baseline_df, iterative_df


#################################################
# PLOTTING HELPERS
#################################################

def plot_iterative_cumulative_gain(gain_df):
    """
    Plot cumulative expected gain as populations are saved one-by-one.

    The function adds a "CumulativeGain" column to the supplied DataFrame
    (which is modified in-place) and creates a simple line plot.

    Parameters
    ----------
    gain_df : pandas.DataFrame
        DataFrame created by expected_diversity_gain_ranking for iterative gains.
    """
    cumulative_gain = np.cumsum(gain_df["Gain"])
    gain_df["CumulativeGain"] = cumulative_gain

    plt.figure(figsize=(6, 4))
    plt.plot(gain_df["Rank"], cumulative_gain, marker='o')
    plt.xlabel("Number of populations saved")
    plt.ylabel("Cumulative increase in expected heterozygosity")
    plt.title("Cumulative expected diversity gain (iHED curve)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#################################################
# DEMONSTRATION / SCRIPT (runs only when executed directly)
#################################################
if __name__ == "__main__":
    # Measure runtime for demo only — safe to import the module without running.
    start_time = time.time()

    # Load example data (adjust d_path / pops_col_name as needed)
    d = load_frequencies(pops_col_name='pops')

    d_np = d.to_numpy()

    # Example population sizes and extinction probabilities (small demo)
    pop_size_rand = [17, 27, 43]
    ext_prob = [0.9, 0.8, 0.1]

    # Compute baseline & iterative rankings (default mode het_mc)
    baseline_df, iterative_df = expected_diversity_gain_ranking(
        freqs=d,
        pop_size=pop_size_rand,
        ext_probs=ext_prob,
        mode="het_mc",
        n_samples=1000,
        random_state=None
    )
    shap_dict = shapley_values_mc_with_extinction(
        freqs=d,
        pop_size=pop_size_rand,
        score_func=Het_Pooling_with_pop_size,
        ext_probs=ext_prob,
        n_permutations=1000,
        random_state=None,
        return_conditional=False,
        show_progress=True
    )
    # Plot cumulative gain curve for the iterative ranking
    plot_iterative_cumulative_gain(gain_df=iterative_df)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")
