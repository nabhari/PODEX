# main.py
"""
Main execution script for the PODEX project.
(Use this to run analyses; comment-out sections you don't need.)
"""
import time
# import other modules
from podex.diversity import (
    load_frequencies,
    Het_Pooling_with_pop_size,
    shapley_values_mc,
    shapley_values_exact,
    expected_diversity_gain_ranking,
    plot_iterative_cumulative_gain
)

DATA_PATH = "example_data.csv"
POPS_COL_NAME = "pops"
POP_SIZES = [17, 27, 43]
EXTINCTION_PROBS = [0.9, 0.8, 0.1]
SEED = 42

def main():
    start = time.time()
    print("\n=== PODEX: Diversity Analysis ===\n")
    freqs = load_frequencies(d_path=DATA_PATH, pops_col_name=POPS_COL_NAME)
    print("\nComputing expected diversity gain ranking...\n")
    baseline_df, iterative_df = expected_diversity_gain_ranking(
        freqs=freqs,
        pop_size=POP_SIZES,
        ext_probs=EXTINCTION_PROBS,
        mode="het_mc",
        n_samples=5000,
        random_state=SEED
    )
    print("\n--- Baseline Ranking ---")
    print(baseline_df)
    print("\n--- Iterative Ranking ---")
    print(iterative_df)
    print("\nPlotting cumulative gain curve...")
    plot_iterative_cumulative_gain(iterative_df)
    elapsed = time.time() - start
    print(f"\nFinished. Total runtime: {elapsed:.2f} seconds.\n")

if __name__ == "__main__":
    main()
