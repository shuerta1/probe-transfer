from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "session2"

MODELS = {
    "Math 7B": "math-7b",
    "Llama 8B": "llama-8b",
    "Qwen 14B": "qwen-14b",
    "Qwen 32B": "qwen-32b",
}

SOURCES = ["orbench_hard", "orbench_toxic", "alpaca"]

def main():
    summary_rows = []

    for model_name, prefix in MODELS.items():
        path = RESULTS_DIR / f"{prefix}_external_scores.csv"
        df = pd.read_csv(path)

        plt.figure(figsize=(8, 5))

        for source in SOURCES:
            subset = df[df["source"] == source]["own_base"]

            plt.hist(
                subset,
                bins=40,
                alpha=0.5,
                density=True,
                label=source
            )

            summary_rows.append({
                "Model": model_name,
                "Source": source,
                "Count": len(subset),
                "Mean Score": subset.mean(),
                "Median Score": subset.median(),
                "Std Dev": subset.std(),
                "Min": subset.min(),
                "Max": subset.max(),
                "Pct > 0": (subset > 0).mean() * 100,
            })

        plt.axvline(0, linestyle="--", color="black", label="Threshold = 0")
        plt.title(f"Score Distributions: {model_name}")
        plt.xlabel("Probe Score (own_base)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        out_path = ROOT / "my-experiment" / f"{prefix}_score_distribution.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = ROOT / "my-experiment" / "score_distribution_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Score Distribution Summary ===\n")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_csv}")
    print("Saved one distribution plot per model in my-experiment/")

if __name__ == "__main__":
    main()