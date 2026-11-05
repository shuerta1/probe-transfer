
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "session2"

MODELS = {
    "Math 7B": "math-7b",
    "Llama 8B": "llama-8b",
    "Qwen 14B": "qwen-14b",
    "Qwen 32B": "qwen-32b",
}


def main():
    plt.figure(figsize=(8, 6))
    results = []

    for model_name, prefix in MODELS.items():
        path = RESULTS_DIR / f"{prefix}_external_scores.csv"
        df = pd.read_csv(path)

        # Only OR-Bench prompts are included in this ROC analysis.
        data = df[
            df["source"].isin(["orbench_hard", "orbench_toxic"])
        ]

        labels = (data["label"] == "unsafe").astype(int)
        scores = data["own_base"]

        auc = roc_auc_score(labels, scores)

        # Compute false-positive and true-positive rates
        # across all possible score thresholds.
        fpr, tpr, thresholds = roc_curve(labels, scores)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

        # Find the threshold that first reaches at least 80% recall.
        candidates = [
            i for i, recall in enumerate(tpr)
            if recall >= 0.80
        ]
        index = candidates[0]

        results.append({
            "Model": model_name,
            "Threshold at 80% Recall": thresholds[index],
            "Recall (%)": tpr[index] * 100,
            "False Positive (%)": fpr[index] * 100,
        })

    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves: Base Probes on OR-Bench")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_dir = ROOT / "my-experiment"
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()

    results_df = pd.DataFrame(results)

    print("\n=== Thresholds for 80% Toxic Recall ===\n")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        output_dir / "threshold_results.csv", index=False
    )

    print("\nSaved roc_curves.png and threshold_results.csv")


if __name__ == "__main__":
    main()
