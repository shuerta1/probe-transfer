
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "session2"
OUTPUT_DIR = ROOT / "my-experiment"

MODELS = {
    "Math 7B": "math-7b",
    "Llama 8B": "llama-8b",
    "Qwen 14B": "qwen-14b",
    "Qwen 32B": "qwen-32b",
}


def main():
    data = {}

    for name, prefix in MODELS.items():
        path = RESULTS_DIR / f"{prefix}_external_scores.csv"
        data[name] = pd.read_csv(path).reset_index(drop=True)

    # Verify that rows appear consistently ordered.
    reference = data["Math 7B"]

    for name, df in data.items():
        assert len(df) == len(reference), (
            f"Row count differs for {name}"
        )
        assert df["source"].equals(reference["source"]), (
            f"Source order differs for {name}"
        )
        assert df["label"].equals(reference["label"]), (
            f"Label order differs for {name}"
        )

    # Focus on benign-but-alarming OR-Bench prompts.
    hard_mask = reference["source"] == "orbench_hard"

    errors = pd.DataFrame(index=reference.index[hard_mask])

    for name, df in data.items():
        # A positive score means the probe incorrectly
        # flags a benign-hard prompt as unsafe.
        errors[name] = df.loc[hard_mask, "own_base"] > 0

    errors["Number of Models Flagging"] = errors.sum(axis=1)

    print("\n=== False-Positive Overlap ===\n")

    counts = errors["Number of Models Flagging"].value_counts()
    counts = counts.sort_index()

    for number in range(5):
        count = counts.get(number, 0)
        percentage = count / len(errors) * 100

        print(
            f"Flagged by {number} models: "
            f"{count} prompts ({percentage:.1f}%)"
        )

    print("\n=== Pairwise Error Overlap ===\n")

    names = list(MODELS)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = errors[names[i]]
            b = errors[names[j]]

            intersection = (a & b).sum()
            union = (a | b).sum()

            jaccard = intersection / union if union else 0

            print(
                f"{names[i]} vs {names[j]}: "
                f"Jaccard = {jaccard:.3f}"
            )

    # Save classifications and underlying scores.
    output = pd.DataFrame(index=errors.index)
    output["source"] = reference.loc[hard_mask, "source"]
    output["label"] = reference.loc[hard_mask, "label"]

    for name, df in data.items():
        output[f"{name} Score"] = df.loc[
            hard_mask, "own_base"
        ]
        output[f"{name} Flagged"] = errors[name]

    output["Number of Models Flagging"] = (
        errors["Number of Models Flagging"]
    )

    output_path = OUTPUT_DIR / "error_overlap_results.csv"
    output.to_csv(output_path, index_label="original_row")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
