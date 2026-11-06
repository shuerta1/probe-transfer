
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "my-experiment"

# Load reconstructed prompt text and saved classification results.
prompts = pd.read_csv(EXPERIMENT_DIR / "external_prompts.csv")
errors = pd.read_csv(EXPERIMENT_DIR / "error_overlap_results.csv")

# Confirm that source and label values agree for every selected row.
indices = errors["original_row"].astype(int)

assert indices.is_unique, "Duplicate row indices found."
assert indices.between(0, len(prompts) - 1).all()

matched = prompts.iloc[indices].reset_index(drop=True)
errors = errors.reset_index(drop=True)

assert matched["source"].equals(errors["source"])
assert matched["label"].equals(errors["label"])

# Combine prompt text with probe scores.
combined = pd.concat(
    [
        matched[["prompt", "source", "category", "label"]],
        errors.drop(columns=["source", "label"]),
    ],
    axis=1,
)

# Select benign prompts flagged by all four models.
shared = combined[
    combined["Number of Models Flagging"] == 4
].copy()

# Save the shared false positives.
output_path = EXPERIMENT_DIR / "shared_false_positives.csv"
shared.to_csv(output_path, index=False)

print("\n=== Shared False Positives ===")
print(f"Total benign-hard prompts: {len(combined)}")
print(f"Flagged by all four models: {len(shared)}")

print("\n=== Categories ===")
print(shared["category"].value_counts().to_string())

print("\n=== First 5 Shared False Positives ===")

for number, prompt in enumerate(shared["prompt"].head(5), start=1):
    print(f"\n{number}. {prompt}")

print(f"\nResults saved to: {output_path}")
