
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "my-experiment"

prompts = pd.read_csv(EXP_DIR / "external_prompts.csv")
errors = pd.read_csv(EXP_DIR / "error_overlap_results.csv")

# Keep only benign-but-alarming prompts.
hard = prompts[
    prompts["source"] == "orbench_hard"
].copy()

# Use original row positions to associate saved scores
# with the reconstructed prompt categories.
errors = errors.set_index("original_row")

hard["Number of Models Flagging"] = errors.loc[
    hard.index, "Number of Models Flagging"
].to_numpy()

# Count total prompts and shared errors for every category.
summary = hard.groupby("category").agg(
    Total_Prompts=("category", "size"),
    Shared_False_Positives=(
        "Number of Models Flagging",
        lambda x: (x == 4).sum()
    ),
    Three_Or_More=(
        "Number of Models Flagging",
        lambda x: (x >= 3).sum()
    ),
)

summary["Shared_Error_Rate (%)"] = (
    summary["Shared_False_Positives"]
    / summary["Total_Prompts"] * 100
)

summary["Three_Or_More_Rate (%)"] = (
    summary["Three_Or_More"]
    / summary["Total_Prompts"] * 100
)

summary = summary.sort_values(
    "Shared_Error_Rate (%)",
    ascending=False
)

print("\n=== Category-Level Error Analysis ===\n")
print(summary.round(2).to_string())

output = EXP_DIR / "category_analysis.csv"
summary.to_csv(output)

print(f"\nResults saved to: {output}")
