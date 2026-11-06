
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "my-experiment"

# Load the results from our previous experiment.
df = pd.read_csv(EXP_DIR / "category_analysis.csv")

results = []

for _, row in df.iterrows():
    category = row["category"]
    total = int(row["Total_Prompts"])
    errors = int(row["Shared_False_Positives"])

    # Wilson 95% confidence interval for a binomial proportion.
    lower, upper = proportion_confint(
        count=errors,
        nobs=total,
        alpha=0.05,
        method="wilson"
    )

    results.append({
        "Category": category,
        "Total": total,
        "Shared Errors": errors,
        "Error Rate (%)": errors / total * 100,
        "95% CI Lower": lower * 100,
        "95% CI Upper": upper * 100,
    })

summary = pd.DataFrame(results)

print("\n=== Shared False-Positive Rates with 95% CIs ===\n")
print(summary.round(2).to_string(index=False))

output = EXP_DIR / "category_confidence.csv"
summary.to_csv(output, index=False)

print(f"\nSaved to: {output}")
