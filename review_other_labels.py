
import pandas as pd
from pathlib import Path

INPUT_FILE = "review_other_labels.csv"
OUTPUT_FILE = "label_corrections.csv"

df = pd.read_csv(INPUT_FILE).fillna("")
df = df[df["true_label"] == "other"].reset_index(drop=True)

if df.empty:
    print("No 'other' examples found in the review file.")
    raise SystemExit(0)

output_path = Path(OUTPUT_FILE)

if output_path.exists():
    corrections = pd.read_csv(output_path).fillna("")
    existing_ids = set(corrections["instance_id"].astype(str))
else:
    corrections = pd.DataFrame(columns=[
        "instance_id", "old_label", "new_label", "reason"
    ])
    existing_ids = set()

labels = {
    "b": "bug",
    "f": "feature",
    "o": "other",
}

print("\nMiMo-RL Curated — Other Label Review")
print("b = bug | f = feature | o = other")
print("s = skip | q = save and quit\n")

for i, row in df.iterrows():
    instance_id = str(row["instance_id"])

    if instance_id in existing_ids:
        continue

    print("\n" + "=" * 80)
    print(f"Example {i + 1}/{len(df)}: {instance_id}")
    print(f"Current label: {row['true_label']}")
    print(f"Previous model prediction: {row['predicted_label']}")
    print("-" * 80)
    print(str(row["prompt"]))

    while True:
        choice = input(
            "\nCorrect label? [b/f/o/s/q]: "
        ).strip().lower()

        if choice == "q":
            corrections.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved {len(corrections)} corrections to {OUTPUT_FILE}")
            return_code = 0
            raise SystemExit(return_code)

        if choice == "s":
            break

        if choice in labels:
            new_label = labels[choice]
            old_label = str(row["true_label"])

            if new_label == old_label:
                print("Confirmed: no correction needed.")
                break

            reason = input(
                "Why is the original label incorrect? "
            ).strip()

            if not reason:
                print("Please enter a short reason.")
                continue

            corrections = corrections[
                corrections["instance_id"].astype(str) != instance_id
            ]

            corrections.loc[len(corrections)] = [
                instance_id,
                old_label,
                new_label,
                reason,
            ]

            corrections.to_csv(OUTPUT_FILE, index=False)
            print(f"Correction saved: {old_label} -> {new_label}")
            break

        print("Invalid choice. Enter b, f, o, s, or q.")

corrections.to_csv(OUTPUT_FILE, index=False)
print(f"\nReview complete. Corrections saved: {len(corrections)}")
print(f"Output: {OUTPUT_FILE}")
