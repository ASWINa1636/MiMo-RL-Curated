
import pandas as pd
from pathlib import Path

labels_path = Path("labels_all.csv")
corrections_path = Path("label_corrections.csv")
output_path = Path("labels_all_corrected.csv")

labels = pd.read_csv(labels_path)
corrections = pd.read_csv(corrections_path)

# Validate columns
assert {"instance_id", "label", "source"}.issubset(labels.columns)
assert {"instance_id", "old_label", "new_label"}.issubset(
    corrections.columns
)

# Validate IDs
assert labels["instance_id"].is_unique, "Duplicate IDs in labels_all.csv"
assert corrections["instance_id"].is_unique, "Duplicate correction IDs"

label_ids = set(labels["instance_id"])
correction_ids = set(corrections["instance_id"])

missing = correction_ids - label_ids
if missing:
    raise ValueError(f"Correction IDs not found: {sorted(missing)}")

# Verify the old labels before applying corrections
current = labels.set_index("instance_id")["label"]

for row in corrections.itertuples(index=False):
    actual = current.loc[row.instance_id]
    if actual != row.old_label:
        raise ValueError(
            f"{row.instance_id}: expected {row.old_label!r}, "
            f"found {actual!r}"
        )

# Apply corrections to a copy
corrected = labels.copy()
new_labels = corrections.set_index("instance_id")["new_label"]
mask = corrected["instance_id"].isin(new_labels.index)

corrected.loc[mask, "label"] = (
    corrected.loc[mask, "instance_id"].map(new_labels)
)

# Verify and save without overwriting the original
assert int(mask.sum()) == len(corrections)
corrected.to_csv(output_path, index=False)

print("Corrections applied successfully!")
print(f"Original labels: {len(labels)}")
print(f"Corrections applied: {int(mask.sum())}")
print(f"Output file: {output_path}")
print("\nUpdated label distribution:")
print(corrected["label"].value_counts(dropna=False))
print("\nUpdated source distribution:")
print(corrected["source"].value_counts(dropna=False))
