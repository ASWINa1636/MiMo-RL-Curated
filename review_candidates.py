import os
import pandas as pd

CANDIDATES_FILE = "other_review_candidates.csv"
OUTPUT_FILE = "other_candidate_labels.csv"

candidates = pd.read_csv(CANDIDATES_FILE)
required = {"instance_id", "prompt"}

if not required.issubset(candidates.columns):
    raise ValueError(f"Candidate file must contain {required}")

done = (
    pd.read_csv(OUTPUT_FILE)
    if os.path.exists(OUTPUT_FILE)
    else pd.DataFrame(columns=["instance_id", "label"])
)

seen = set(done["instance_id"])
rows = []

print("Keys: b = bug | f = feature | o = other")
print("      s = skip | q = save and quit")
print(f"Candidates: {len(candidates)}")
print(f"Already labeled: {len(seen)}")

for n, (_, r) in enumerate(candidates.iterrows(), start=1):
    instance_id = r["instance_id"]

    if instance_id in seen:
        continue

    print("\n" + "=" * 80)
    print(f"[{n}/{len(candidates)}] {instance_id}")
    print(str(r["prompt"])[:3000])

    while True:
        key = input("\nlabel (b/f/o/s/q): ").strip().lower()

        if key == "q":
            if rows:
                updated = pd.concat(
                    [done, pd.DataFrame(rows)], ignore_index=True
                )
                updated.drop_duplicates(
                    subset="instance_id", keep="last", inplace=True
                )
                updated.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved progress to {OUTPUT_FILE}")
            raise SystemExit

        if key == "s":
            break

        if key in {"b", "f", "o"}:
            label = {"b": "bug", "f": "feature", "o": "other"}[key]
            rows.append({"instance_id": instance_id, "label": label})
            seen.add(instance_id)

            updated = pd.concat(
                [done, pd.DataFrame(rows)], ignore_index=True
            )
            updated.drop_duplicates(
                subset="instance_id", keep="last", inplace=True
            )
            updated.to_csv(OUTPUT_FILE, index=False)
            done = updated
            rows = []
            break

        print("Invalid key. Use b, f, o, s, or q.")

print(f"\nReview complete. Labels saved to {OUTPUT_FILE}")
print(done["label"].value_counts())
