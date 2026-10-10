import os
import pandas as pd

df = pd.read_parquet("data/code_final_v2.parquet")
N = 150
sample = df.sample(N, random_state=7)

OUT = "labels.csv"
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["instance_id", "label"])
seen = set(done["instance_id"])

print("Keys:  b = bug   f = feature   o = other   s = skip   q = save & quit")
print("bug = something that exists works wrongly (incl. crashes)")
print("feature = ask for something new or an extra option")
print("other = refactor, docs, performance, question, unclear\n")

rows = done.to_dict("records")
for n, (_, r) in enumerate(sample.iterrows(), 1):
    if r["instance_id"] in seen:
        continue
    print("=" * 70)
    print(f"[{n}/{N}]  {r['instance_id']}")
    print(r["prompt"][:700].replace("\r", ""))
    while True:
        k = input("\nlabel (b/f/o/s/q): ").strip().lower()
        if k in ("b", "f", "o", "s", "q"):
            break
    if k == "q":
        break
    if k == "s":
        continue
    rows.append({"instance_id": r["instance_id"],
                 "label": {"b": "bug", "f": "feature", "o": "other"}[k]})
    pd.DataFrame(rows).to_csv(OUT, index=False)

print("\nSaved", len(rows), "labels to", OUT)