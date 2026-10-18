import os
import pandas as pd

df = pd.read_parquet("data/code_final_v2.parquet").set_index("instance_id")
ai = pd.read_csv("labels_ai.csv")

# audit set = every "unsure" task + 20 random others (seeded, so it is the same each run)
rest = ai[~ai["unsure"]].sample(20, random_state=1)
audit = pd.concat([ai[ai["unsure"]].assign(group="unsure"), rest.assign(group="random")])
audit = audit.sample(frac=1, random_state=2)             # mix the order

OUT = "audit_answers.csv"
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["instance_id", "group", "ai_label", "human_label"])
seen = set(done["instance_id"])
rows = done.to_dict("records")
M = {"b": "bug", "f": "feature", "o": "other"}

print("For each task you see the AI label.")
print("Press ENTER if you agree, or b / f / o to change it, or q to save and quit.\n")

for n, (_, r) in enumerate(audit.iterrows(), 1):
    if r["instance_id"] in seen:
        continue
    print("=" * 70)
    print(f"[{n}/{len(audit)}]  {r['instance_id']}")
    print(df.loc[r["instance_id"], "prompt"][:700].replace("\r", ""))
    print(f"\nAI says: {r['label'].upper()}")
    while True:
        k = input("ENTER = agree, or b/f/o, q = quit: ").strip().lower()
        if k in ("", "b", "f", "o", "q"):
            break
    if k == "q":
        break
    human = r["label"] if k == "" else M[k]
    rows.append({"instance_id": r["instance_id"], "group": r["group"],
                 "ai_label": r["label"], "human_label": human})
    pd.DataFrame(rows).to_csv(OUT, index=False)

a = pd.DataFrame(rows)
print(f"\nAudited {len(a)} of {len(audit)} tasks")
if len(a):
    for g in ["random", "unsure"]:
        sub = a[a["group"] == g]
        if len(sub):
            print(f"  {g:7s}: AI and you agree on {(sub.ai_label == sub.human_label).sum()} of {len(sub)}"
                  f" ({(sub.ai_label == sub.human_label).mean():.0%})")

# final training labels: AI labels, overwritten by your corrections
final = ai[["instance_id", "label"]].set_index("instance_id")
final.loc[a["instance_id"].values, "label"] = a["human_label"].values
final.reset_index().to_csv("labels.csv", index=False)
print("Wrote labels.csv with", len(final), "labels (your corrections applied)")
