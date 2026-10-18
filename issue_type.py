import re
import pandas as pd

df = pd.read_parquet("data/code_scored.parquet")

CRASH_STRONG = r"traceback|segfault|segmentation fault|panic:|stack trace|nullpointerexception|unhandled (exception|rejection)|core dumped|crashes|crashed|crashing"
FEATURE_HEAD = r"\[feat\]|\[new rule\]|\[enhancement\]|feature request|^\W*i want|^\W*(# |## )?(add|support|allow|expose|introduce|implement|provide|make|enable|let)\b|i would like|it would be (nice|great|useful)|please add|add support|should (also )?(support|allow|provide|expose)|enhancement|proposal|we want a"
BUG_WORDS = r"\bbug\b|incorrect|wrong|fails?\b|failing|doesn'?t work|does not work|broken|unexpected|regression|not working|silently|error|raises?|ignored|never|instead of|inconsistent|leak|missing"

def classify(text):
    t = text.lower()
    head = t[:300]
    body = t[:1500]
    if re.search(CRASH_STRONG, body):
        return "crash"
    if re.search(FEATURE_HEAD, head, flags=re.M):
        return "feature"
    if re.search(BUG_WORDS, body):
        return "bug"
    return "other"

df["issue_type"] = df["prompt"].apply(classify)
df.to_parquet("data/code_labeled.parquet", index=False)

print(df["issue_type"].value_counts())
print(pd.crosstab(df["issue_type"], df["difficulty"]))