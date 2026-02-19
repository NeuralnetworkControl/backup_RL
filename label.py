import pandas as pd

RAW_DATA = "merged.csv"
OUT_LABELS = "data2/labels.csv"

# 读取原始数据
df = pd.read_csv(RAW_DATA)

# 基本检查
assert "founder_uuid" in df.columns, "missing founder_uuid"
assert "success" in df.columns, "missing success label"

# 只保留 label 相关列
labels_df = df[["founder_uuid", "success"]].copy()

# 强制转换为 int (0/1)
labels_df["success"] = labels_df["success"].astype(int)

# 去重检查
if labels_df["founder_uuid"].duplicated().any():
    raise ValueError("Duplicate founder_uuid found in labels")

# 保存
labels_df.to_csv(OUT_LABELS, index=False)

print(f"Saved labels to {OUT_LABELS}")
print(labels_df.head())
