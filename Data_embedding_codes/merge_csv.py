import pandas as pd

df1 = pd.read_csv("vcbench_final_private.csv")
df2 = pd.read_csv("vcbench_final_public.csv")

# 直接追加行
merged = pd.concat([df1, df2], ignore_index=True)

# 如果同一个 founder_uuid 可能重复，想去重就加这一行（保留第一条）
# merged = merged.drop_duplicates(subset=["founder_uuid"], keep="first")

merged.to_csv("merged.csv", index=False, encoding="utf-8-sig")
