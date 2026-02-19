import pandas as pd

# df = pd.read_csv("data2/labels.csv")  # founder_uuid, success
# seed = 42
#
# val_success = df[df.success == 1].sample(n=162, random_state=seed)
# val_fail    = df[df.success == 0].sample(n=1638, random_state=seed)
#
# val_df = pd.concat([val_success, val_fail], ignore_index=True)\
#           .sample(frac=1, random_state=seed).reset_index(drop=True)
#
# train_df = df.merge(val_df[["founder_uuid"]], on="founder_uuid", how="left", indicator=True)
# train_df = train_df[train_df["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)
#
# val_df.to_csv("labels_val.csv", index=False)
# train_df.to_csv("labels_train.csv", index=False)

# ------------------------------------------------------------------------------------------------------------

train_path = "labels_train.csv"   # 如果你放在 data/ 里就改成 "data/labels_train.csv"

df = pd.read_csv(train_path)

train_success = df[df["success"] == 1].reset_index(drop=True)
train_fail    = df[df["success"] == 0].reset_index(drop=True)

train_success.to_csv("labels_train_success.csv", index=False)
train_fail.to_csv("labels_train_fail.csv", index=False)

print("train total:", len(df))
print("train success:", len(train_success))
print("train fail:", len(train_fail))