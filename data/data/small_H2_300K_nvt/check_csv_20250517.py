import pandas as pd

def _temporal_train_valid_split(df, valid_fraction):
    df = df.sort_values(["name", "temp", "t_start"]).reset_index(drop=True)
    valid_indices = []
    for _, group in df.groupby(["name", "temp"], sort=False):
        step = max(1, round(1.0 / valid_fraction))
        val_idx = group.index[::step].tolist()
        valid_indices.extend(val_idx)
    mask = df.index.isin(valid_indices)
    return df[~mask].copy(), df[mask].copy()

df = pd.read_csv("train_1000ps_20260513.csv")
df_train, df_valid = _temporal_train_valid_split(df, valid_fraction=0.2)
print(f"train: {len(df_train)}行, t_start: {df_train['t_start'].min()}〜{df_train['t_start'].max()}")
print(f"val:   {len(df_valid)}行, t_start: {df_valid['t_start'].min()}〜{df_valid['t_start'].max()}")
print(f"val t_start一覧: {sorted(df_valid['t_start'].tolist())}")