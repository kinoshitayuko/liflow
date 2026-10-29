# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd


# -------- user parameters --------
# input_csv = Path("flow_trace_finetuned_300K_seed42.csv")
input_csv = Path("20261014_flow_trace_normal_300K_seed42.csv")
# output_csv = Path("flow_trace_finetuned_300K_seed43_selected_atoms_31.csv")
output_csv = Path("20261014_flow_trace_normal_300K_seed42.csv_selected_atoms_31.csv")

# target_atom_index = [30, 31]  # 0始まり。atom_index列の値を指定
target_atom_index = [31]  # 0始まり。atom_index列の値を指定

# 絞り込まない場合は None
target_outer_step = None  # 例: 1 → 外側ステップ1のみ
target_phase = None       # 例: "prop", "corr", "after_com"

# 大きなCSVを一度にメモリへ読み込まないための設定
chunksize = 100_000
# ---------------------------------


def main():
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("input_csvとoutput_csvには異なるファイルを指定してください。")

    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()

    required_columns = {"atom_index"}
    if target_outer_step is not None:
        required_columns.add("outer_step")
    if target_phase is not None:
        required_columns.add("phase")

    missing_columns = required_columns - set(columns)
    if missing_columns:
        raise ValueError(
            f"必要な列がありません: {sorted(missing_columns)}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 出力を初期化。該当行が0件でもヘッダーを保存する。
    pd.DataFrame(columns=columns).to_csv(
        output_csv, index=False, encoding="utf-8"
    )

    total_rows = 0
    extracted_rows = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        total_rows += len(chunk)

        # mask = chunk["atom_index"] == target_atom_index
        # 変更後
        mask = chunk["atom_index"].isin(target_atom_index)

        if target_outer_step is not None:
            mask &= chunk["outer_step"] == target_outer_step

        if target_phase is not None:
            mask &= chunk["phase"] == target_phase

        selected = chunk.loc[mask]

        if selected.empty:
            continue

        selected.to_csv(
            output_csv,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )
        extracted_rows += len(selected)

    print(f"Input: {input_csv}")
    print(f"Target atom_index: {target_atom_index}")
    print(f"Target outer_step: {target_outer_step}")
    print(f"Target phase: {target_phase}")
    print(f"Read rows: {total_rows:,}")
    print(f"Extracted rows: {extracted_rows:,}")
    print(f"Saved: {output_csv}")

    if extracted_rows == 0:
        print(
            "Warning: 該当する行がありません。"
            "原子インデックスや絞り込み条件を確認してください。"
        )


if __name__ == "__main__":
    main()