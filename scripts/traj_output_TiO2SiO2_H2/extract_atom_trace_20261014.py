# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd


# -------- user parameters --------
input_csv = Path(
    #"20261014_flow_trace_normal_300K_seed42_2.csv"
    "20261014_flow_trace_finetune_300K_seed42.csv"
)

output_csv = Path(
    "20261014_flow_trace_finetune_300K_seed42"
    "_prior_vs_final_atoms31.csv"
)

# 複数原子を指定可能。0始まり。
target_atom_indices = [31]
# target_atom_indices = [30, 31]

# Noneなら全外側ステップを取り出す。
target_outer_step = None

chunksize = 100_000
# ---------------------------------


def main():
    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    if input_csv.resolve() == output_csv.resolve():
        raise ValueError(
            "input_csvとoutput_csvには異なるファイルを指定してください。"
        )

    selected_atoms = sorted(set(target_atom_indices))
    if not selected_atoms:
        raise ValueError(
            "target_atom_indicesに原子インデックスを指定してください。"
        )

    # 初期ノイズと最終変位を対応付けるキー
    key_columns = [
        "outer_step",
        "atom_index",
        "retry_index",
    ]

    prior_columns = [
        "flow_prior_dx",
        "flow_prior_dy",
        "flow_prior_dz",
    ]

    final_columns = [
        "net_dx",
        "net_dy",
        "net_dz",
    ]

    required_columns = (
        key_columns
        + ["phase", "flow_index"]
        + prior_columns
        + final_columns
    )

    columns = pd.read_csv(
        input_csv, nrows=0
    ).columns.tolist()

    missing_columns = (
        set(required_columns) - set(columns)
    )

    if missing_columns:
        raise ValueError(
            f"必要な列がありません: {sorted(missing_columns)}"
        )

    prior_parts = []
    final_parts = []
    total_rows = 0

    for chunk in pd.read_csv(
        input_csv,
        usecols=required_columns,
        chunksize=chunksize,
    ):
        total_rows += len(chunk)

        # 対象原子を選択
        mask = chunk["atom_index"].isin(selected_atoms)

        if target_outer_step is not None:
            mask &= (
                chunk["outer_step"] == target_outer_step
            )

        selected = chunk.loc[mask]

        if selected.empty:
            continue

        # Propagatorの開始時：実際に渡したprior
        prior_mask = (
            selected["phase"].eq("prop")
            & selected["flow_index"].eq(0)
        )

        prior_rows = selected.loc[
            prior_mask,
            key_columns + prior_columns,
        ]

        if not prior_rows.empty:
            prior_parts.append(prior_rows.copy())

        # 採用された最終変位：Corrector・重心補正の後
        final_mask = selected["phase"].eq("after_com")

        final_rows = selected.loc[
            final_mask,
            key_columns + final_columns,
        ]

        if not final_rows.empty:
            final_parts.append(final_rows.copy())

    if not prior_parts:
        raise ValueError(
            "対象条件のPropagator開始行がありません。"
            "原子番号やouter_stepを確認してください。"
        )

    if not final_parts:
        raise ValueError(
            "対象条件のafter_com行がありません。"
            "記録内容を確認してください。"
        )

    prior_df = pd.concat(
        prior_parts, ignore_index=True
    ).rename(
        columns={
            "flow_prior_dx": "prior_dx",
            "flow_prior_dy": "prior_dy",
            "flow_prior_dz": "prior_dz",
        }
    )

    final_df = pd.concat(
        final_parts, ignore_index=True
    ).rename(
        columns={
            "net_dx": "final_dx",
            "net_dy": "final_dy",
            "net_dz": "final_dz",
        }
    )

    # 各ステップ・原子・試行に、開始行と終了行が1つずつあるか確認
    for label, frame in [
        ("prior", prior_df),
        ("final", final_df),
    ]:
        if frame.duplicated(key_columns).any():
            raise ValueError(
                f"{label}側に同じステップ・原子・試行の重複があります。"
            )

    # 同じステップ・原子・試行の初期ノイズと最終変位を結合
    result = prior_df.merge(
        final_df,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = result["_merge"].ne("both")

    if unmatched.any():
        examples = result.loc[
            unmatched,
            key_columns + ["_merge"],
        ].head(10)

        raise ValueError(
            "初期ノイズと最終変位の対応が取れない行があります。\n"
            f"{examples.to_string(index=False)}"
        )

    result = result.drop(columns="_merge")

    output_columns = [
        "outer_step",
        "atom_index",
        "retry_index",
        "prior_dx",
        "prior_dy",
        "prior_dz",
        "final_dx",
        "final_dy",
        "final_dz",
    ]

    result = result[output_columns].sort_values(
        key_columns
    ).reset_index(drop=True)

    value_columns = [
        "prior_dx", "prior_dy", "prior_dz",
        "final_dx", "final_dy", "final_dz",
    ]

    if result[value_columns].isna().any().any():
        raise ValueError(
            "抽出した初期ノイズまたは最終変位に欠損値があります。"
        )

    output_csv.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    print(f"Input: {input_csv}")
    print(f"Target atoms: {selected_atoms}")
    print(f"Target outer_step: {target_outer_step}")
    print(f"Read rows: {total_rows:,}")
    print(f"Saved rows: {len(result):,}")

    print("\nRows per atom:")
    for atom_index in selected_atoms:
        count = int(
            result["atom_index"].eq(atom_index).sum()
        )
        print(f"  atom_index={atom_index}: {count:,}")

        if count == 0:
            print("    Warning: この原子の該当行はありません。")

    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()