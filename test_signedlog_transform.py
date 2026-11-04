# -*- coding: utf-8 -*-
from pathlib import Path

import torch

import liflow.utils.transforms as transforms_module
from liflow.utils.transforms import SignedLogTransform


# このテストファイルが置かれているプロジェクトルート。
PROJECT_ROOT = Path(__file__).resolve().parent

JSON_PATH = (
    PROJECT_ROOT
    # / "liflow"
    / "data"
    / "data"
    / "small_H2_300K_nvt"
    / "positions_300K_all_species_lambda0p1_lag50f_parameters.json"
)

# まずはCPUで確認する。
DEVICE = torch.device("cpu")


def run_test(dtype):
    transform = SignedLogTransform.from_json(
        JSON_PATH
    ).to(DEVICE)

    # H, He, O, Si, Tiの内部元素ID。
    elements = torch.tensor(
        [0, 1, 2, 3, 4],
        dtype=torch.long,
        device=DEVICE,
    )

    displacement = torch.tensor(
        [
            [0.1, -0.2, 0.0],    # H
            [3.0, -2.0, 1.0],    # He
            [0.1, 0.0, -0.1],    # O
            [0.0, 0.1, -0.1],    # Si
            [-0.1, 0.2, 0.0],    # Ti
        ],
        dtype=dtype,
        device=DEVICE,
        requires_grad=True,
    )

    # ① 全元素について、変換→逆変換を確認。
    transformed = transform.transform(
        displacement,
        elements,
    )

    restored = transform.inverse(
        transformed,
        elements,
    )

    tolerance = (
        1e-5 if dtype == torch.float32 else 1e-12
    )

    torch.testing.assert_close(
        restored,
        displacement,
        rtol=tolerance,
        atol=tolerance,
    )

    max_error = (
        (restored - displacement)
        .abs()
        .max()
        .detach()
        .item()
    )

    # ② ゼロがゼロになることを確認。
    zeros = torch.zeros_like(
        displacement,
        requires_grad=False,
    )

    torch.testing.assert_close(
        transform.transform(zeros, elements),
        zeros,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        transform.inverse(zeros, elements),
        zeros,
        rtol=0,
        atol=0,
    )

    # ③ 符号が保たれることを確認。
    assert torch.equal(
        torch.sign(transformed),
        torch.sign(displacement),
    )

    # ④ 往復した関数の勾配が1になることを確認。
    # ゼロの成分も含めて確認する。
    restored.sum().backward()

    torch.testing.assert_close(
        displacement.grad,
        torch.ones_like(displacement),
        rtol=tolerance,
        atol=tolerance,
    )

    max_gradient_error = (
        (displacement.grad - 1)
        .abs()
        .max()
        .item()
    )

    # ⑤ 係数がstate_dictに保存されることを確認。
    state = transform.state_dict()

    assert "a_per_A" in state
    assert "valid_ids" in state

    # 変換係数は学習対象のパラメータではない。
    assert len(list(transform.parameters())) == 0

    # ⑥ 未登録IDがエラーになることを確認。
    invalid_id = transform.valid_ids.numel()

    try:
        transform.transform(
            torch.zeros(
                (1, 3),
                dtype=dtype,
                device=DEVICE,
            ),
            torch.tensor(
                [invalid_id],
                dtype=torch.long,
                device=DEVICE,
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unregistered element ID was not rejected."
        )

    print(f"\n[{dtype}]")
    print("Transformed displacement:")
    print(transformed.detach().cpu())

    print(
        "Maximum inverse error:",
        f"{max_error:.6e} Å",
    )
    print(
        "Maximum round-trip gradient error:",
        f"{max_gradient_error:.6e}",
    )
    print(
        "State-dict keys:",
        list(state.keys()),
    )
    print("PASS")


def main():
    print("Transform implementation:")
    print(transforms_module.__file__)

    print("\nParameter JSON:")
    print(JSON_PATH)

    print("\nDevice:", DEVICE)

    if not JSON_PATH.is_file():
        raise FileNotFoundError(
            "JSONが見つかりません。"
            "JSON_PATHのファイル名・場所を確認してください。\n"
            f"{JSON_PATH}"
        )

    run_test(torch.float32)
    run_test(torch.float64)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()