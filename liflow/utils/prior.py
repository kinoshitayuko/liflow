# import abc

# import numpy as np
# from ase import units


# class Prior(abc.ABC):
#     @abc.abstractmethod
#     def sample(self, shape: tuple) -> np.ndarray:
#         pass


# class NormalPrior:
#     def __init__(self, scale: float = 1.0, seed: int = 42):
#         self.scale = scale
#         self.rng = np.random.default_rng(seed)

#     def sample(self, shape: tuple) -> np.ndarray:
#         return self.rng.normal(scale=self.scale, size=shape)


# class UniformScaleNormalPrior:
#     def __init__(self, scale: float = 1.0, seed: int = 42):
#         self.scale = scale
#         self.rng = np.random.default_rng(seed)

#     def sample(self, shape: tuple) -> np.ndarray:
#         scale = self.rng.uniform(0, self.scale, size=shape[0])
#         return self.rng.normal(scale=scale[:, None], size=shape)


# class MaxwellBoltzmannPrior:
#     def __init__(self, scale: float = 1.0, seed: int = 42):
#         self.scale = scale
#         self.rng = np.random.default_rng(seed)

#     def sample(
#         self, temperature: float, masses: np.ndarray, shape: tuple
#     ) -> np.ndarray:
#         scale = self.scale * np.sqrt(units.kB * temperature / masses)
#         return self.rng.normal(scale=scale[:, None], size=shape)


# class AdaptiveMaxwellBoltzmannPrior:
#     def __init__(
#         self,
#         scale: list[list[float]] = [[1.0, 10.0], [1.0, 5.0]],  # [Li_scale, prior_scale]
#         seed: int = 42,
#     ):
#         self.scale = scale
#         self.rng = np.random.default_rng(seed)

#     def sample(
#         self,
#         temperature: float,
#         atomic_numbers: np.ndarray,
#         masses: np.ndarray,
#         scale_Li_index: int,
#         scale_frame_index: int,
#         shape: tuple,
#     ) -> np.ndarray:
#         scale_Li = self.scale[0][scale_Li_index]
#         scale_frame = self.scale[1][scale_frame_index]
#         #prefactor = np.where(atomic_numbers == 3, scale_Li, scale_frame)
#         prefactor = np.where(atomic_numbers == 2, scale_Li, scale_frame)
#         scale = prefactor * np.sqrt(units.kB * temperature / masses)
#         return self.rng.normal(scale=scale[:, None], size=shape)


# def get_prior(class_name: str, **kwargs):
#     return {
#         "NormalPrior": NormalPrior,
#         "UniformScaleNormalPrior": UniformScaleNormalPrior,
#         "MaxwellBoltzmannPrior": MaxwellBoltzmannPrior,
#         "AdaptiveMaxwellBoltzmannPrior": AdaptiveMaxwellBoltzmannPrior,
#     }[class_name](**kwargs)



import abc

import numpy as np
from ase import units


class Prior(abc.ABC):
    @abc.abstractmethod
    def sample(self, shape: tuple) -> np.ndarray:
        pass


class NormalPrior:
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple) -> np.ndarray:
        return self.rng.normal(scale=self.scale, size=shape)


class UniformScaleNormalPrior:
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple) -> np.ndarray:
        scale = self.rng.uniform(0, self.scale, size=shape[0])
        return self.rng.normal(scale=scale[:, None], size=shape)


class MaxwellBoltzmannPrior:
    def __init__(self, scale: float = 1.0, seed: int = 42):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        temperature: float,
        masses: np.ndarray,
        shape: tuple,
    ) -> np.ndarray:
        scale = self.scale * np.sqrt(units.kB * temperature / masses)
        return self.rng.normal(scale=scale[:, None], size=shape)


class AdaptiveMaxwellBoltzmannPrior:
    def __init__(
        self,
        scale: list[list[float]] = [[1.0, 10.0], [1.0, 5.0]],
        seed: int = 42,
    ):
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        temperature: float,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        scale_Li_index: int,
        scale_frame_index: int,
        shape: tuple,
    ) -> np.ndarray:
        scale_Li = self.scale[0][scale_Li_index]
        scale_frame = self.scale[1][scale_frame_index]

        # 現在の実装：原子番号2を拡散原子として扱う
        prefactor = np.where(
            atomic_numbers == 2,
            scale_Li,
            scale_frame,
        )

        scale = prefactor * np.sqrt(
            units.kB * temperature / masses
        )

        return self.rng.normal(
            scale=scale[:, None],
            size=shape,
        )


# ============================================================
# 追加：拡散原子だけを2成分混合ガウスにするprior
# ============================================================
class GaussianMixtureAdaptivePrior(AdaptiveMaxwellBoltzmannPrior):
    """
    対象の拡散原子：
        平均0・等方的な2成分混合ガウス

    フレーム原子：
        従来のAdaptiveMaxwellBoltzmannPriorと同じ分布

    mixture_sigmasは、各軸成分の標準偏差をÅで直接指定する。
    対象原子の混合ガウスにはsqrt(kB*T/m)を追加で掛けない。
    """

    def __init__(
        self,
        scale: list[list[float]] = [[1.0, 10.0], [1.0, 5.0]],
        seed: int = 42,
        mixture_weights: tuple[float, float] = (0.7023, 0.2977),
        mixture_sigmas: tuple[float, float] = (0.9306, 3.0208),
        target_atomic_number: int = 2,
    ):
        # 既存のscale設定と乱数生成器をそのまま利用
        super().__init__(scale=scale, seed=seed)

        weights = np.asarray(mixture_weights, dtype=float)
        sigmas = np.asarray(mixture_sigmas, dtype=float)

        if weights.shape != (2,):
            raise ValueError(
                "mixture_weightsには2つの値を指定してください。"
            )

        if sigmas.shape != (2,):
            raise ValueError(
                "mixture_sigmasには2つの値を指定してください。"
            )

        if not np.isfinite(weights).all():
            raise ValueError(
                "mixture_weightsには有限の値を指定してください。"
            )

        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(
                "mixture_weightsは非負で、合計が正である必要があります。"
            )

        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError(
                "mixture_sigmasには有限の正の値を指定してください。"
            )

        # 丸め誤差などに対応するため、重みの合計を1にする
        self.mixture_weights = weights / weights.sum()
        self.mixture_sigmas = sigmas
        self.target_atomic_number = int(target_atomic_number)

    def sample(
        self,
        temperature: float,
        atomic_numbers: np.ndarray,
        masses: np.ndarray,
        scale_Li_index: int,
        scale_frame_index: int,
        shape: tuple,
    ) -> np.ndarray:
        atomic_numbers = np.asarray(atomic_numbers)
        masses = np.asarray(masses, dtype=float)
        shape = tuple(shape)

        if len(shape) != 2 or shape[1] != 3:
            raise ValueError(
                "shapeは(n_atoms, 3)である必要があります。"
            )

        if atomic_numbers.shape != (shape[0],):
            raise ValueError(
                "atomic_numbersの長さとshape[0]が一致していません。"
            )

        if masses.shape != (shape[0],):
            raise ValueError(
                "massesの長さとshape[0]が一致していません。"
            )

        # 1. まず、既存の方法で全原子のpriorを作る。
        #    フレーム原子の分布・スケールは従来どおり。
        sampled_prior = super().sample(
            temperature=temperature,
            atomic_numbers=atomic_numbers,
            masses=masses,
            scale_Li_index=scale_Li_index,
            scale_frame_index=scale_frame_index,
            shape=shape,
        )

        # 2. 混合ガウスへ置き換える対象原子を選ぶ。
        target_mask = (
            atomic_numbers == self.target_atomic_number
        )
        n_target = int(np.count_nonzero(target_mask))

        if n_target == 0:
            return sampled_prior

        # 3. 対象原子ごとに、狭い／広い成分を1つ選ぶ。
        #    同じ原子のxyzでは同じ成分のsigmaを使う。
        component_indices = self.rng.choice(
            2,
            size=n_target,
            p=self.mixture_weights,
        )

        sigma_per_atom = self.mixture_sigmas[
            component_indices
        ]

        # 4. 選んだsigmaでxyzノイズを生成する。
        mixture_noise = self.rng.normal(
            loc=0.0,
            scale=sigma_per_atom[:, None],
            size=(n_target, 3),
        )

        # 5. 対象原子のノイズだけ置き換える。
        #    フレーム原子の要素は変更しない。
        sampled_prior[target_mask] = mixture_noise

        return sampled_prior


def get_prior(class_name: str, **kwargs):
    return {
        "NormalPrior": NormalPrior,
        "UniformScaleNormalPrior": UniformScaleNormalPrior,
        "MaxwellBoltzmannPrior": MaxwellBoltzmannPrior,
        "AdaptiveMaxwellBoltzmannPrior": AdaptiveMaxwellBoltzmannPrior,

        # 追加
        "GaussianMixtureAdaptivePrior": GaussianMixtureAdaptivePrior,
    }[class_name](**kwargs)