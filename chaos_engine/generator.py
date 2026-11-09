"""Pseudo-random source for chaos-stage permutation and diffusion.

The default backend is ``aes_ctr`` (AES-256-CTR keystream), which provides a
cryptographic PRF for diffusion masks and permutation indices. The legacy
``logistic`` backend is preserved so that V1 baseline experiments and the
logistic-vs-AEAD ablation remain reproducible.

Both backends share the same external interface so :class:`ChaosPermutation`
and the rest of the pipeline are oblivious to the change.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .keystream import AESCTRKeystream, seed_material_from_legacy
from .logistic import LogisticMap


_DEFAULT_BACKEND = "aes_ctr"


class ChaosGenerator:
    """Unified interface for generating pseudo-random sequences and masks.

    Parameters
    ----------
    seed:
        For ``backend="logistic"`` this is the floating-point initial state in
        ``(0, 1)``. For ``backend="aes_ctr"`` this can be any of:
        ``bytes``/``bytearray`` seed material, an ``int`` (encoded as 8 LE
        bytes), or a ``float`` (kept for config compatibility -- combined with
        ``r`` via :func:`seed_material_from_legacy`).
    r:
        Logistic map control parameter. Ignored by the AES-CTR backend except
        when deriving deterministic seed material from a legacy ``(seed, r)``
        config.
    backend:
        ``"aes_ctr"`` (default) or ``"logistic"``.
    nonce:
        Optional 16-byte nonce for the AES-CTR backend. Defaults to a
        deterministic hash of the seed material so the backend is
        reproducible.
    """

    def __init__(
        self,
        seed: Union[bytes, bytearray, int, float] = 0.123,
        r: float = 3.99,
        backend: str = _DEFAULT_BACKEND,
        nonce: Optional[bytes] = None,
    ) -> None:
        backend = (backend or _DEFAULT_BACKEND).strip().lower()
        if backend not in {"aes_ctr", "logistic"}:
            raise ValueError(f"Unsupported chaos backend: {backend!r}")

        self.seed = seed
        self.r = r
        self.backend = backend
        self._nonce = nonce
        self.engine = self._build_engine()

    def _build_engine(self):
        if self.backend == "logistic":
            return LogisticMap(float(self.seed), float(self.r))
        seed_material = self._coerce_seed_material()
        return AESCTRKeystream(seed_material, nonce=self._nonce)

    def _coerce_seed_material(self) -> bytes:
        seed = self.seed
        if isinstance(seed, (bytes, bytearray)):
            return bytes(seed)
        if isinstance(seed, int):
            return seed.to_bytes(8, byteorder="little", signed=False)
        if isinstance(seed, float):
            return seed_material_from_legacy(seed, self.r)
        if isinstance(seed, str):
            return seed.encode("utf-8")
        raise TypeError(f"Unsupported seed type for AES-CTR backend: {type(seed)!r}")

    def reset(self) -> None:
        """Reset internal state so subsequent generation replays the same stream."""
        if self.backend == "logistic":
            self.engine.reset()
        else:
            self.engine.reset()

    def generate_sequence(self, length: int) -> np.ndarray:
        """Return ``length`` floats in ``[0, 1)`` for permutation index sorting."""
        if length <= 0:
            return np.empty(0, dtype=np.float64)
        if self.backend == "logistic":
            return self.engine.generate(length)
        return self.engine.read_uniform_floats(length)

    def generate_bytes(self, length: int) -> bytes:
        """Return ``length`` pseudo-random bytes."""
        if length <= 0:
            return b""
        if self.backend == "logistic":
            seq = self.engine.generate(length)
            return (seq * 255).astype(np.uint8).tobytes()
        return self.engine.read(length)

    def generate_mask(self, shape: tuple) -> np.ndarray:
        """Return a uint8 mask with the requested shape."""
        size = int(np.prod(shape))
        if size <= 0:
            return np.empty(shape, dtype=np.uint8)
        if self.backend == "logistic":
            seq = self.engine.generate(size)
            return (seq * 255).astype(np.uint8).reshape(shape)
        return self.engine.read_uint8(size).reshape(shape)
