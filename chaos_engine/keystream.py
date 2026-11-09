"""AES-CTR keystream generator for diffusion masks and permutation sources.

This module replaces the previous logistic-map-based pseudo-random source for
the chaos-stage diffusion. AES-CTR with a fresh nonce per generator instance is
indistinguishable from random under standard assumptions (PRP-PRF switching
lemma + AES security), giving the diffusion stage a defensible cryptographic
keystream rather than a chaotic-but-breakable map.

The legacy `LogisticMap` is preserved for ablation comparison (set
``ChaosGenerator(backend="logistic", ...)``).
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


_AES_KEY_BYTES = 32  # AES-256
_AES_BLOCK_BYTES = 16
_DEFAULT_BUFFER_BLOCKS = 4096  # 64 KiB per refill


def _derive_key(seed_material: bytes) -> bytes:
    """Derive a 256-bit AES key from arbitrary seed material via SHA-256.

    SHA-256 is a one-way function so the seed cannot be recovered from the key,
    and any byte string yields a uniformly distributed 32-byte key.
    """
    return hashlib.sha256(seed_material).digest()


class AESCTRKeystream:
    """A streaming source of cryptographically pseudo-random bytes.

    The keystream is produced by running AES-256 in counter (CTR) mode over the
    all-zero plaintext stream, which yields the raw keystream. The block
    counter is implicit in the ``Cipher`` object; reset() rebuilds the cipher
    so the same seed reproduces the same stream.

    This generator is deterministic given the seed material, which is what we
    need for a reproducible image-encryption pipeline. For unique per-session
    diffusion, callers must supply a per-session seed (e.g., session_id).
    """

    def __init__(
        self,
        seed_material: bytes,
        nonce: Optional[bytes] = None,
        buffer_blocks: int = _DEFAULT_BUFFER_BLOCKS,
    ) -> None:
        if not isinstance(seed_material, (bytes, bytearray)):
            raise TypeError("seed_material must be bytes")
        if len(seed_material) == 0:
            raise ValueError("seed_material must not be empty")

        self._seed_material = bytes(seed_material)
        # A deterministic per-instance nonce (derived from the seed) keeps the
        # generator reproducible. Callers wanting domain separation should add
        # context to seed_material directly.
        self._nonce = nonce if nonce is not None else hashlib.sha256(
            b"chaos_engine.aes_ctr.nonce|" + self._seed_material
        ).digest()[:_AES_BLOCK_BYTES]
        if len(self._nonce) != _AES_BLOCK_BYTES:
            raise ValueError(f"nonce must be {_AES_BLOCK_BYTES} bytes")

        self._key = _derive_key(self._seed_material)
        self._buffer_size = max(1, int(buffer_blocks)) * _AES_BLOCK_BYTES
        self._buffer = b""
        self._buffer_offset = 0
        self._encryptor = None
        self.reset()

    def reset(self) -> None:
        """Re-initialize the keystream so it replays from the start."""
        cipher = Cipher(
            algorithms.AES(self._key),
            modes.CTR(self._nonce),
            backend=default_backend(),
        )
        self._encryptor = cipher.encryptor()
        self._buffer = b""
        self._buffer_offset = 0

    def _refill(self) -> None:
        zeros = b"\x00" * self._buffer_size
        self._buffer = self._encryptor.update(zeros)
        self._buffer_offset = 0

    def read(self, length: int) -> bytes:
        """Read ``length`` keystream bytes."""
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return b""

        chunks = []
        remaining = length
        while remaining > 0:
            if self._buffer_offset >= len(self._buffer):
                self._refill()
            available = len(self._buffer) - self._buffer_offset
            take = min(available, remaining)
            chunks.append(self._buffer[self._buffer_offset : self._buffer_offset + take])
            self._buffer_offset += take
            remaining -= take
        return b"".join(chunks)

    def read_uint8(self, count: int) -> np.ndarray:
        """Read ``count`` uniformly distributed bytes as a uint8 ndarray."""
        return np.frombuffer(self.read(count), dtype=np.uint8).copy()

    def read_uniform_floats(self, count: int) -> np.ndarray:
        """Read ``count`` uniform floats in [0, 1) using 53-bit precision.

        We use 8 keystream bytes per float and mask to 53 bits, matching the
        precision of double-precision floats. This avoids the bias that would
        result from naively dividing a uint64 keystream by 2**64.
        """
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        raw = self.read(count * 8)
        u64 = np.frombuffer(raw, dtype=np.uint64).copy()
        mantissa = u64 >> np.uint64(11)
        return mantissa.astype(np.float64) * (1.0 / (1 << 53))


def seed_material_from_legacy(seed: float, r: float) -> bytes:
    """Convert a legacy ``(seed, r)`` pair to deterministic seed bytes.

    Existing configs and tests pass scalar ``seed`` and ``r``. This helper
    converts them losslessly so the new keystream backend can replace the
    logistic backend without changing config schema.
    """
    payload = f"chaos.legacy_pair|seed={seed!r}|r={r!r}".encode("utf-8")
    return hashlib.sha256(payload).digest()
