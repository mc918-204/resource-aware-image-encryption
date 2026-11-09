"""Per-image, domain-separated protection session (v2 profile).

This mirrors the interface of the frozen :class:`~src.crypto_core.session.CryptoSession`
so the pipeline can switch profiles without restructuring, while fixing the
two defects the V1 session carries (plan items F1 and M10):

===================  ==========================================  ==================================================
Property             V1 (:class:`CryptoSession`)                 v2 (:class:`DomainSeparatedSession`)
===================  ==========================================  ==================================================
Keys per image       2 (``session_id||_HIGH``, ``..._MED``)      one per ``(purpose, block_index, level)``
HKDF salt            ``None``                                    fresh 16-byte per-image ``image_nonce``
Block index binding  AEAD associated data only                   key derivation **and** associated data
AEAD nonce           ``os.urandom(12)``, stored per block         derived deterministically, recomputable
Diffusion mask key   ``SHA-256(LE8(chaos.seed))``, public         ``HKDF-Expand(prk, MASK|idx|level)``, secret
Permutation source   public ``chaos.seed``                       ``HKDF-Expand(prk, PERM)``, secret and fresh
===================  ==========================================  ==================================================

The ``low`` level still applies keystream diffusion **without** AEAD, exactly as
in V1. That is deliberate: it keeps the v1/v2 comparison apples-to-apples so the
CPA experiment isolates the effect of key derivation rather than conflating it
with the addition of authentication. ``low`` therefore still provides **no
integrity and no authenticated confidentiality** under either profile -- what v2
fixes is that its keystream is no longer derivable from public constants and no
longer reused across images.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.chaos_engine.generator import ChaosGenerator
from src.chaos_engine.permutation import ChaosPermutation
from src.crypto_core.aead import AEADPrimitive
from src.crypto_core.domain_kdf import (
    IMAGE_NONCE_BYTES,
    PURPOSE_AEAD,
    PURPOSE_MASK,
    derive_block_key,
    derive_block_nonce,
    derive_image_prk,
    derive_permutation_seed_material,
    generate_image_nonce,
)

__all__ = ["DomainSeparatedSession", "LEVEL_ALGORITHMS"]


LEVEL_ALGORITHMS: Dict[str, str] = {
    "high": "AES-GCM",
    "medium": "ChaCha20-Poly1305",
}
"""AEAD algorithm per protection level, matching the V1 session."""


class DomainSeparatedSession:
    """Derive and apply per-block protection material for a single image.

    One instance corresponds to exactly one image, because the per-image
    ``image_nonce`` is what supplies HKDF freshness. Reusing an instance across
    images would reintroduce keystream reuse.

    Args:
        master_key: The 32-byte long-term secret.
        image_nonce: Optional 16-byte public salt. A fresh value is generated
            when omitted (encryption); decryption must pass the value read from
            the container header.
    """

    def __init__(self, master_key: bytes, image_nonce: bytes | None = None) -> None:
        self.image_nonce = (
            generate_image_nonce() if image_nonce is None else bytes(image_nonce)
        )
        if len(self.image_nonce) != IMAGE_NONCE_BYTES:
            raise ValueError(f"image_nonce must be {IMAGE_NONCE_BYTES} bytes")
        self._master_key = bytes(master_key)
        self.prk = derive_image_prk(self._master_key, self.image_nonce)
        self._aead_cache: Dict[Tuple[int, str], AEADPrimitive] = {}

    # ------------------------------------------------------------------
    # Key material
    # ------------------------------------------------------------------

    def permutation_seed_material(self) -> bytes:
        """Return the secret 32-byte seed for the block permutation source."""
        return derive_permutation_seed_material(self.prk)

    def build_permutation(self) -> ChaosPermutation:
        """Build a permutation source seeded from the secret per-image PRK.

        The V1 profile seeds this from the public ``chaos.seed``, which is why
        it had to publish ``perm_indices``. Here the permutation is
        reconstructible only with the master key.
        """
        return ChaosPermutation(
            ChaosGenerator(seed=self.permutation_seed_material(), backend="aes_ctr")
        )

    def mask_seed_material(self, block_index: int, level: str) -> bytes:
        """Return the secret keystream seed for one block's diffusion mask."""
        return derive_block_key(self.prk, PURPOSE_MASK, block_index, level)

    def _aead_for(self, block_index: int, level: str) -> AEADPrimitive:
        """Return (and memoize) the AEAD primitive for one block and level."""
        cache_key = (block_index, level)
        primitive = self._aead_cache.get(cache_key)
        if primitive is None:
            algorithm = LEVEL_ALGORITHMS[level]
            key = derive_block_key(self.prk, PURPOSE_AEAD, block_index, level)
            primitive = AEADPrimitive(key=key, algorithm=algorithm)
            self._aead_cache[cache_key] = primitive
        return primitive

    # ------------------------------------------------------------------
    # Block-level operations
    # ------------------------------------------------------------------

    def diffuse(self, data: np.ndarray, rounds: int, block_index: int, level: str) -> np.ndarray:
        """XOR ``data`` with a per-block keystream mask.

        The operation is its own inverse for a fixed ``(block_index, level,
        rounds)`` because a fresh generator restarts the same keystream.

        Args:
            data: Block pixels.
            rounds: Number of XOR rounds; ``<= 0`` returns a copy unchanged.
            block_index: Zero-based index used for domain separation.
            level: Protection level used for domain separation.

        Returns:
            The diffused (or, when applied twice, restored) block.
        """
        if rounds <= 0:
            return data.copy()
        generator = ChaosGenerator(
            seed=self.mask_seed_material(block_index, level), backend="aes_ctr"
        )
        result = data.copy()
        for _ in range(rounds):
            result = np.bitwise_xor(result, generator.generate_mask(result.shape))
        return result

    def _aad(self, block_index: int, level: str, extra_aad: bytes) -> bytes:
        """Build associated data binding image, index, and level.

        Mirrors the V1 format but substitutes the per-image ``image_nonce`` for
        the session id, so a block cannot be replayed into another image.
        """
        return (
            self.image_nonce
            + block_index.to_bytes(4, "big")
            + level.encode("ascii")
            + extra_aad
        )

    def encrypt_block(
        self,
        data: bytes,
        level: str,
        block_index: int,
        extra_aad: bytes = b"",
    ) -> Tuple[bytes, bytes]:
        """Encrypt one block under a derived key and deterministic nonce.

        Args:
            data: Block bytes, already diffused when the level calls for it.
            level: ``high``, ``medium``, or a weak level.
            block_index: Zero-based block index.
            extra_aad: Extra context to authenticate.

        Returns:
            ``(nonce, ciphertext)``. Weak levels return ``(b"", data)``,
            matching V1: they carry no authentication tag.
        """
        if level not in LEVEL_ALGORITHMS:
            return b"", data
        nonce = derive_block_nonce(self.prk, block_index, level)
        return self._aead_for(block_index, level).encrypt(
            data, aad=self._aad(block_index, level, extra_aad), nonce=nonce
        )

    def decrypt_block(
        self,
        nonce: bytes,
        ciphertext: bytes,
        level: str,
        block_index: int,
        extra_aad: bytes = b"",
    ) -> bytes:
        """Authenticate and decrypt one block.

        ``nonce`` is accepted for interface parity with the V1 session but is
        re-derived rather than trusted, so a tampered stored nonce cannot steer
        decryption.
        """
        if level not in LEVEL_ALGORITHMS:
            return ciphertext
        del nonce
        derived_nonce = derive_block_nonce(self.prk, block_index, level)
        return self._aead_for(block_index, level).decrypt(
            derived_nonce, ciphertext, self._aad(block_index, level, extra_aad)
        )

    def get_session_info(self) -> Dict[str, str]:
        """Describe the active profile for run artifacts."""
        return {
            "kdf_profile": "v2_domain_separated",
            "image_nonce": self.image_nonce.hex(),
            "algorithm_high": LEVEL_ALGORITHMS["high"],
            "algorithm_medium": LEVEL_ALGORITHMS["medium"],
            "nonce_source": "derived_deterministic",
            "mask_key_source": "hkdf_expand_from_master_key",
            "permutation_key_source": "hkdf_expand_from_master_key",
        }
