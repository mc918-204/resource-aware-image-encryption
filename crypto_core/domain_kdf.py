"""Domain-separated key derivation for the v2 protection profile.

Motivation (F1/M10 of the 20260907 improvement plan)
----------------------------------------------------
The frozen V1 profile derives the diffusion-mask keystream from
``chaos.seed`` -- a public configuration constant -- via
``SHA-256(LE8(seed))`` (see :mod:`src.chaos_engine.keystream`). The
long-term ``master_key`` never enters that derivation. Under the
white-box threat model the manuscript itself assumes, this makes the
weak-protection (``low``) layer a *deterministically recoverable*
one-time pad that is additionally reused across every image encrypted
with the same configuration -- a textbook two-time-pad exposure.

This module provides the v2 replacement. Every keystream, permutation
source, block key, and block nonce is derived from the secret
``master_key`` through HKDF-SHA256 with

* a *per-image* random 16-byte ``image_nonce`` as the HKDF **salt**
  (public, stored in the container header, but fresh for every image),
  which supplies the freshness that the V1 profile lacks; and
* a structured ``info`` string that binds the *purpose*, *block index*,
  and *protection level*, which supplies the domain separation.

Nothing here mutates :mod:`src.crypto_core.key_manager` or
:mod:`src.crypto_core.session`; the V1 path stays byte-for-byte
reproducible. Callers opt in through the ``encryption.kdf_profile``
configuration key.

Comparison with related work
----------------------------
TDS-SE (Sun et al., 2026) derives ``PRK_t = HKDF-Extract(salt=N_t,
IKM=K_master)`` and ``K_{t,i,j} = HKDF-Expand(PRK_t, "TILE"||i||j, 32)``
per remote-sensing tile. The construction below is the same standard
two-step HKDF pattern applied to image blocks, extended to also cover
the diffusion mask, the permutation source, and deterministic nonces.
Domain-separated derivation is therefore **not** claimed as novel; what
is new in this project is the measured demonstration that omitting it
from a selective-encryption weak layer is exploitable (see
``experiments/run_cpa_weak_layer_attack.py``).
"""

from __future__ import annotations

import os
from typing import Final

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand

__all__ = [
    "AEAD_NONCE_BYTES",
    "IMAGE_NONCE_BYTES",
    "KDF_LABEL",
    "KDF_PROFILE_V1",
    "KDF_PROFILE_V2",
    "PRK_BYTES",
    "PURPOSE_AEAD",
    "PURPOSE_MASK",
    "PURPOSE_NONCE",
    "PURPOSE_PERM",
    "SUPPORTED_KDF_PROFILES",
    "derive_block_key",
    "derive_block_nonce",
    "derive_image_prk",
    "derive_permutation_seed_material",
    "generate_image_nonce",
    "normalize_kdf_profile",
]


KDF_LABEL: Final[bytes] = b"RAISE-v1"
"""Protocol label prefixed to every ``info`` string.

``RAISE`` = Resource-Aware Image Selective Encryption. The trailing
``v1`` versions the *info encoding*, not the profile name; bumping it
would invalidate previously derived keys, so it is frozen.
"""

IMAGE_NONCE_BYTES: Final[int] = 16
PRK_BYTES: Final[int] = 32
AEAD_NONCE_BYTES: Final[int] = 12

KDF_PROFILE_V1: Final[str] = "v1_session"
KDF_PROFILE_V2: Final[str] = "v2_domain_separated"
SUPPORTED_KDF_PROFILES: Final[tuple[str, ...]] = (KDF_PROFILE_V1, KDF_PROFILE_V2)

PURPOSE_AEAD: Final[bytes] = b"AEAD"
PURPOSE_MASK: Final[bytes] = b"MASK"
PURPOSE_PERM: Final[bytes] = b"PERM"
PURPOSE_NONCE: Final[bytes] = b"NONCE"

_ALLOWED_PURPOSES: Final[frozenset[bytes]] = frozenset(
    {PURPOSE_AEAD, PURPOSE_MASK, PURPOSE_PERM, PURPOSE_NONCE}
)
_ALLOWED_LEVELS: Final[frozenset[str]] = frozenset({"none", "low", "medium", "high"})
_MAX_BLOCK_INDEX: Final[int] = 2**32 - 1
_INFO_SEPARATOR: Final[bytes] = b"|"


def normalize_kdf_profile(value: object) -> str:
    """Normalize a configured KDF profile name, defaulting to V1.

    ``None`` and the empty string resolve to :data:`KDF_PROFILE_V1` so
    that historical configs and packages without a ``kdf_profile`` field
    keep decrypting under the frozen V1 semantics.

    Raises:
        TypeError: ``value`` is neither ``None`` nor a string.
        ValueError: ``value`` names an unsupported profile.
    """
    if value is None:
        return KDF_PROFILE_V1
    if not isinstance(value, str):
        raise TypeError("kdf_profile must be a string or None")
    normalized = value.strip().lower()
    if not normalized:
        return KDF_PROFILE_V1
    if normalized not in SUPPORTED_KDF_PROFILES:
        raise ValueError(
            f"Unsupported kdf_profile {value!r}; "
            f"expected one of {list(SUPPORTED_KDF_PROFILES)}"
        )
    return normalized


def generate_image_nonce() -> bytes:
    """Return a fresh 16-byte per-image HKDF salt from the OS CSPRNG.

    The value is public: it is written to the container header in the
    clear so that decryption can re-derive every subkey. Its role is
    freshness, not secrecy -- it guarantees that two images protected
    under the same ``master_key`` never share a keystream, which is
    precisely the property the V1 profile lacks.
    """
    return os.urandom(IMAGE_NONCE_BYTES)


def _validate_master_key(master_key: bytes) -> bytes:
    """Validate the long-term key material used as HKDF input keying material."""
    if not isinstance(master_key, (bytes, bytearray)):
        raise TypeError("master_key must be bytes")
    if len(master_key) != 32:
        raise ValueError("master_key must be exactly 32 bytes (256 bits)")
    return bytes(master_key)


def _validate_image_nonce(image_nonce: bytes) -> bytes:
    """Validate the per-image HKDF salt."""
    if not isinstance(image_nonce, (bytes, bytearray)):
        raise TypeError("image_nonce must be bytes")
    if len(image_nonce) != IMAGE_NONCE_BYTES:
        raise ValueError(f"image_nonce must be exactly {IMAGE_NONCE_BYTES} bytes")
    return bytes(image_nonce)


def _validate_prk(prk: bytes) -> bytes:
    """Validate a pseudo-random key produced by :func:`derive_image_prk`."""
    if not isinstance(prk, (bytes, bytearray)):
        raise TypeError("prk must be bytes")
    if len(prk) != PRK_BYTES:
        raise ValueError(f"prk must be exactly {PRK_BYTES} bytes")
    return bytes(prk)


def _validate_purpose(purpose: bytes) -> bytes:
    """Validate a domain-separation purpose tag."""
    if not isinstance(purpose, (bytes, bytearray)):
        raise TypeError("purpose must be bytes")
    value = bytes(purpose)
    if value not in _ALLOWED_PURPOSES:
        raise ValueError(
            f"Unsupported purpose {value!r}; expected one of "
            f"{sorted(item.decode('ascii') for item in _ALLOWED_PURPOSES)}"
        )
    return value


def _validate_block_index(block_index: int) -> int:
    """Validate a block index that will be encoded as four big-endian bytes."""
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise TypeError("block_index must be an int")
    if not 0 <= block_index <= _MAX_BLOCK_INDEX:
        raise ValueError(f"block_index must lie in [0, {_MAX_BLOCK_INDEX}]")
    return block_index


def _validate_level(level: str) -> str:
    """Validate a protection level name used for domain separation."""
    if not isinstance(level, str):
        raise TypeError("level must be a string")
    normalized = level.strip().lower()
    if normalized not in _ALLOWED_LEVELS:
        raise ValueError(
            f"Unsupported level {level!r}; expected one of {sorted(_ALLOWED_LEVELS)}"
        )
    return normalized


def _i2osp4(value: int) -> bytes:
    """Encode a non-negative integer as four big-endian bytes (RFC 8017 I2OSP)."""
    return value.to_bytes(4, byteorder="big", signed=False)


def _build_info(purpose: bytes, block_index: int | None, level: str | None) -> bytes:
    """Assemble a length-unambiguous HKDF ``info`` string.

    The encoding is ``RAISE-v1|<PURPOSE>[|<I2OSP4(idx)>][|<level>]``.
    ``purpose`` is drawn from a closed set of fixed-width tags and the
    block index is fixed-width, so no two distinct argument tuples can
    produce the same byte string.
    """
    parts = [KDF_LABEL, purpose]
    if block_index is not None:
        parts.append(_i2osp4(block_index))
    if level is not None:
        parts.append(level.encode("ascii"))
    return _INFO_SEPARATOR.join(parts)


def derive_image_prk(master_key: bytes, image_nonce: bytes) -> bytes:
    """Derive a per-image pseudo-random key via ``HKDF-Extract``.

    Implements ``PRK = HKDF-Extract(salt=image_nonce, IKM=master_key)``.
    The 16-byte ``image_nonce`` is public but fresh per image, so every
    image gets an independent key hierarchy even under a fixed
    ``master_key``.

    Args:
        master_key: The 32-byte long-term secret.
        image_nonce: The 16-byte public per-image salt, typically from
            :func:`generate_image_nonce`.

    Returns:
        A 32-byte pseudo-random key to pass to the ``derive_block_*``
        helpers.
    """
    key = _validate_master_key(master_key)
    salt = _validate_image_nonce(image_nonce)
    # `cryptography` exposes no bare HKDF-Extract, so this is
    # Extract(salt, IKM) followed by a single-block Expand with an empty
    # info. The composition is still a PRF of (salt, IKM), which is the
    # only property the block-level Expand calls below rely on.
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=PRK_BYTES,
        salt=salt,
        info=b"",
        backend=default_backend(),
    )
    return hkdf.derive(key)


def _expand(prk: bytes, info: bytes, length: int) -> bytes:
    """Run ``HKDF-Expand`` on a validated PRK."""
    expander = HKDFExpand(
        algorithm=hashes.SHA256(),
        length=length,
        info=info,
        backend=default_backend(),
    )
    return expander.derive(prk)


def derive_block_key(
    prk: bytes,
    purpose: bytes,
    block_index: int,
    level: str,
) -> bytes:
    """Derive a 32-byte per-block key for one purpose and protection level.

    Implements ``HKDF-Expand(prk, info=RAISE-v1|purpose|I2OSP4(idx)|level, 32)``.

    Unlike the V1 session profile -- which derives exactly two keys per
    session and feeds the block index only into the AEAD's associated
    data -- the block index and level here are inputs to the *key
    derivation* itself. Distinct ``(purpose, block_index, level)`` tuples
    therefore yield computationally independent keys.

    Args:
        prk: A 32-byte PRK from :func:`derive_image_prk`.
        purpose: One of :data:`PURPOSE_AEAD`, :data:`PURPOSE_MASK`, or
            :data:`PURPOSE_PERM`.
        block_index: Zero-based block index, at most ``2**32 - 1``.
        level: Protection level name (``none``/``low``/``medium``/``high``).

    Returns:
        A 32-byte key suitable for AES-256, ChaCha20, or an AES-CTR
        keystream seed.
    """
    validated_prk = _validate_prk(prk)
    validated_purpose = _validate_purpose(purpose)
    if validated_purpose == PURPOSE_NONCE:
        raise ValueError(
            "purpose=NONCE is reserved for derive_block_nonce; "
            "use PURPOSE_AEAD, PURPOSE_MASK, or PURPOSE_PERM"
        )
    index = _validate_block_index(block_index)
    normalized_level = _validate_level(level)
    info = _build_info(validated_purpose, index, normalized_level)
    return _expand(validated_prk, info, PRK_BYTES)


def derive_block_nonce(prk: bytes, block_index: int, level: str) -> bytes:
    """Derive a deterministic 12-byte AEAD nonce for one block.

    Implements ``HKDF-Expand(prk, info=RAISE-v1|NONCE|I2OSP4(idx)|level, 12)``.

    This replaces ``os.urandom(12)`` in :meth:`AEADPrimitive.encrypt`
    for the v2 profile. Because the nonce is a function of the fresh
    per-image PRK and of ``(block_index, level)``, it is:

    * **unique** within an image -- distinct blocks get distinct info
      strings, and each ``(block_index, level)`` pair is used at most
      once per image;
    * **unique** across images -- a fresh ``image_nonce`` changes the
      PRK, hence every derived nonce;
    * **reproducible** -- the same inputs regenerate the same nonce, so
      nonces need not be stored per block; and
    * free of the birthday-bound concern that random nonces carry under
      a long-lived session key.

    Args:
        prk: A 32-byte PRK from :func:`derive_image_prk`.
        block_index: Zero-based block index, at most ``2**32 - 1``.
        level: Protection level name.

    Returns:
        A 12-byte nonce for AES-GCM or ChaCha20-Poly1305.
    """
    validated_prk = _validate_prk(prk)
    index = _validate_block_index(block_index)
    normalized_level = _validate_level(level)
    info = _build_info(PURPOSE_NONCE, index, normalized_level)
    return _expand(validated_prk, info, AEAD_NONCE_BYTES)


def derive_permutation_seed_material(prk: bytes) -> bytes:
    """Derive the 32-byte seed material for the block permutation source.

    The V1 profile builds its permutation from the public ``chaos.seed``,
    so ``perm_indices`` had to be stored in the clear (and could be
    recomputed by anyone regardless). Seeding from the per-image PRK
    makes the permutation both secret and fresh, which is what allows
    the v2 container to stop publishing ``perm_indices``.

    Args:
        prk: A 32-byte PRK from :func:`derive_image_prk`.

    Returns:
        32 bytes to pass as ``ChaosGenerator(seed=...)`` with the
        ``aes_ctr`` backend.
    """
    validated_prk = _validate_prk(prk)
    info = _build_info(PURPOSE_PERM, None, None)
    return _expand(validated_prk, info, PRK_BYTES)
