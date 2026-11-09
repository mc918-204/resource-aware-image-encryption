"""Container encapsulations that trade bytes for metadata privacy (plan C7).

The defect being addressed (finding M9)
---------------------------------------
The V1 JSON container publishes everything an attacker needs to locate the
protected regions without touching a single ciphertext byte. The project's
own ``protection_chain_audit.json`` records ``perm_indices_public: true``,
``level_per_payload_public: true``, ``nonce_per_payload_public: true`` and
``session_id_public: true``. Worse, the per-block record *length* leaks the
level even if the ``level`` field were removed: ``none``/``low`` blocks are
768 bytes while ``medium``/``high`` blocks are 796 bytes, because AEAD
appends a 16-byte tag. Correlation with external sensitive regions must be
evaluated with aligned labels and an explicit attacker mapping capability.

Content confidentiality and allocation-metadata confidentiality are
therefore require separate evaluation.

The four encapsulations
-----------------------
Each is fully reversible, so the mitigation cost can be measured rather than
asserted.

===================  ===========================================  ==================================
Encapsulation        What it changes                              Expected cost
===================  ===========================================  ==================================
``plain_v1``         nothing (baseline, matches the V1 format)    0
``padded``           every record padded to one uniform length    +3.6% payload bytes
``hidden_levels``    level map and permutation moved into an       + header bytes
                     AEAD-encrypted header
``padded_hidden``    both of the above                            largest
===================  ===========================================  ==================================

Padding is filled from a key-derived keystream rather than zeros, so padded
regions are not distinguishable from ciphertext by inspection.

Why the true length need not be stored
--------------------------------------
Recording each record's real length would reintroduce exactly the leak that
padding removes. It is unnecessary: given the block geometry (``block_size``,
``channels``, ``dtype``, all needed for reconstruction anyway) and the
block's level, the true length is determined -- ``footprint`` bytes for
``none``/``low``, ``footprint + 16`` for AEAD levels. Under the
``hidden_*`` variants the decryptor recovers the level map from the encrypted
header first, then parses. An attacker without the key gets neither.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src.chaos_engine.generator import ChaosGenerator
from src.crypto_core.aead import AEADPrimitive
from src.crypto_core.domain_kdf import (
    PURPOSE_AEAD,
    PURPOSE_MASK,
    derive_block_key,
    derive_image_prk,
    generate_image_nonce,
)
from src.data_io.metadata import EncryptedPackage, ImageMetadata

__all__ = [
    "AEAD_TAG_BYTES",
    "CONTAINER_ENCAPSULATIONS",
    "ENCAP_HIDDEN_LEVELS",
    "ENCAP_PADDED",
    "ENCAP_PADDED_HIDDEN",
    "ENCAP_PLAIN_V1",
    "ContainerStats",
    "ContainerV2",
    "MetadataObservation",
]

AEAD_TAG_BYTES = 16
"""Authentication tag length appended by AES-GCM and ChaCha20-Poly1305."""

ENCAP_PLAIN_V1 = "plain_v1"
ENCAP_PADDED = "padded"
ENCAP_HIDDEN_LEVELS = "hidden_levels"
ENCAP_PADDED_HIDDEN = "padded_hidden"

CONTAINER_ENCAPSULATIONS: tuple[str, ...] = (
    ENCAP_PLAIN_V1,
    ENCAP_PADDED,
    ENCAP_HIDDEN_LEVELS,
    ENCAP_PADDED_HIDDEN,
)

_WEAK_LEVELS = frozenset({"none", "low"})
_HEADER_BLOCK_INDEX = 0
_HEADER_LEVEL = "high"


@dataclass(frozen=True)
class ContainerStats:
    """Byte and timing accounting for one serialized container."""

    encapsulation: str
    record_count: int
    payload_bytes: int
    header_bytes: int
    metadata_bytes: int
    total_bytes: int
    distinct_record_lengths: int
    levels_public: bool
    permutation_public: bool
    nonces_public: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible record for run artifacts."""
        return {
            "encapsulation": self.encapsulation,
            "record_count": self.record_count,
            "payload_bytes": self.payload_bytes,
            "header_bytes": self.header_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "distinct_record_lengths": self.distinct_record_lengths,
            "levels_public": self.levels_public,
            "permutation_public": self.permutation_public,
            "nonces_public": self.nonces_public,
        }


@dataclass(frozen=True)
class MetadataObservation:
    """Exactly what a metadata-only attacker can read from a container.

    Deliberately excludes every ciphertext byte. This is the input to the E4
    attacks, so the class boundary *is* the threat model: anything present here
    is conceded to the attacker, anything absent is protected.
    """

    encapsulation: str
    record_lengths: List[int]
    has_nonce: List[bool]
    levels: List[str] | None
    permutation: List[int] | None
    record_count: int
    total_payload_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible record."""
        return {
            "encapsulation": self.encapsulation,
            "record_lengths": list(self.record_lengths),
            "has_nonce": list(self.has_nonce),
            "levels": None if self.levels is None else list(self.levels),
            "permutation": None if self.permutation is None else list(self.permutation),
            "record_count": self.record_count,
            "total_payload_bytes": self.total_payload_bytes,
        }


class ContainerV2:
    """Serialize and parse an :class:`EncryptedPackage` under one encapsulation.

    Args:
        master_key: 32-byte long-term key. Required by the ``hidden_*``
            encapsulations to key the header AEAD and the padding keystream.
        encapsulation: One of :data:`CONTAINER_ENCAPSULATIONS`.

    Raises:
        ValueError: unknown encapsulation, or a malformed master key.
    """

    def __init__(self, master_key: bytes, encapsulation: str = ENCAP_PLAIN_V1) -> None:
        if encapsulation not in CONTAINER_ENCAPSULATIONS:
            raise ValueError(
                f"unsupported encapsulation {encapsulation!r}; "
                f"expected one of {list(CONTAINER_ENCAPSULATIONS)}"
            )
        if not isinstance(master_key, (bytes, bytearray)) or len(master_key) != 32:
            raise ValueError("master_key must be exactly 32 bytes")
        self.master_key = bytes(master_key)
        self.encapsulation = encapsulation

    # ------------------------------------------------------------------
    # Properties of the chosen encapsulation
    # ------------------------------------------------------------------

    @property
    def pads_records(self) -> bool:
        """Whether records are padded to a uniform length."""
        return self.encapsulation in {ENCAP_PADDED, ENCAP_PADDED_HIDDEN}

    @property
    def hides_allocation(self) -> bool:
        """Whether the level map and permutation are encrypted."""
        return self.encapsulation in {ENCAP_HIDDEN_LEVELS, ENCAP_PADDED_HIDDEN}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_footprint(metadata: ImageMetadata) -> int:
        """Plaintext bytes per block, from geometry already needed to decrypt."""
        dtype_size = np.dtype(np.uint8).itemsize
        return metadata.block_size * metadata.block_size * metadata.channels * dtype_size

    def _uniform_record_length(self, metadata: ImageMetadata) -> int:
        """Target length every padded record is raised to.

        The maximum over levels: an AEAD record is ``footprint + tag``, which is
        strictly the largest, so no record is ever truncated.
        """
        return self._block_footprint(metadata) + AEAD_TAG_BYTES

    def _header_key(self, header_nonce: bytes) -> bytes:
        """Derive the AEAD key protecting the allocation header."""
        prk = derive_image_prk(self.master_key, header_nonce)
        return derive_block_key(prk, PURPOSE_AEAD, _HEADER_BLOCK_INDEX, _HEADER_LEVEL)

    def _padding_generator(self, header_nonce: bytes) -> ChaosGenerator:
        """Build one keystream that supplies all padding for an image.

        Zero padding would be trivially separable from ciphertext, defeating the
        point of equalizing lengths, so padding is drawn from a key-derived
        AES-CTR keystream.

        A single generator per image is used rather than one per record: a
        per-record HKDF plus generator construction dominated the encapsulation
        cost (measured 176 ms for a ~1000-block image versus 19 ms unpadded),
        which would have made the mitigation look far more expensive than the
        padding itself is. Padding bytes never need to be re-derived at a
        specific offset -- they are discarded on parse -- so sequential
        consumption from one stream is sufficient.
        """
        prk = derive_image_prk(self.master_key, header_nonce)
        seed = derive_block_key(prk, PURPOSE_MASK, 0, "high")
        return ChaosGenerator(seed=seed, backend="aes_ctr")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self, package: EncryptedPackage) -> tuple[Dict[str, Any], ContainerStats]:
        """Encode a package into a container document plus byte accounting.

        Args:
            package: The package produced by the encryption pipeline.

        Returns:
            ``(document, stats)``. ``document`` is JSON-serializable and can be
            fed back to :meth:`parse`.
        """
        metadata = package.metadata
        records = sorted(package.payload, key=lambda item: item["index"])
        footprint = self._block_footprint(metadata)
        target_length = self._uniform_record_length(metadata)

        header_nonce = generate_image_nonce() if self.hides_allocation or self.pads_records else b""
        levels = [str(item["level"]) for item in records]

        padding = self._padding_generator(header_nonce) if self.pads_records else None
        encoded_records: List[Dict[str, Any]] = []
        payload_bytes = 0
        for position, item in enumerate(records):
            data = bytes(item["data"])
            nonce = bytes(item["nonce"])
            if padding is not None:
                # Equalize length. `none`/`low` records are `footprint` bytes and
                # AEAD records are `footprint + tag`; both are raised to the
                # same target so the level is no longer readable from length.
                shortfall = target_length - len(data)
                if shortfall < 0:
                    raise ValueError(
                        f"record {position} is {len(data)} bytes, longer than the "
                        f"uniform target {target_length}"
                    )
                data = data + padding.generate_bytes(shortfall)
                # A missing nonce is itself a level tell (weak levels carry
                # none), so weak records get a constant-length pseudorandom
                # filler in the nonce field.
                if not nonce:
                    nonce = padding.generate_bytes(12)

            record: Dict[str, Any] = {
                "index": int(item["index"]),
                "data": base64.b64encode(data).decode("ascii"),
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "shape": list(item["shape"]),
                "dtype": str(item["dtype"]),
            }
            if not self.hides_allocation:
                record["level"] = str(item["level"])
            encoded_records.append(record)
            payload_bytes += len(data) + len(nonce)

        metadata_dict = metadata.to_dict()
        header_bytes = 0
        document: Dict[str, Any] = {
            "container_version": "container_v2",
            "encapsulation": self.encapsulation,
            "payload": encoded_records,
        }

        if self.hides_allocation:
            # The level map and the permutation are the two strongest location
            # predictors, so they move behind AEAD together.
            allocation = json.dumps(
                {
                    "levels": levels,
                    "perm_indices": metadata_dict.get("perm_indices"),
                },
                sort_keys=True,
            ).encode("utf-8")
            primitive = AEADPrimitive(self._header_key(header_nonce), "AES-GCM")
            aad = b"container_v2|allocation_header|" + self.encapsulation.encode("ascii")
            used_nonce, ciphertext = primitive.encrypt(allocation, aad=aad)
            document["allocation_header"] = {
                "nonce": base64.b64encode(used_nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
            header_bytes = len(used_nonce) + len(ciphertext)
            metadata_dict = dict(metadata_dict)
            metadata_dict["perm_indices"] = None

        if header_nonce:
            document["header_nonce"] = base64.b64encode(header_nonce).decode("ascii")
            header_bytes += len(header_nonce)

        document["metadata"] = metadata_dict
        metadata_bytes = len(
            json.dumps(metadata_dict, sort_keys=True, default=str).encode("utf-8")
        )

        record_lengths = {len(base64.b64decode(r["data"])) for r in encoded_records}
        stats = ContainerStats(
            encapsulation=self.encapsulation,
            record_count=len(encoded_records),
            payload_bytes=payload_bytes,
            header_bytes=header_bytes,
            metadata_bytes=metadata_bytes,
            total_bytes=payload_bytes + header_bytes + metadata_bytes,
            distinct_record_lengths=len(record_lengths),
            levels_public=not self.hides_allocation,
            permutation_public=(
                not self.hides_allocation and metadata_dict.get("perm_indices") is not None
            ),
            nonces_public=True,
            )
        return document, stats

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, document: Mapping[str, Any]) -> EncryptedPackage:
        """Recover the original package from a container document.

        Reverses padding and, for the ``hidden_*`` variants, decrypts the
        allocation header to learn the levels needed for parsing.

        Raises:
            ValueError: the document's encapsulation does not match this
                instance, or a hidden header is missing.
        """
        encapsulation = str(document.get("encapsulation"))
        if encapsulation != self.encapsulation:
            raise ValueError(
                f"container encapsulation is {encapsulation!r}, "
                f"but this parser is configured for {self.encapsulation!r}"
            )

        metadata = ImageMetadata.from_dict(dict(document["metadata"]))
        footprint = self._block_footprint(metadata)
        records = list(document["payload"])

        header_nonce = (
            base64.b64decode(document["header_nonce"])
            if document.get("header_nonce")
            else b""
        )

        levels: List[str]
        perm_indices = metadata.perm_indices
        if self.hides_allocation:
            header = document.get("allocation_header")
            if not header:
                raise ValueError(
                    f"encapsulation {self.encapsulation!r} requires an allocation_header"
                )
            primitive = AEADPrimitive(self._header_key(header_nonce), "AES-GCM")
            aad = b"container_v2|allocation_header|" + self.encapsulation.encode("ascii")
            plaintext = primitive.decrypt(
                base64.b64decode(header["nonce"]),
                base64.b64decode(header["ciphertext"]),
                aad,
            )
            allocation = json.loads(plaintext.decode("utf-8"))
            levels = [str(level) for level in allocation["levels"]]
            perm_indices = allocation.get("perm_indices")
        else:
            levels = [str(record["level"]) for record in records]

        payload: List[Dict[str, Any]] = []
        for position, record in enumerate(records):
            level = levels[position]
            data = base64.b64decode(record["data"])
            nonce = base64.b64decode(record["nonce"])
            if self.pads_records:
                # The true length is determined by the level, so no stored
                # length field is needed -- and none is stored, since that field
                # would leak exactly what padding hides.
                true_length = (
                    footprint if level in _WEAK_LEVELS else footprint + AEAD_TAG_BYTES
                )
                data = data[:true_length]
                if level in _WEAK_LEVELS:
                    nonce = b""
            payload.append(
                {
                    "index": int(record["index"]),
                    "level": level,
                    "nonce": nonce,
                    "data": data,
                    "shape": tuple(record["shape"]),
                    "dtype": str(record["dtype"]),
                }
            )

        restored_metadata = ImageMetadata.from_dict(
            {**metadata.to_dict(), "perm_indices": perm_indices}
        )
        return EncryptedPackage(metadata=restored_metadata, payload=payload)

    # ------------------------------------------------------------------
    # Attacker's view
    # ------------------------------------------------------------------

    def observe(self, document: Mapping[str, Any]) -> MetadataObservation:
        """Extract only what a metadata-only attacker can read.

        No ciphertext content is included, by construction. This is the sole
        input to the E4 attacks.
        """
        records = list(document["payload"])
        lengths = [len(base64.b64decode(record["data"])) for record in records]
        has_nonce = [bool(base64.b64decode(record["nonce"])) for record in records]
        levels = (
            None
            if self.hides_allocation
            else [str(record["level"]) for record in records]
        )
        metadata = dict(document["metadata"])
        permutation = metadata.get("perm_indices")
        return MetadataObservation(
            encapsulation=self.encapsulation,
            record_lengths=lengths,
            has_nonce=has_nonce,
            levels=levels,
            permutation=None if permutation is None else list(permutation),
            record_count=len(records),
            total_payload_bytes=int(sum(lengths)),
        )

    # ------------------------------------------------------------------
    # Disk round-trip
    # ------------------------------------------------------------------

    def save(self, package: EncryptedPackage, path: str | Path) -> ContainerStats:
        """Serialize a package to a JSON file and return its byte accounting."""
        document, stats = self.serialize(package)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document), encoding="utf-8")
        return stats

    def load(self, path: str | Path) -> EncryptedPackage:
        """Read a container file and recover the package."""
        return self.parse(json.loads(Path(path).read_text(encoding="utf-8")))
