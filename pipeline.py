import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import numpy as np

from src.utils.logger import setup_logger
from src.data_io.metadata import ImageMetadata, EncryptedPackage
from src.preprocess.padding import pad_image, unpad_image
from src.preprocess.block_split import split_into_blocks
from src.preprocess.block_merge import merge_blocks
from src.preprocess.features import extract_features_batch
from src.policy_model.predictor import Predictor
from src.budget_allocator.allocator import BudgetAllocator
from src.chaos_engine.generator import ChaosGenerator
from src.chaos_engine.permutation import ChaosPermutation
from src.crypto_core.session import CryptoSession
from src.crypto_core.domain_kdf import (
    KDF_PROFILE_V1,
    KDF_PROFILE_V2,
    normalize_kdf_profile,
)
from src.crypto_core.domain_session import DomainSeparatedSession


class DecryptionAuthenticationError(RuntimeError):
    """Raised when strict research decryption rejects an authenticated block."""


class EncryptionPipeline:
    SUPPORTED_BUDGET_TYPES = ("time",)

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.logger = setup_logger(self.cfg['app']['name'])
        
        # Initialize components
        self.block_size = self.cfg['image']['block_size']
        
        # Crypto Session
        master_key_hex = self.cfg['encryption']['master_key']
        self.crypto_session = CryptoSession(bytes.fromhex(master_key_hex))
        
        # Policy & Allocation
        self.predictor = Predictor(self.cfg)
        self.allocator = BudgetAllocator() # Uses updated cost model
        
        # A missing backend denotes the frozen V1 logistic format. Research
        # configs select hardened alternatives explicitly.
        chaos_cfg = self.cfg['chaos']
        self.chaos_seed = chaos_cfg['seed']
        self.chaos_r = chaos_cfg['r']
        self.chaos_backend = str(chaos_cfg.get('backend', 'logistic')).strip().lower()
        self.chaos = self._build_chaos(self.chaos_backend)

        # Additive research control (plan item C1). `v1_session` reproduces the
        # frozen V1 key schedule exactly and stays the default; a config must
        # opt into `v2_domain_separated` explicitly.
        self.master_key = bytes.fromhex(master_key_hex)
        self.kdf_profile = normalize_kdf_profile(
            self.cfg['encryption'].get('kdf_profile')
        )

    def _build_chaos(self, backend: str) -> ChaosPermutation:
        """Build a reproducible permutation/diffusion stage for one backend."""
        generator = ChaosGenerator(
            seed=self.chaos_seed,
            r=self.chaos_r,
            backend=backend,
        )
        return ChaosPermutation(generator)
        
    def encrypt(
        self,
        image: np.ndarray,
        budget: float = None,
        budget_type: str = "time",
    ) -> EncryptedPackage:
        """
        Execute full encryption pipeline.
        """
        self.logger.info(">>> Starting Encryption Pipeline")
        if budget_type not in self.SUPPORTED_BUDGET_TYPES:
            raise NotImplementedError(
                f"Budget type `{budget_type}` is not supported in the current pipeline. "
                "Only `time` is publication-safe in M1."
            )
        
        # Default budget
        if budget is None:
            budget = self.cfg.get('budget', {}).get('default_time', 2000.0)
            
        # 1. Preprocess: Pad & Split
        self.logger.info(f"Input image shape: {image.shape}")
        padded_img, pad_meta = pad_image(image, self.block_size)
        blocks, positions, split_meta = split_into_blocks(padded_img, self.block_size)
        self.logger.info(f"Split into {len(blocks)} blocks")
        
        # 2. Analysis: Feature Extraction & Scoring
        features = extract_features_batch(blocks)
        
        # Support fusion mode transparently
        mode = self.cfg['policy'].get('scorer', 'rule_based')
        if mode == "fusion":
            # Predictor returns dictionary in fusion mode if return_dict=True is implemented,
            # but we ensured predict_block_scores returns just the final array by default
            scores = self.predictor.predict_block_scores(blocks, features, mode=mode)
        elif mode == "cnn_classifier":
            res = self.predictor.predict_block_scores(blocks, features, mode=mode, return_dict=True)
            scores = res["scores"]
        else:
            scores = self.predictor.predict_block_scores(blocks, features, mode=mode)
        
        # 3. Allocation: Budget Allocation
        self.logger.info(f"Allocating budget: {budget} (Strategy: Prioritized Greedy)")
        alloc_result = self.allocator.allocate(scores, budget=budget, budget_type=budget_type)
        levels = alloc_result['levels']
        stats = alloc_result['stats']
        
        # Enhanced Logging
        self.logger.info(f"Allocation Result:")
        self.logger.info(f"  - Total Budget: {budget:.2f}")
        self.logger.info(f"  - Cost Used:    {stats['budget_used']:.2f} ({stats['budget_utilization']:.1f}%)")
        
        protection_level = stats.get('protection_level', stats.get('feasibility_status', 'UNKNOWN'))
        if protection_level == 'UNKNOWN':
            if stats['budget_utilization'] > 100:
                protection_level = "DEGRADED"
            elif stats['high_ratio'] == 1.0:
                protection_level = "FULL"
            else:
                protection_level = "PARTIAL"
                
        self.logger.info(f"  - Protection:   {protection_level}")
        self.logger.info(f"  - High Blocks:  {stats['high']} ({stats['high_ratio']*100:.1f}%)")
        self.logger.info(f"  - Med Blocks:   {stats['medium']} ({stats['medium_ratio']*100:.1f}%)")
        self.logger.info(f"  - Low Blocks:   {stats['low']} ({stats['low_ratio']*100:.1f}%)")
        self.logger.info(f"  - None Blocks:  {stats.get('none', 0)} ({stats.get('none_ratio', 0)*100:.1f}%)")
        
        package, _ = self._protect_blocks(padded_img, pad_meta, blocks, levels)
        self.logger.info("Encryption completed successfully.")
        return package

    def encrypt_with_levels(
        self,
        image: np.ndarray,
        levels: List[str],
    ) -> tuple[EncryptedPackage, Dict[str, float]]:
        """Encrypt using an externally supplied per-block level assignment.

        This is the entry point the benchmark runner uses so that the measured
        system is the *same* implementation the paper describes (plan item F2).
        Scoring and allocation happen in the runner -- which owns the method
        matrix, the fairness axes, and the timing protocol -- while the
        cryptographic chain here stays the single source of truth. No crypto is
        reimplemented in ``experiments/``.

        Args:
            image: The image to protect.
            levels: One level name per block, in unshuffled block order. Its
                length must equal the block count for ``image``.

        Returns:
            ``(package, stage_timings_ms)`` where ``stage_timings_ms`` reports
            per-stage wall-clock cost in milliseconds plus container sizes. See
            :meth:`_protect_blocks` for the key list.

        Raises:
            ValueError: ``levels`` has the wrong length or an unknown level.
        """
        padded_img, pad_meta = pad_image(image, self.block_size)
        blocks, _, _ = split_into_blocks(padded_img, self.block_size)
        if len(levels) != len(blocks):
            raise ValueError(
                f"levels length {len(levels)} does not match block count {len(blocks)}"
            )
        unknown = set(levels) - {"none", "low", "medium", "high"}
        if unknown:
            raise ValueError(f"unsupported protection levels: {sorted(unknown)}")
        return self._protect_blocks(padded_img, pad_meta, blocks, list(levels))

    def _protect_blocks(
        self,
        padded_img: np.ndarray,
        pad_meta: Dict[str, Any],
        blocks: np.ndarray,
        levels: List[str],
    ) -> tuple[EncryptedPackage, Dict[str, float]]:
        """Run permutation, diffusion, and AEAD over pre-assigned levels.

        Shared by :meth:`encrypt` and :meth:`encrypt_with_levels` so both go
        through identical cryptography.

        Returns:
            ``(package, stage_timings_ms)``. The timing dict carries
            ``permute_ms``, ``diffuse_aead_ms``, ``serialize_ms``,
            ``encryption_total_ms``, ``ciphertext_bytes_total``,
            ``metadata_bytes``, and ``payload_bytes_total``.
        """
        # Chaos: Permutation (Shuffling)
        # We shuffle ALL blocks, including 'none' ones, so their positions are
        # not directly readable from the container ordering.
        #
        # Under `v2_domain_separated` the permutation source is seeded from the
        # per-image secret PRK instead of the public `chaos.seed`, so the
        # resulting index vector need not be published (see C1/C7).
        timings: Dict[str, float] = {}
        domain_session = (
            DomainSeparatedSession(self.master_key)
            if self.kdf_profile == KDF_PROFILE_V2
            else None
        )
        permuter = (
            domain_session.build_permutation() if domain_session is not None else self.chaos
        )
        if domain_session is None:
            # Correctness fix found while wiring the real chain into the
            # benchmark (C2). `self.chaos` wraps a single *stateful* AES-CTR
            # generator: each permutation and diffusion call consumes keystream
            # and advances it. `decrypt()` has always called
            # `chaos.generator.reset()` before regenerating the permutation, so
            # it assumes the stream started at offset 0. Without a matching
            # reset here, only the FIRST encrypt() after construction was
            # recoverable -- a second encrypt() on the same instance silently
            # produced an undecryptable package (measured SAD 1032758 on a
            # 64x64 image).
            #
            # This is backward compatible: immediately after construction the
            # generator is already at offset 0, so the reset is a no-op for the
            # first call and every archived V1 package. It only repairs the
            # reuse case, which the benchmark needs because it holds one
            # pipeline across all images.
            self.chaos.generator.reset()
        permute_start = time.perf_counter()
        shuffled_blocks, perm_indices = permuter.shuffle_blocks(blocks)
        levels_arr = np.array(levels)
        shuffled_levels = levels_arr[perm_indices]
        timings["permute_ms"] = (time.perf_counter() - permute_start) * 1000.0

        # Block Processing (parallel when beneficial)
        num_workers = self.cfg.get('pipeline', {}).get('parallel_workers', 0)
        protect_start = time.perf_counter()
        payload = self._process_blocks(
            shuffled_blocks, shuffled_levels, num_workers, domain_session=domain_session
        )
        timings["diffuse_aead_ms"] = (time.perf_counter() - protect_start) * 1000.0

        # Metadata Construction
        serialize_start = time.perf_counter()
        meta = ImageMetadata(
            original_shape=pad_meta["original_shape"],
            padded_shape=padded_img.shape,
            block_size=self.block_size,
            num_blocks=len(blocks),
            channels=pad_meta["original_shape"][2] if len(pad_meta["original_shape"]) > 2 else 1,
            padding_info=pad_meta,
            session_id=self.crypto_session.session_id.decode('utf-8'),
            chaos_backend=self.chaos_backend,
            # V1 must keep publishing perm_indices for backward-compatible
            # decryption. v2 omits them: they are recoverable from the master
            # key, so publishing them would leak block order for free.
            perm_indices=None if domain_session is not None else perm_indices.tolist(),
            block_levels=None,
            kdf_profile=self.kdf_profile,
            image_nonce=(
                domain_session.image_nonce.hex() if domain_session is not None else None
            ),
        )
        package = EncryptedPackage(metadata=meta, payload=payload)
        timings["serialize_ms"] = (time.perf_counter() - serialize_start) * 1000.0
        timings["encryption_total_ms"] = (
            timings["permute_ms"] + timings["diffuse_aead_ms"] + timings["serialize_ms"]
        )

        # Container size accounting. Selective encryption trades coverage for
        # bytes, so ciphertext expansion is a first-class cost axis (plan C3.5).
        ciphertext_bytes = sum(len(item["data"]) for item in payload)
        nonce_bytes = sum(len(item["nonce"]) for item in payload)
        timings["ciphertext_bytes_total"] = float(ciphertext_bytes)
        timings["payload_bytes_total"] = float(ciphertext_bytes + nonce_bytes)
        timings["metadata_bytes"] = float(
            len(json.dumps(meta.to_dict(), default=str).encode("utf-8"))
        )
        return package, timings

    def decrypt(
        self,
        package: EncryptedPackage,
        *,
        fail_closed: bool | None = None,
    ) -> np.ndarray:
        """
        Execute full decryption pipeline.

        ``fail_closed`` is an additive research control. When enabled, an
        AEAD authentication or block decoding error aborts the operation with
        :class:`DecryptionAuthenticationError`. The default remains the
        historical V1 recovery behavior unless ``encryption.fail_closed`` is
        explicitly set in the configuration.
        """
        self.logger.info(">>> Starting Decryption Pipeline")

        strict_authentication = (
            self.cfg.get("encryption", {}).get("fail_closed", False)
            if fail_closed is None
            else fail_closed
        )
        if not isinstance(strict_authentication, bool):
            raise TypeError("fail_closed must be a boolean when provided")
        
        meta = package.metadata
        payload = package.payload

        # The KDF profile is read from the package, not from the local config, so
        # a v2-configured pipeline still decrypts archived V1 packages. Packages
        # written before this field existed normalize to `v1_session`.
        package_profile = normalize_kdf_profile(getattr(meta, 'kdf_profile', None))
        domain_session: DomainSeparatedSession | None = None
        if package_profile == KDF_PROFILE_V2:
            image_nonce_hex = getattr(meta, 'image_nonce', None)
            if not image_nonce_hex:
                raise ValueError(
                    "kdf_profile=v2_domain_separated requires image_nonce in metadata"
                )
            domain_session = DomainSeparatedSession(
                self.master_key, image_nonce=bytes.fromhex(str(image_nonce_hex))
            )

        # Session & Chaos Reset
        if meta.session_id != self.crypto_session.session_id.decode('utf-8'):
            self.logger.warning(f"Session ID mismatch! Re-initializing session to {meta.session_id}")
            self.crypto_session = CryptoSession(
                self.crypto_session.key_manager.master_key,
                session_id=meta.session_id.encode('utf-8')
            )

        package_backend = str(getattr(meta, 'chaos_backend', 'logistic')).strip().lower()
        chaos = (
            self.chaos
            if package_backend == self.chaos_backend
            else self._build_chaos(package_backend)
        )
        chaos.generator.reset()

        # 1. Regenerate Permutation Indices FIRST
        num_blocks = len(payload)
        if domain_session is not None:
            perm_indices = domain_session.build_permutation().generate_permutation_indices(
                num_blocks
            )
        else:
            perm_indices = chaos.generate_permutation_indices(num_blocks)

        # 2. Block Decryption & Inverse Diffusion
        payload.sort(key=lambda x: x['index'])
        
        decrypted_shuffled_blocks = []
        
        for item in payload:
            idx = item['index']
            level = item['level']
            nonce = item['nonce']
            ct = item['data']
            shape = item['shape']
            dtype = np.dtype(item['dtype'])
            
            if level == "none":
                # Raw data
                pt_bytes = ct
                block = np.frombuffer(pt_bytes, dtype=dtype).reshape(shape)
                # No diffusion inverse
                original_block = block

            else:
                # AEAD Decrypt
                session = domain_session if domain_session is not None else self.crypto_session
                try:
                    pt_bytes = session.decrypt_block(nonce, ct, level, idx)
                except Exception as e:
                    self.logger.error(f"Decryption failed at block {idx}: {e}")
                    if strict_authentication:
                        raise DecryptionAuthenticationError(
                            f"authenticated decryption failed at block {idx}"
                        ) from e
                    size = int(np.prod(shape)) * dtype.itemsize
                    pt_bytes = bytes(size)

                block = np.frombuffer(pt_bytes, dtype=dtype).reshape(shape)

                # Inverse Diffusion
                chaos_rounds = self.cfg['encryption']['levels'].get(level, {}).get('chaos_rounds', 0)
                if domain_session is not None:
                    # Per-block keystream: order-independent, unlike V1's shared
                    # stateful generator.
                    original_block = domain_session.diffuse(
                        block, chaos_rounds, block_index=idx, level=level
                    )
                else:
                    original_block = chaos.diffuse(block, chaos_rounds)

            decrypted_shuffled_blocks.append(original_block)
            
        shuffled_blocks_arr = np.array(decrypted_shuffled_blocks)
        
        # 3. Inverse Permutation
        restored_blocks = chaos.unshuffle_blocks(shuffled_blocks_arr, perm_indices)
        
        # 4. Merge & Unpad
        self.logger.info("Merging blocks...")
        split_meta = {
            "n_rows": meta.padded_shape[0] // meta.block_size,
            "n_cols": meta.padded_shape[1] // meta.block_size,
            "block_size": meta.block_size,
            "channels": meta.channels,
            "dtype": str(dtype)
        }
        
        merged_img = merge_blocks(restored_blocks, None, split_meta)
        final_img = unpad_image(merged_img, meta.padding_info)
        
        self.logger.info("Decryption completed.")
        return final_img

    def ciphertext_visual_projection(
        self,
        package: EncryptedPackage,
        image_shape: tuple,
    ) -> np.ndarray:
        """Project a real ciphertext container onto an image-shaped array.

        Image-domain metrics (PSNR, SSIM, histogram entropy, adjacent-pixel
        correlation) require an array with the original geometry, but a
        selective-encryption container is not an image: protected blocks carry
        authentication tags and weak blocks carry raw or XOR-masked pixels.

        This helper lays out, per block position, the block's *actual* protected
        bytes: raw pixels for ``none``, the XOR-diffused pixels for ``low``, and
        the leading ciphertext bytes for ``medium``/``high``. Every value shown
        therefore comes from real protected output rather than a SHA-256
        expansion of it.

        It remains a **projection**: AEAD ciphertext is longer than the block it
        replaces, so it is truncated to the block footprint. Any figure or table
        derived from this must be labelled a projection, not the ciphertext
        itself. Leakage metrics should be computed on the container bytes
        instead (plan item C2.2).

        Args:
            package: A package returned by :meth:`encrypt` or
                :meth:`encrypt_with_levels`.
            image_shape: Target shape, the original (unpadded) image shape.

        Returns:
            A ``uint8`` array of shape ``image_shape``.
        """
        meta = package.metadata
        block_size = meta.block_size
        payload = sorted(package.payload, key=lambda item: item["index"])

        # Undo the permutation so projected blocks land at their true positions.
        # This is a key-holder operation: v2 does not publish `perm_indices`, so
        # it is re-derived from the master key. An attacker without the key
        # cannot perform this step, which is exactly the point of C1.
        num_blocks = len(payload)
        if meta.perm_indices is not None:
            perm_indices = np.asarray(meta.perm_indices, dtype=np.int64)
        else:
            profile = normalize_kdf_profile(getattr(meta, "kdf_profile", None))
            if profile == KDF_PROFILE_V2 and meta.image_nonce:
                session = DomainSeparatedSession(
                    self.master_key, image_nonce=bytes.fromhex(str(meta.image_nonce))
                )
                perm_indices = np.asarray(
                    session.build_permutation().generate_permutation_indices(num_blocks),
                    dtype=np.int64,
                )
            else:
                perm_indices = np.asarray(
                    self.chaos.generate_permutation_indices(num_blocks), dtype=np.int64
                )

        rows = meta.padded_shape[0] // block_size
        cols = meta.padded_shape[1] // block_size
        channels = meta.channels
        block_footprint = block_size * block_size * channels

        projected = np.zeros((rows * cols, block_footprint), dtype=np.uint8)
        for position, item in enumerate(payload):
            raw = item["data"]
            buffer = np.frombuffer(raw[:block_footprint], dtype=np.uint8)
            if buffer.size < block_footprint:
                buffer = np.pad(buffer, (0, block_footprint - buffer.size))
            projected[position] = buffer

        # `payload[i]` holds the block that permutation moved to slot i, i.e.
        # original index perm_indices[i].
        restored = np.empty_like(projected)
        restored[perm_indices[: len(projected)]] = projected
        merged = restored.reshape(rows, cols, block_size, block_size, channels)
        merged = merged.transpose(0, 2, 1, 3, 4).reshape(
            rows * block_size, cols * block_size, channels
        )
        cropped = merged[: image_shape[0], : image_shape[1]]
        if len(image_shape) == 2:
            return cropped[:, :, 0]
        return cropped[:, :, : image_shape[2]]

    def _process_single_block(
        self,
        args: tuple,
        domain_session: DomainSeparatedSession | None = None,
    ) -> dict:
        """Process a single block for encryption (worker-safe)."""
        i, block, level = args
        if level == "none":
            ciphertext = block.tobytes()
            nonce = b""
        else:
            chaos_rounds = self.cfg['encryption']['levels'].get(level, {}).get('chaos_rounds', 0)
            if domain_session is not None:
                processed_block = domain_session.diffuse(
                    block, chaos_rounds, block_index=i, level=level
                )
                nonce, ciphertext = domain_session.encrypt_block(
                    processed_block.tobytes(), level, block_index=i
                )
            else:
                processed_block = self.chaos.diffuse(block, chaos_rounds)
                nonce, ciphertext = self.crypto_session.encrypt_block(
                    processed_block.tobytes(), level, block_index=i
                )
        return {
            'index': i,
            'level': level,
            'nonce': nonce,
            'data': ciphertext,
            'shape': block.shape,
            'dtype': str(block.dtype),
        }

    def _process_blocks(
        self,
        blocks: np.ndarray,
        levels: np.ndarray,
        num_workers: int = 0,
        domain_session: DomainSeparatedSession | None = None,
    ) -> List[dict]:
        """Process blocks for encryption, with optional parallel execution.

        Note on parallelism: the V1 path shares one stateful chaos generator, so
        mask bytes depend on scheduling order and parallel execution is only
        sound because ``parallel_workers`` defaults to 0. The v2 path derives an
        independent keystream per block and is order-independent by
        construction.
        """
        n = len(blocks)
        min_parallel = self.cfg.get('pipeline', {}).get('parallel_min_blocks', 64)
        if num_workers <= 0 or n < min_parallel:
            payload = []
            for i, (block, level) in enumerate(zip(blocks, levels)):
                payload.append(
                    self._process_single_block((i, block, level), domain_session)
                )
            return payload

        tasks = [(i, blocks[i], levels[i]) for i in range(n)]
        payload = [None] * n
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._process_single_block, t, domain_session): t[0]
                for t in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                payload[result['index']] = result
        return payload
