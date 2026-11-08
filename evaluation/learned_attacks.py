"""Learning-based attacks against the resource-aware selective encryption pipeline.

The empirical-security argument we make in the paper rests on showing that an
adaptive adversary, given white-box knowledge of the encryption framework
(scorer family, allocator, AEAD primitive, protection-level distribution) and
black-box query access to the encryption oracle, cannot recover sensitive
content beyond a quantified information-leak bound.

This module provides four attackers intended for a stronger paper-level
empirical security analysis:

1. **CNNReconstructionAttack** -- a U-Net-style conv decoder that maps
   ciphertext payload bytes to a reconstructed image. Trained per
   (method, budget) configuration.

2. **ROIDetectionAttack** -- a binary classifier that decides, given the
   ciphertext payload of a single block, whether an independently labelled
   plaintext block was sensitive. This measures whether the protection-level
   *distribution* leaks externally defined ROI location.

3. **MINEMutualInformationAttack** -- the Belghazi et al. (2018) MINE
   estimator applied to (plaintext-block, ciphertext-block) pairs. Gives a
   neural lower bound on I(M; C); it is an empirical leakage indicator, not an
   upper bound on what any adversary can recover.

4. **PerceptualSimilarityAttack** -- LPIPS distance between the plaintext
   image and the reconstructed image (output of #1). Bounded in [0, 1];
   higher means the reconstruction is perceptually closer to the plaintext.

All four are wrapped behind ``run_attack_suite(...)`` so the experiment runner
can invoke them once per (method, budget) and write a row to
``raw_attack_results.csv`` with consistent columns.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Heavy ML imports are deferred so that callers that only need the dataclass
# definitions or want to inspect the suite without GPU can import this module
# cheaply. Each attacker calls ``_lazy_import_torch()`` at instantiation.


def _lazy_import_torch():
    """Import torch on first use."""
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
    return __import__("torch"), __import__("torch.nn", fromlist=["dummy"]), __import__(
        "torch.nn.functional", fromlist=["dummy"]
    )


@dataclass
class AttackResult:
    """One row of the empirical-security attack suite output."""

    attack_name: str
    method_name: str
    budget_ratio: float
    seed: int
    attack_success_rate: Optional[float] = None
    auc: Optional[float] = None
    # `lpips` stays in the schema but is only populated when a real LPIPS network
    # is loaded, which this implementation does not do. It was previously filled
    # with a mean-absolute-error proxy under the LPIPS name, so any historical
    # value in this column is an MAE and must not be read as a perceptual
    # distance (finding F5). The MAE now has its own column, `recon_mae`.
    lpips: Optional[float] = None
    recon_mae: Optional[float] = None
    mi_lower_bound_nats: Optional[float] = None
    reconstruction_psnr_db: Optional[float] = None
    # Real structural similarity. Previously this column held `1 - MAE`, which is
    # not SSIM; it is now computed with a standard windowed SSIM implementation.
    reconstruction_ssim: Optional[float] = None
    n_train: int = 0
    n_validation: int = 0
    n_test: int = 0
    n_test_positive: int = 0
    n_test_negative: int = 0
    n_eval: int = 0
    train_time_s: float = 0.0
    eval_time_s: float = 0.0
    notes: str = ""
    status: str = "FULL"
    execution_status: str = "success"
    split_strategy: str = ""
    n_train_images: int = 0
    n_validation_images: int = 0
    n_test_images: int = 0
    n_eval_images: int = 0
    split_overlap_count: int = 0
    train_image_digest: str = ""
    validation_image_digest: str = ""
    test_image_digest: str = ""
    eval_image_digest: str = ""
    label_source: str = ""
    # Attack budget, recorded per row so a strength ladder is self-describing and
    # rows from different rungs cannot be pooled by accident.
    #
    # Two distinct things are recorded. `rung_epochs` is the budget the ladder
    # *requested* and identifies which rung a row belongs to, so all attackers of
    # one rung group together. `attack_epochs` is what the attacker actually ran,
    # which is 0 for closed-form attackers that do not train in epochs. Merging
    # the two would either misreport ridge as having been tuned for epochs, or
    # split it into a rung of its own.
    rung_epochs: int = 0
    attack_epochs: int = 0
    attack_train_pairs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class AttackConfig:
    """Hyperparameters shared across attackers."""

    # Data sizes; adjust for full-run vs. smoke
    train_pairs: int = 4096
    validation_pairs: int = 1024
    eval_pairs: int = 1024
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3

    # Model dims
    block_size: int = 16
    image_channels: int = 3
    cipher_pad_bytes: int = 32  # GCM nonce(12)+tag(16)+slack
    feature_dim: int = 256

    # MINE
    mine_epochs: int = 5
    mine_hidden: int = 512

    # Misc
    device: str = "cuda"
    seed: int = 0
    split_strategy: str = "image_disjoint"
    validation_image_fraction: float = 0.1
    eval_image_fraction: float = 0.2

    @property
    def cipher_block_bytes(self) -> int:
        """Worst-case ciphertext bytes per block (uint8 input + AEAD overhead)."""
        plain = self.block_size * self.block_size * self.image_channels
        return plain + self.cipher_pad_bytes


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


@dataclass
class AttackPair:
    """A single (plaintext, ciphertext, sensitive_label) sample."""

    image_id: str
    plaintext_block: np.ndarray  # (block_size, block_size, channels) uint8
    ciphertext_bytes: bytes  # variable length
    sensitive: Optional[bool]
    protection_level: str  # one of {high, medium, low, none}
    label_source: str
    # Zero-based index in row-major block order. Needed to condition an attack on
    # spatial position -- for instance training on centre blocks and testing on
    # border blocks -- which is impossible to express without it.
    block_index: int = -1


@dataclass(frozen=True)
class AttackSplit:
    """Deterministic image-disjoint train/validation/test split.

    The ``eval_*`` properties are backward-compatible aliases for the frozen
    test partition. Validation is used only for model selection; reported
    attack metrics are computed only on test samples.
    """

    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    train_image_ids: Tuple[str, ...]
    validation_image_ids: Tuple[str, ...]
    test_image_ids: Tuple[str, ...]
    split_strategy: str = "image_disjoint"

    @property
    def eval_indices(self) -> Tuple[int, ...]:
        """Return the frozen test indices under the legacy evaluation name."""
        return self.test_indices

    @property
    def eval_image_ids(self) -> Tuple[str, ...]:
        """Return the frozen test image IDs under the legacy evaluation name."""
        return self.test_image_ids

    @property
    def overlap_count(self) -> int:
        partitions = (
            set(self.train_image_ids),
            set(self.validation_image_ids),
            set(self.test_image_ids),
        )
        return sum(
            len(partitions[left] & partitions[right])
            for left in range(len(partitions))
            for right in range(left + 1, len(partitions))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_strategy": self.split_strategy,
            "split_protocol": "train_validation_test",
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
            "eval_indices": list(self.test_indices),
            "train_image_ids": list(self.train_image_ids),
            "validation_image_ids": list(self.validation_image_ids),
            "test_image_ids": list(self.test_image_ids),
            "eval_image_ids": list(self.test_image_ids),
            "n_train_images": len(self.train_image_ids),
            "n_validation_images": len(self.validation_image_ids),
            "n_test_images": len(self.test_image_ids),
            "n_eval_images": len(self.test_image_ids),
            "overlap_count": self.overlap_count,
            "train_image_digest": _image_id_digest(self.train_image_ids),
            "validation_image_digest": _image_id_digest(self.validation_image_ids),
            "test_image_digest": _image_id_digest(self.test_image_ids),
            "eval_image_digest": _image_id_digest(self.test_image_ids),
        }


def _image_id_digest(image_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in image_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_image_disjoint_split(
    pairs: Sequence[AttackPair],
    cfg: AttackConfig,
) -> AttackSplit:
    """Create deterministic image-disjoint train, validation, and test samples."""
    if cfg.split_strategy != "image_disjoint":
        raise ValueError("Only image_disjoint attack splitting is publication-valid.")
    image_ids = [str(getattr(pair, "image_id", "")).strip() for pair in pairs]
    if not image_ids or any(not image_id for image_id in image_ids):
        raise ValueError("Every AttackPair must include a non-empty image_id.")
    unique_images = sorted(set(image_ids))
    if len(unique_images) < 3:
        raise ValueError(
            "Image-disjoint attack evaluation requires at least three image_id groups "
            "for train, validation, and test."
        )
    if not 0.0 < float(cfg.validation_image_fraction) < 1.0:
        raise ValueError("validation_image_fraction must lie strictly between zero and one.")
    if not 0.0 < float(cfg.eval_image_fraction) < 1.0:
        raise ValueError("eval_image_fraction must lie strictly between zero and one.")
    if float(cfg.validation_image_fraction) + float(cfg.eval_image_fraction) >= 1.0:
        raise ValueError("validation and test image fractions must sum to less than one.")

    rng = np.random.default_rng(cfg.seed)
    shuffled_images = [unique_images[index] for index in rng.permutation(len(unique_images))]
    n_test_images = max(1, int(round(len(shuffled_images) * float(cfg.eval_image_fraction))))
    n_validation_images = max(
        1, int(round(len(shuffled_images) * float(cfg.validation_image_fraction)))
    )
    while n_test_images + n_validation_images >= len(shuffled_images):
        if n_test_images >= n_validation_images and n_test_images > 1:
            n_test_images -= 1
        elif n_validation_images > 1:
            n_validation_images -= 1
        else:
            break
    test_images = set(shuffled_images[:n_test_images])
    validation_images = set(
        shuffled_images[n_test_images : n_test_images + n_validation_images]
    )
    train_images = set(shuffled_images[n_test_images + n_validation_images :])
    train_indices = np.asarray(
        [index for index, image_id in enumerate(image_ids) if image_id in train_images],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for index, image_id in enumerate(image_ids) if image_id in validation_images],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, image_id in enumerate(image_ids) if image_id in test_images],
        dtype=np.int64,
    )
    train_indices = train_indices[rng.permutation(len(train_indices))]
    validation_indices = validation_indices[rng.permutation(len(validation_indices))]
    test_indices = test_indices[rng.permutation(len(test_indices))]
    if cfg.train_pairs > 0:
        train_indices = train_indices[: cfg.train_pairs]
    if cfg.validation_pairs > 0:
        validation_indices = validation_indices[: cfg.validation_pairs]
    if cfg.eval_pairs > 0:
        test_indices = test_indices[: cfg.eval_pairs]
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError(
            "Image-disjoint split produced an empty train, validation, or test partition."
        )

    selected_train_images = tuple(sorted({image_ids[index] for index in train_indices}))
    selected_validation_images = tuple(
        sorted({image_ids[index] for index in validation_indices})
    )
    selected_test_images = tuple(sorted({image_ids[index] for index in test_indices}))
    return AttackSplit(
        train_indices=tuple(int(index) for index in train_indices),
        validation_indices=tuple(int(index) for index in validation_indices),
        test_indices=tuple(int(index) for index in test_indices),
        train_image_ids=selected_train_images,
        validation_image_ids=selected_validation_images,
        test_image_ids=selected_test_images,
    )


def _split_result_fields(split: AttackSplit) -> Dict[str, Any]:
    return {
        "split_strategy": split.split_strategy,
        "n_train_images": len(split.train_image_ids),
        "n_validation_images": len(split.validation_image_ids),
        "n_test_images": len(split.test_image_ids),
        "n_eval_images": len(split.test_image_ids),
        "split_overlap_count": split.overlap_count,
        "train_image_digest": _image_id_digest(split.train_image_ids),
        "validation_image_digest": _image_id_digest(split.validation_image_ids),
        "test_image_digest": _image_id_digest(split.test_image_ids),
        "eval_image_digest": _image_id_digest(split.test_image_ids),
    }


def _mean_block_ssim(
    reference: np.ndarray, candidate: np.ndarray, max_samples: int = 512
) -> Optional[float]:
    """Mean structural similarity over reconstructed blocks.

    Replaces a ``1 - MAE`` value that was previously reported under the name
    ``reconstruction_ssim`` (finding F5). SSIM is a windowed statistic over local
    means, variances and covariance, so it is not a monotone function of MAE and
    the substitution was not a rescaling.

    Args:
        reference: Ground-truth blocks, shape ``(N, C, H, W)``, values in [0, 1].
        candidate: Reconstructed blocks, same shape.
        max_samples: Cap on blocks scored, since SSIM is far more expensive than
            MAE and the mean over several hundred 16x16 blocks is already stable.

    Returns:
        Mean SSIM, or ``None`` when scikit-image is unavailable or the blocks are
        too small for any SSIM window.
    """
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    if reference.size == 0 or reference.shape != candidate.shape:
        return None

    count = min(len(reference), max_samples)
    # Blocks are (C, H, W); SSIM wants (H, W, C).
    height, width = reference.shape[2], reference.shape[3]
    # `win_size` must be odd and no larger than the smaller spatial dimension.
    window = min(7, height if height % 2 else height - 1, width if width % 2 else width - 1)
    if window < 3:
        return None

    scores: List[float] = []
    for index in range(count):
        ref = np.transpose(reference[index], (1, 2, 0)).astype(np.float64)
        cand = np.transpose(candidate[index], (1, 2, 0)).astype(np.float64)
        try:
            scores.append(
                float(
                    structural_similarity(
                        ref,
                        cand,
                        data_range=1.0,
                        channel_axis=2,
                        win_size=window,
                    )
                )
            )
        except Exception:
            continue
    return float(np.mean(scores)) if scores else None


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Compute pairwise binary AUC with half credit for ties."""
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    positives = score_values[label_values > 0.5]
    negatives = score_values[label_values <= 0.5]
    if not len(positives) or not len(negatives):
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def _partial_attack_result(
    attack_name: str,
    method_name: str,
    budget_ratio: float,
    cfg: AttackConfig,
    notes: str,
    *,
    split: Optional[AttackSplit] = None,
    n_train: int = 0,
    n_validation: int = 0,
    n_test: int = 0,
    n_eval: int = 0,
    label_source: str = "",
) -> AttackResult:
    split_fields = _split_result_fields(split) if split is not None else {}
    return AttackResult(
        attack_name=attack_name,
        method_name=method_name,
        budget_ratio=budget_ratio,
        seed=cfg.seed,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test or n_eval,
        n_eval=n_test or n_eval,
        notes=notes,
        status="PARTIAL",
        execution_status="skipped",
        label_source=label_source,
        rung_epochs=cfg.epochs,
        attack_epochs=cfg.epochs,
        attack_train_pairs=cfg.train_pairs,
        **split_fields,
    )


def _pad_or_truncate_bytes(payload: bytes, target_len: int) -> np.ndarray:
    """Right-pad with zeros (or truncate) to ``target_len`` and return uint8."""
    arr = np.frombuffer(payload, dtype=np.uint8)
    if len(arr) >= target_len:
        return arr[:target_len].copy()
    out = np.zeros(target_len, dtype=np.uint8)
    out[: len(arr)] = arr
    return out


def stack_pairs(pairs: Sequence[AttackPair], cfg: AttackConfig) -> Dict[str, np.ndarray]:
    """Stack a list of AttackPair into model-ready tensors."""
    n = len(pairs)
    plain_arr = np.zeros((n, cfg.image_channels, cfg.block_size, cfg.block_size), dtype=np.float32)
    cipher_arr = np.zeros((n, cfg.cipher_block_bytes), dtype=np.float32)
    label = np.zeros(n, dtype=np.float32)
    for i, p in enumerate(pairs):
        block = p.plaintext_block.astype(np.float32) / 255.0
        if block.ndim == 2:
            block = np.stack([block] * cfg.image_channels, axis=-1)
        block = block.transpose(2, 0, 1)
        plain_arr[i] = block
        cipher_arr[i] = _pad_or_truncate_bytes(p.ciphertext_bytes, cfg.cipher_block_bytes).astype(np.float32) / 255.0
        label[i] = np.nan if p.sensitive is None else (1.0 if p.sensitive else 0.0)
    return {
        "plain": plain_arr,
        "cipher": cipher_arr,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Attacker 1 -- CNN reconstruction attack
# ---------------------------------------------------------------------------


def _build_reconstruction_decoder(cfg: AttackConfig):
    """Return a torch.nn.Module mapping ciphertext bytes -> reconstructed block.

    Architecture: 1D-projection MLP -> reshape to spatial -> transposed conv
    decoder. Lightweight (~1.5M params) so 10 epochs on 4k samples fits in
    a few minutes on the laptop GPU.
    """
    torch, nn, F = _lazy_import_torch()
    nn = __import__("torch.nn", fromlist=["dummy"])

    class ReconDecoder(nn.Module):
        def __init__(self, cfg: AttackConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.in_proj = nn.Sequential(
                nn.Linear(cfg.cipher_block_bytes, cfg.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.feature_dim, cfg.feature_dim * 4 * 4),
                nn.ReLU(inplace=True),
            )
            self.spatial = nn.Sequential(
                nn.ConvTranspose2d(cfg.feature_dim, 128, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, cfg.image_channels, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            h = self.in_proj(x)
            h = h.view(-1, self.cfg.feature_dim, 4, 4)
            return self.spatial(h)

    return ReconDecoder(cfg)


def train_reconstruction_attack(
    pairs: Sequence[AttackPair],
    cfg: AttackConfig,
    *,
    method_name: str,
    budget_ratio: float,
    split: Optional[AttackSplit] = None,
) -> AttackResult:
    """Train the CNN reconstruction attacker and report PSNR/SSIM/LPIPS."""
    if len(pairs) < 8:
        return _partial_attack_result(
            "cnn_reconstruction",
            method_name,
            budget_ratio,
            cfg,
            "insufficient_samples",
            n_train=len(pairs),
        )

    torch, nn, F = _lazy_import_torch()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    try:
        split = split or build_image_disjoint_split(pairs, cfg)
    except ValueError as exc:
        return _partial_attack_result(
            "cnn_reconstruction", method_name, budget_ratio, cfg, f"invalid_split: {exc}"
        )
    train_idx = np.asarray(split.train_indices, dtype=np.int64)
    validation_idx = np.asarray(split.validation_indices, dtype=np.int64)
    test_idx = np.asarray(split.test_indices, dtype=np.int64)
    n_train = len(train_idx)
    n_validation = len(validation_idx)
    n_test = len(test_idx)

    train_data = stack_pairs([pairs[i] for i in train_idx], cfg)
    validation_data = stack_pairs([pairs[i] for i in validation_idx], cfg)
    test_data = stack_pairs([pairs[i] for i in test_idx], cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = _build_reconstruction_decoder(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    plain_t = torch.from_numpy(train_data["plain"]).to(device)
    cipher_t = torch.from_numpy(train_data["cipher"]).to(device)
    validation_plain_t = torch.from_numpy(validation_data["plain"]).to(device)
    validation_cipher_t = torch.from_numpy(validation_data["cipher"]).to(device)

    t0 = time.time()
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(cfg.epochs):
        model.train()
        idx = rng.permutation(n_train)
        for start in range(0, n_train, cfg.batch_size):
            batch = idx[start : start + cfg.batch_size]
            optim.zero_grad()
            pred = model(cipher_t[batch])
            loss = F.mse_loss(pred, plain_t[batch])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            validation_loss = F.mse_loss(
                model(validation_cipher_t), validation_plain_t
            ).item()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    t0 = time.time()
    model.eval()
    with torch.no_grad():
        test_plain_t = torch.from_numpy(test_data["plain"]).to(device)
        test_cipher_t = torch.from_numpy(test_data["cipher"]).to(device)
        test_pred = model(test_cipher_t)
        # PSNR
        mse = F.mse_loss(test_pred, test_plain_t).item()
        psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        mae = float((test_pred - test_plain_t).abs().mean().item())
        ssim_value = _mean_block_ssim(
            test_plain_t.detach().cpu().numpy(), test_pred.detach().cpu().numpy()
        )
    eval_time = time.time() - t0

    return AttackResult(
        attack_name="cnn_reconstruction",
        method_name=method_name,
        budget_ratio=budget_ratio,
        seed=cfg.seed,
        reconstruction_psnr_db=psnr,
        reconstruction_ssim=ssim_value,
        recon_mae=mae,
        # Left unset: no LPIPS network is loaded, so no perceptual distance is
        # available. Populating it with an MAE under the LPIPS name is the
        # mislabeling that finding F5 identified.
        lpips=None,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        n_eval=n_test,
        train_time_s=train_time,
        eval_time_s=eval_time,
        notes=(
            "reconstruction_ssim is a real windowed SSIM; recon_mae is the mean "
            "absolute error; lpips is unset because no LPIPS network is loaded. "
            f"best_epoch={best_epoch}; final metrics use frozen test partition"
        ),
        rung_epochs=cfg.epochs,
        attack_epochs=cfg.epochs,
        attack_train_pairs=cfg.train_pairs,
        **_split_result_fields(split),
    )


# ---------------------------------------------------------------------------
# Attacker 1b -- linear (ridge) reconstruction probe
# ---------------------------------------------------------------------------


def train_linear_probe_attack(
    pairs: Sequence[AttackPair],
    cfg: AttackConfig,
    *,
    method_name: str,
    budget_ratio: float,
    split: Optional[AttackSplit] = None,
) -> AttackResult:
    """Closed-form ridge regression from ciphertext bytes to plaintext pixels.

    This is not a weaker CNN. It answers a different question: how much of the
    plaintext is recoverable by an *affine* map of the ciphertext bytes. A
    keystream reused across blocks, or a level whose payload is left in the
    clear, is exactly the kind of structure a linear map captures, and reporting
    only a CNN risks understating leakage -- the per-tile domain-separation work
    found linear probes reaching 24-30 dB on configurations where CNN SSIM was
    around 0.02. Omitting it leaves the paper open to the objection that the
    adversary was chosen to be weak.

    Ridge is solved on the normal equations rather than by gradient descent, so
    there is no learning rate, no epoch count and no seed dependence: the result
    is the exact minimiser and cannot be blamed on optimisation. The penalty is
    picked on the validation split from a fixed grid, and the reported metrics
    come from the frozen test partition, matching the CNN's protocol.
    """
    if len(pairs) < 8:
        return _partial_attack_result(
            "linear_probe_reconstruction",
            method_name,
            budget_ratio,
            cfg,
            "insufficient_samples",
            n_train=len(pairs),
        )
    try:
        split = split or build_image_disjoint_split(pairs, cfg)
    except ValueError as exc:
        return _partial_attack_result(
            "linear_probe_reconstruction",
            method_name,
            budget_ratio,
            cfg,
            f"invalid_split: {exc}",
        )

    train_idx = np.asarray(split.train_indices, dtype=np.int64)
    validation_idx = np.asarray(split.validation_indices, dtype=np.int64)
    test_idx = np.asarray(split.test_indices, dtype=np.int64)

    train_data = stack_pairs([pairs[i] for i in train_idx], cfg)
    validation_data = stack_pairs([pairs[i] for i in validation_idx], cfg)
    test_data = stack_pairs([pairs[i] for i in test_idx], cfg)

    def design(block: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Ciphertext bytes with a bias column, and flattened target pixels."""
        x = block["cipher"].astype(np.float64)
        x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
        y = block["plain"].astype(np.float64).reshape(len(x), -1)
        return x, y

    x_train, y_train = design(train_data)
    x_validation, y_validation = design(validation_data)
    x_test, y_test = design(test_data)

    t0 = time.time()
    gram = x_train.T @ x_train
    cross = x_train.T @ y_train
    identity = np.eye(gram.shape[0], dtype=np.float64)
    # Never penalise the bias column; shrinking the intercept biases the
    # reconstruction toward zero rather than toward the mean block.
    identity[-1, -1] = 0.0

    best = {"penalty": None, "loss": float("inf"), "weights": None}
    for penalty in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        try:
            weights = np.linalg.solve(gram + penalty * identity, cross)
        except np.linalg.LinAlgError:
            continue
        loss = float(np.mean((x_validation @ weights - y_validation) ** 2))
        if loss < best["loss"]:
            best = {"penalty": penalty, "loss": loss, "weights": weights}
    train_time = time.time() - t0

    if best["weights"] is None:
        return _partial_attack_result(
            "linear_probe_reconstruction",
            method_name,
            budget_ratio,
            cfg,
            "ridge_system_singular_at_every_penalty",
            split=split,
            n_train=len(train_idx),
            n_validation=len(validation_idx),
            n_test=len(test_idx),
        )

    t0 = time.time()
    prediction = np.clip(x_test @ best["weights"], 0.0, 1.0)
    mse = float(np.mean((prediction - y_test) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    mae = float(np.mean(np.abs(prediction - y_test)))
    shape = (len(x_test), cfg.image_channels, cfg.block_size, cfg.block_size)
    ssim_value = _mean_block_ssim(
        y_test.reshape(shape).astype(np.float32),
        prediction.reshape(shape).astype(np.float32),
    )
    eval_time = time.time() - t0

    return AttackResult(
        attack_name="linear_probe_reconstruction",
        method_name=method_name,
        budget_ratio=budget_ratio,
        seed=cfg.seed,
        reconstruction_psnr_db=psnr,
        reconstruction_ssim=ssim_value,
        recon_mae=mae,
        lpips=None,
        n_train=len(train_idx),
        n_validation=len(validation_idx),
        n_test=len(test_idx),
        n_eval=len(test_idx),
        train_time_s=train_time,
        eval_time_s=eval_time,
        notes=(
            "closed-form ridge from ciphertext bytes to plaintext pixels; "
            f"penalty={best['penalty']:g} selected on validation; bias column "
            "unpenalised; metrics on frozen test partition. Reported alongside "
            "the CNN because a linear map can expose reused-keystream structure "
            "that a conv decoder may miss"
        ),
        # Closed form: `attack_epochs` is 0 because ridge does not train in
        # epochs, while `rung_epochs` still records which ladder rung this row
        # belongs to so it groups with its siblings.
        rung_epochs=cfg.epochs,
        attack_epochs=0,
        attack_train_pairs=cfg.train_pairs,
        **_split_result_fields(split),
    )


# ---------------------------------------------------------------------------
# Attacker 2 -- ROI detection attack (binary classifier on ciphertext)
# ---------------------------------------------------------------------------


def train_roi_detection_attack(
    pairs: Sequence[AttackPair],
    cfg: AttackConfig,
    *,
    method_name: str,
    budget_ratio: float,
    split: Optional[AttackSplit] = None,
) -> AttackResult:
    """Train a binary classifier: given ciphertext bytes, is the block sensitive?"""
    if len(pairs) < 8:
        return _partial_attack_result(
            "roi_detection",
            method_name,
            budget_ratio,
            cfg,
            "insufficient_samples",
            n_train=len(pairs),
        )

    torch, nn, F = _lazy_import_torch()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    try:
        split = split or build_image_disjoint_split(pairs, cfg)
    except ValueError as exc:
        return _partial_attack_result(
            "roi_detection", method_name, budget_ratio, cfg, f"invalid_split: {exc}"
        )
    label_sources = {
        str(getattr(pair, "label_source", "")).strip()
        for pair in pairs
        if pair.sensitive is not None and str(getattr(pair, "label_source", "")).strip()
    }
    if not label_sources:
        return _partial_attack_result(
            "roi_detection",
            method_name,
            budget_ratio,
            cfg,
            "external_sensitive_labels_unavailable",
            split=split,
        )
    if any(pair.sensitive is None for pair in pairs):
        return _partial_attack_result(
            "roi_detection",
            method_name,
            budget_ratio,
            cfg,
            "external_sensitive_labels_incomplete",
            split=split,
        )
    if len(label_sources) != 1:
        return _partial_attack_result(
            "roi_detection",
            method_name,
            budget_ratio,
            cfg,
            "inconsistent_external_label_sources",
            split=split,
        )
    label_source = next(iter(label_sources))
    train_pairs = [pairs[index] for index in split.train_indices if pairs[index].sensitive is not None]
    validation_pairs = [
        pairs[index] for index in split.validation_indices if pairs[index].sensitive is not None
    ]
    test_pairs = [pairs[index] for index in split.test_indices if pairs[index].sensitive is not None]
    if len(train_pairs) < 2 or len(validation_pairs) < 2 or len(test_pairs) < 2:
        return _partial_attack_result(
            "roi_detection",
            method_name,
            budget_ratio,
            cfg,
            "insufficient_labeled_train_validation_test_samples",
            split=split,
            n_train=len(train_pairs),
            n_validation=len(validation_pairs),
            n_test=len(test_pairs),
            label_source=label_source,
        )

    train_data = stack_pairs(train_pairs, cfg)
    validation_data = stack_pairs(validation_pairs, cfg)
    test_data = stack_pairs(test_pairs, cfg)
    n_train = len(train_pairs)
    n_validation = len(validation_pairs)
    n_test = len(test_pairs)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    head = nn.Sequential(
        nn.Linear(cfg.cipher_block_bytes, cfg.feature_dim),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.feature_dim, cfg.feature_dim),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.feature_dim, 1),
    ).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=cfg.lr)

    cipher_t = torch.from_numpy(train_data["cipher"]).to(device)
    label_t = torch.from_numpy(train_data["label"]).to(device).unsqueeze(1)
    validation_cipher_t = torch.from_numpy(validation_data["cipher"]).to(device)
    validation_label_t = torch.from_numpy(validation_data["label"]).to(device).unsqueeze(1)

    t0 = time.time()
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    for epoch in range(cfg.epochs):
        head.train()
        idx = rng.permutation(n_train)
        for start in range(0, n_train, cfg.batch_size):
            batch = idx[start : start + cfg.batch_size]
            optim.zero_grad()
            logits = head(cipher_t[batch])
            loss = F.binary_cross_entropy_with_logits(logits, label_t[batch])
            loss.backward()
            optim.step()
        head.eval()
        with torch.no_grad():
            validation_loss = F.binary_cross_entropy_with_logits(
                head(validation_cipher_t), validation_label_t
            ).item()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
    train_time = time.time() - t0
    head.load_state_dict(best_state)

    t0 = time.time()
    head.eval()
    with torch.no_grad():
        test_cipher_t = torch.from_numpy(test_data["cipher"]).to(device)
        test_label = test_data["label"]
        n_test_positive = int(np.sum(test_label > 0.5))
        n_test_negative = int(np.sum(test_label <= 0.5))
        logits = head(test_cipher_t).cpu().numpy().reshape(-1)
        preds = (logits > 0.0).astype(np.float32)
        acc = float((preds == test_label).mean())
        auc = binary_auc(logits, test_label)
    eval_time = time.time() - t0

    return AttackResult(
        attack_name="roi_detection",
        method_name=method_name,
        budget_ratio=budget_ratio,
        seed=cfg.seed,
        attack_success_rate=acc,
        auc=auc,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        n_test_positive=n_test_positive,
        n_test_negative=n_test_negative,
        n_eval=n_test,
        train_time_s=train_time,
        eval_time_s=eval_time,
        notes=(
            f"binary head over flattened ciphertext bytes; best_epoch={best_epoch}; "
            "final metrics use frozen test partition"
            if auc is not None
            else "auc_undefined_single_class_test"
        ),
        status="FULL" if auc is not None else "PARTIAL",
        execution_status="success",
        label_source=label_source,
        rung_epochs=cfg.epochs,
        attack_epochs=cfg.epochs,
        attack_train_pairs=cfg.train_pairs,
        **_split_result_fields(split),
    )


# ---------------------------------------------------------------------------
# Attacker 3 -- MINE mutual information lower bound (Belghazi et al. 2018)
# ---------------------------------------------------------------------------


def estimate_mine_lower_bound(
    pairs: Sequence[AttackPair],
    cfg: AttackConfig,
    *,
    method_name: str,
    budget_ratio: float,
    split: Optional[AttackSplit] = None,
) -> AttackResult:
    """Train a small MINE network to lower-bound I(plaintext; ciphertext)."""
    if len(pairs) < 8:
        return _partial_attack_result(
            "mine_mi_bound",
            method_name,
            budget_ratio,
            cfg,
            "insufficient_samples",
            n_train=len(pairs),
        )

    torch, nn, F = _lazy_import_torch()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    data = stack_pairs(pairs, cfg)
    plain_flat = data["plain"].reshape(len(pairs), -1)
    cipher_flat = data["cipher"]
    try:
        split = split or build_image_disjoint_split(pairs, cfg)
    except ValueError as exc:
        return _partial_attack_result(
            "mine_mi_bound", method_name, budget_ratio, cfg, f"invalid_split: {exc}"
        )
    train_idx = np.asarray(split.train_indices, dtype=np.int64)
    validation_idx = np.asarray(split.validation_indices, dtype=np.int64)
    test_idx = np.asarray(split.test_indices, dtype=np.int64)
    n_train = len(train_idx)
    n_validation = len(validation_idx)
    n_test = len(test_idx)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    in_dim = plain_flat.shape[1] + cipher_flat.shape[1]

    statistics = nn.Sequential(
        nn.Linear(in_dim, cfg.mine_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.mine_hidden, cfg.mine_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.mine_hidden, 1),
    ).to(device)
    optim = torch.optim.Adam(statistics.parameters(), lr=cfg.lr)

    p_t = torch.from_numpy(plain_flat[train_idx]).to(device)
    c_t = torch.from_numpy(cipher_flat[train_idx]).to(device)
    vp_t = torch.from_numpy(plain_flat[validation_idx]).to(device)
    vc_t = torch.from_numpy(cipher_flat[validation_idx]).to(device)

    t0 = time.time()
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in statistics.state_dict().items()
    }
    for epoch in range(cfg.mine_epochs):
        statistics.train()
        idx = rng.permutation(n_train)
        for start in range(0, n_train, cfg.batch_size):
            batch = idx[start : start + cfg.batch_size]
            shuffled = batch[torch.randperm(len(batch)).numpy()]
            joint = torch.cat([p_t[batch], c_t[batch]], dim=1)
            marginal = torch.cat([p_t[batch], c_t[shuffled]], dim=1)
            t_joint = statistics(joint)
            t_marginal = statistics(marginal)
            mine_lb = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
            loss = -mine_lb
            optim.zero_grad()
            loss.backward()
            optim.step()
        statistics.eval()
        with torch.no_grad():
            validation_order = torch.randperm(len(validation_idx), device=device)
            validation_joint = torch.cat([vp_t, vc_t], dim=1)
            validation_marginal = torch.cat([vp_t, vc_t[validation_order]], dim=1)
            validation_bound = statistics(validation_joint).mean() - torch.log(
                torch.exp(statistics(validation_marginal)).mean() + 1e-8
            )
            validation_loss = -float(validation_bound.item())
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in statistics.state_dict().items()
            }
    train_time = time.time() - t0
    statistics.load_state_dict(best_state)

    t0 = time.time()
    statistics.eval()
    with torch.no_grad():
        tp_t = torch.from_numpy(plain_flat[test_idx]).to(device)
        tc_t = torch.from_numpy(cipher_flat[test_idx]).to(device)
        joint = torch.cat([tp_t, tc_t], dim=1)
        shuffled_idx = torch.randperm(len(test_idx), device=device)
        marginal = torch.cat([tp_t, tc_t[shuffled_idx]], dim=1)
        t_joint = statistics(joint)
        t_marginal = statistics(marginal)
        mi_estimate = float(t_joint.mean().item() - torch.log(torch.exp(t_marginal).mean() + 1e-8).item())
    eval_time = time.time() - t0

    return AttackResult(
        attack_name="mine_mi_bound",
        method_name=method_name,
        budget_ratio=budget_ratio,
        seed=cfg.seed,
        mi_lower_bound_nats=mi_estimate,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        n_eval=n_test,
        train_time_s=train_time,
        eval_time_s=eval_time,
        notes=(
            "Belghazi 2018 MINE-DV lower bound; nats; lower=less leakage; "
            f"best_epoch={best_epoch}; final metric uses frozen test partition"
        ),
        rung_epochs=cfg.epochs,
        attack_epochs=cfg.mine_epochs,
        attack_train_pairs=cfg.train_pairs,
        **_split_result_fields(split),
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_attack_suite(
    pairs: Sequence[AttackPair],
    cfg: AttackConfig,
    *,
    method_name: str,
    budget_ratio: float,
    skip: Iterable[str] = (),
    split: Optional[AttackSplit] = None,
) -> List[AttackResult]:
    """Run all enabled attackers on a single (method, budget) configuration."""
    skip_set = set(skip)
    results: List[AttackResult] = []
    try:
        split = split or build_image_disjoint_split(pairs, cfg)
    except ValueError as exc:
        for attack_name in (
            "cnn_reconstruction",
            "linear_probe_reconstruction",
            "roi_detection",
            "mine_mi_bound",
        ):
            if attack_name not in skip_set:
                results.append(
                    _partial_attack_result(
                        attack_name,
                        method_name,
                        budget_ratio,
                        cfg,
                        f"invalid_split: {exc}",
                    )
                )
        return results
    if "cnn_reconstruction" not in skip_set:
        results.append(
            train_reconstruction_attack(
                pairs,
                cfg,
                method_name=method_name,
                budget_ratio=budget_ratio,
                split=split,
            )
        )
    if "linear_probe_reconstruction" not in skip_set:
        results.append(
            train_linear_probe_attack(
                pairs,
                cfg,
                method_name=method_name,
                budget_ratio=budget_ratio,
                split=split,
            )
        )
    if "roi_detection" not in skip_set:
        results.append(
            train_roi_detection_attack(
                pairs,
                cfg,
                method_name=method_name,
                budget_ratio=budget_ratio,
                split=split,
            )
        )
    if "mine_mi_bound" not in skip_set:
        results.append(
            estimate_mine_lower_bound(
                pairs,
                cfg,
                method_name=method_name,
                budget_ratio=budget_ratio,
                split=split,
            )
        )
    return results


def write_attack_results_csv(
    output_path: str | Path,
    results: Sequence[AttackResult],
) -> Path:
    """Write a list of AttackResult rows to a canonical CSV."""
    import csv

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(AttackResult.__dataclass_fields__.keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
    return target
