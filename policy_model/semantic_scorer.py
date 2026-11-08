from __future__ import annotations

import contextlib
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from src.policy_model.backend_types import (
    is_real_semantic_backend,
    normalize_semantic_backend,
)


def _normalize_array(values: np.ndarray) -> np.ndarray:
    """Normalize a 1D array into the stable range [0, 1]."""
    if values.size == 0:
        return values.astype(np.float32)

    values = values.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _safe_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize a matrix row-wise while avoiding division by zero."""
    if matrix.size == 0:
        return matrix.astype(np.float32)

    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return matrix / norms


def _normalize_device_request(device_name: Optional[str]) -> str:
    """Normalize supported device requests while defaulting to auto."""
    if isinstance(device_name, str):
        normalized = device_name.strip().lower()
        if normalized in {"auto", "cpu", "cuda"}:
            return normalized
        if normalized:
            raise ValueError(f"Unsupported device request: {device_name}")
    return "auto"


def resolve_runtime_device_info(
    device_name: Optional[str],
    *,
    strict_cuda: bool = False,
) -> Dict[str, Any]:
    """Resolve runtime device metadata for semantic inference."""
    requested = _normalize_device_request(device_name)
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "NO GPU"

    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif requested == "cuda":
        if strict_cuda and not cuda_available:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        resolved = "cuda"
    else:
        resolved = "cpu"

    return {
        "device_requested": requested,
        "device_resolved": resolved,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def _resolve_device(device_name: Optional[str]) -> str:
    """Resolve the preferred runtime device for the semantic model."""
    return str(resolve_runtime_device_info(device_name)["device_resolved"])


def resolve_semantic_runtime_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve semantic runtime config with explicit precedence and traceability."""
    semantic_cfg = dict(config or {})
    cli_model_path = semantic_cfg.get("cli_model_path") or None
    env_model_path = os.getenv("DINOv2_MODEL_PATH") or None
    config_model_path = semantic_cfg.get("model_path") or semantic_cfg.get("local_model_path") or None
    resolved_model_path = cli_model_path or env_model_path or config_model_path
    device_info = resolve_runtime_device_info(semantic_cfg.get("device"))
    resolved = dict(semantic_cfg)
    resolved["model_path_resolved"] = resolved_model_path
    resolved["device_requested"] = device_info["device_requested"]
    resolved["device_resolved"] = device_info["device_resolved"]
    resolved["cuda_available"] = device_info["cuda_available"]
    resolved["gpu_name"] = device_info["gpu_name"]
    resolved["local_model_path"] = resolved_model_path
    return resolved


def _load_huggingface_dinov2() -> Tuple[Any, Any]:
    """
    Load the Hugging Face classes used by the DINOv2 adapter.

    This is isolated to keep fallback handling and unit test mocking simple.
    """
    try:
        import transformers
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils

        # AutoModel may import generation helpers that probe optional sklearn.
        # The DINOv2 image encoder does not use sklearn, so keep broken optional
        # sklearn installs from blocking real semantic inference.
        # Save and restore original values to avoid global side effects.
        _orig_sklearn = import_utils.is_sklearn_available
        _orig_sklearn_utils = transformers_utils.is_sklearn_available
        try:
            import_utils.is_sklearn_available = lambda: False
            transformers_utils.is_sklearn_available = lambda: False
            from transformers import AutoImageProcessor, AutoModel
        finally:
            import_utils.is_sklearn_available = _orig_sklearn
            transformers_utils.is_sklearn_available = _orig_sklearn_utils
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed; unable to initialize the DINOv2 backend."
        ) from exc
    return AutoImageProcessor, AutoModel


def _safe_from_pretrained(
    loader: Any,
    load_path: str,
    revision: Optional[str] = None,
) -> Any:
    """Load a Hugging Face asset while suppressing tqdm stderr issues.

    Some non-interactive runners expose a stderr stream that raises
    ``OSError: [Errno 22] Invalid argument`` when tqdm tries to flush status
    updates during model materialization. Redirecting stdout/stderr to in-memory
    buffers keeps model loading deterministic without changing model weights or
    inference behavior.
    """
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        kwargs = {"revision": revision} if revision and not os.path.isdir(load_path) else {}
        return loader.from_pretrained(load_path, **kwargs)


def check_dinov2_backend_availability(
    model_name: str,
    local_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify that the real DINOv2 Hugging Face stack can be constructed.

    This helper is intentionally lightweight and is used by diagnosis code to
    fail early before running semantic experiments that must not fall back.
    """
    status: Dict[str, Any] = {
        "transformers_import_ok": False,
        "processor_load_ok": False,
        "model_load_ok": False,
        "model_name": model_name,
        "error": "",
    }
    try:
        auto_processor, auto_model = _load_huggingface_dinov2()
        status["transformers_import_ok"] = True
        load_path = local_model_path if local_model_path else model_name
        auto_processor.from_pretrained(load_path)
        status["processor_load_ok"] = True
        auto_model.from_pretrained(load_path)
        status["model_load_ok"] = True
    except Exception as exc:
        status["error"] = str(exc)
    return status


def _block_to_pil(block: np.ndarray) -> Image.Image:
    """Convert a single block to an RGB PIL image."""
    if block.ndim == 2:
        array = np.stack([block, block, block], axis=2)
    elif block.ndim == 3 and block.shape[2] == 1:
        array = np.repeat(block, 3, axis=2)
    else:
        array = block

    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


@dataclass(frozen=True)
class SemanticBackendInfo:
    """Describes the active semantic backend."""

    semantic_mode: str
    backend: str
    model_name: str
    active_backend: str
    fallback_used: bool
    reason: str
    model_path_resolved: Optional[str] = None
    device_requested: Optional[str] = None
    device_resolved: Optional[str] = None
    cuda_available: Optional[bool] = None
    gpu_name: Optional[str] = None


class SemanticEmbeddingAdapter(ABC):
    """Abstract interface for semantic embedding backends."""

    @abstractmethod
    def encode_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """Encode blocks into an embedding space."""

    @property
    @abstractmethod
    def backend_info(self) -> SemanticBackendInfo:
        """Describe the adapter backend."""

    def supports_image_patches(self) -> bool:
        """Whether this adapter implements :meth:`encode_image_patches`."""
        return False

    def encode_image_patches(
        self, image: np.ndarray, grid_rows: int, grid_cols: int
    ) -> np.ndarray:
        """Encode a whole image once and aggregate patch tokens to a block grid.

        Raises:
            NotImplementedError: adapters that cannot do this must say so via
                :meth:`supports_image_patches` and let the caller fall back.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support whole-image patch encoding"
        )


class FallbackSemanticAdapter(SemanticEmbeddingAdapter):
    """
    Deterministic fallback backend for structure validation.

    This adapter is only used when the real DINOv2 stack is unavailable or
    explicitly disabled. It preserves the semantic scorer interface so tests
    and scripts remain runnable in constrained environments.
    """

    def __init__(
        self,
        semantic_mode: str,
        model_name: str,
        reason: str,
        model_path_resolved: Optional[str] = None,
        device_requested: Optional[str] = None,
        device_resolved: Optional[str] = None,
    ):
        self._backend_info = SemanticBackendInfo(
            semantic_mode=semantic_mode,
            backend="fallback",
            model_name=model_name,
            active_backend="fallback_visual_embedding",
            fallback_used=True,
            reason=reason,
            model_path_resolved=model_path_resolved,
            device_requested=device_requested,
            device_resolved=device_resolved,
        )

    @property
    def backend_info(self) -> SemanticBackendInfo:
        return self._backend_info

    def encode_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """Create deterministic visual embeddings from simple block descriptors."""
        if len(blocks) == 0:
            return np.zeros((0, 16), dtype=np.float32)

        embeddings: List[np.ndarray] = []
        for block in blocks:
            block_float = block.astype(np.float32)
            if block_float.max() > 1.0:
                block_float = block_float / 255.0

            if block_float.ndim == 2:
                gray = block_float
                channel_mean = np.array([float(np.mean(gray))], dtype=np.float32)
                channel_std = np.array([float(np.std(gray))], dtype=np.float32)
            else:
                gray = np.mean(block_float, axis=2)
                channel_mean = np.mean(block_float, axis=(0, 1)).astype(np.float32)
                channel_std = np.std(block_float, axis=(0, 1)).astype(np.float32)

            histogram, _ = np.histogram(gray, bins=8, range=(0.0, 1.0), density=True)
            grad_y, grad_x = np.gradient(gray)
            edge_energy = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
            contrast = float(np.std(gray))

            embedding = np.concatenate(
                [
                    channel_mean.reshape(-1),
                    channel_std.reshape(-1),
                    histogram.astype(np.float32),
                    np.array([edge_energy, contrast], dtype=np.float32),
                ]
            )
            embeddings.append(embedding)

        return np.asarray(embeddings, dtype=np.float32)


class DINOv2HuggingFaceAdapter(SemanticEmbeddingAdapter):
    """
    DINOv2 semantic adapter backed by Hugging Face Transformers.

    The adapter uses `AutoImageProcessor` and `AutoModel` so the exact DINOv2
    checkpoint remains configurable. The default target is
    `facebook/dinov2-small`, which is a practical fit for one-computer
    development.
    """

    def __init__(
        self,
        model_name: str,
        device_requested: str,
        use_pretrained: bool,
        batch_size: int,
        use_fp16: bool = False,
        model_revision: Optional[str] = None,
        local_model_path: Optional[str] = None,
    ):
        if not use_pretrained:
            raise RuntimeError("DINOv2 backend requires `use_pretrained=True`.")

        auto_processor, auto_model = _load_huggingface_dinov2()
        device_info = resolve_runtime_device_info(device_requested, strict_cuda=True)

        self.device = torch.device(device_info["device_resolved"])
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = bool(use_fp16)
        if self.use_fp16 and self.device.type != "cuda":
            raise ValueError("FP16 autocast is supported only for CUDA inference.")
        self.model_name = model_name
        self.model_revision = model_revision
        self.local_model_path = local_model_path
        load_path = local_model_path if local_model_path else model_name
        self.processor = _safe_from_pretrained(auto_processor, load_path, model_revision)
        self.model = _safe_from_pretrained(auto_model, load_path, model_revision).to(self.device)
        self.model.eval()

        self._backend_info = SemanticBackendInfo(
            semantic_mode="dinov2",
            backend="huggingface",
            model_name=model_name,
            active_backend="dinov2_huggingface",
            fallback_used=False,
            reason="Using Hugging Face DINOv2 embeddings.",
            model_path_resolved=load_path,
            device_requested=device_requested,
            device_resolved=device_info["device_resolved"],
            cuda_available=device_info["cuda_available"],
            gpu_name=device_info["gpu_name"],
        )

    @property
    def backend_info(self) -> SemanticBackendInfo:
        return self._backend_info

    def _autocast_context(self) -> Any:
        """Apply FP16 only to the DINOv2 forward pass when configured."""
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def encode_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """Encode blocks with DINOv2 and return one embedding per block."""
        if len(blocks) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        pil_blocks = [_block_to_pil(block) for block in blocks]
        embedding_batches: List[torch.Tensor] = []

        # Keep each batch embedding on the GPU; concatenate and transfer once.
        # Autocast reduces DINOv2 activation memory and compute on CUDA while
        # the downstream semantic scoring remains in float32.
        with torch.inference_mode(), self._autocast_context():
            for start in range(0, len(pil_blocks), self.batch_size):
                batch_images = pil_blocks[start : start + self.batch_size]
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {
                    key: value.to(self.device) if callable(getattr(value, "to", None)) else value
                    for key, value in inputs.items()
                }

                outputs = self.model(**inputs)
                if getattr(outputs, "pooler_output", None) is not None:
                    embeddings = outputs.pooler_output
                elif getattr(outputs, "last_hidden_state", None) is not None:
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError("DINOv2 model output does not expose embeddings.")

                embedding_batches.append(embeddings.detach())

        return (
            torch.cat(embedding_batches, dim=0)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def supports_image_patches(self) -> bool:
        """DINOv2 exposes spatial patch tokens, so this path is available."""
        return True

    def encode_image_patches(
        self, image: np.ndarray, grid_rows: int, grid_cols: int
    ) -> np.ndarray:
        """Encode the whole image once, then aggregate patch tokens per block.

        Motivation (plan item S15/C9). ``encode_blocks`` runs one ViT forward per
        16x16 block, each upscaled to 224x224 -- 1536 forwards for a 768x512
        image. This method runs a single forward over the whole image and
        redistributes the resulting spatial patch tokens onto the block grid,
        which is also closer to the "global semantic deviation" the score is
        meant to express.

        Important resolution caveat
        ---------------------------
        The processor resizes to 224x224 and DINOv2 uses patch 14, giving a
        fixed 16x16 = 256-token spatial grid. Whenever the block grid is finer
        than 16x16 -- which it is for every image larger than 256x256 at
        ``block_size=16`` -- several blocks necessarily read from the same
        patch neighbourhood. This path is therefore a **lower-resolution**
        semantic signal, not an equivalent reformulation of ``encode_blocks``.
        Callers must validate rank agreement before substituting it (the
        improvement plan requires score correlation > 0.9).

        Args:
            image: Whole image, ``(H, W, C)`` or ``(H, W)``.
            grid_rows: Block rows in the target grid.
            grid_cols: Block columns in the target grid.

        Returns:
            ``(grid_rows * grid_cols, embed_dim)`` float32 embeddings in
            row-major block order, matching ``split_into_blocks``.
        """
        if grid_rows <= 0 or grid_cols <= 0:
            return np.zeros((0, 1), dtype=np.float32)

        pil_image = _block_to_pil(image)
        with torch.inference_mode():
            inputs = self.processor(images=[pil_image], return_tensors="pt")
            inputs = {
                key: value.to(self.device) if callable(getattr(value, "to", None)) else value
                for key, value in inputs.items()
            }
            outputs = self.model(**inputs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("DINOv2 model output does not expose patch tokens.")

            # Drop the leading CLS token; the remainder is the spatial grid.
            patches = hidden[:, 1:, :]
            token_count = patches.shape[1]
            side = int(round(float(token_count) ** 0.5))
            if side * side != token_count:
                raise RuntimeError(
                    f"patch token count {token_count} is not a square grid; "
                    "cannot map tokens onto the block grid"
                )
            embed_dim = patches.shape[2]
            # (1, S*S, D) -> (1, D, S, S) for spatial interpolation.
            spatial = patches.reshape(1, side, side, embed_dim).permute(0, 3, 1, 2)
            # Bilinear resampling onto the block grid. `align_corners=False`
            # treats tokens as area samples, which matches how blocks tile the
            # image, and avoids the half-cell shift `align_corners=True` adds.
            resampled = torch.nn.functional.interpolate(
                spatial.float(),
                size=(grid_rows, grid_cols),
                mode="bilinear",
                align_corners=False,
            )
            # (1, D, R, C) -> (R*C, D) in row-major block order.
            block_embeddings = (
                resampled.squeeze(0).permute(1, 2, 0).reshape(grid_rows * grid_cols, embed_dim)
            )
            return block_embeddings.detach().cpu().numpy().astype(np.float32)


class SemanticScorer:
    """
    Block-level semantic scorer for the V2 semantic scorer extension.

    Scoring formula:
    1. Extract one DINOv2 embedding per block.
    2. Form a global semantic context by averaging block embeddings.
    3. Compute `global_contrast = 1 - cosine_similarity(block, global_context)`.
       Blocks that deviate more from the image-level context are treated as
       semantically salient.
    4. Compute `embedding_norm_score` from the raw embedding L2 norm.
       This provides a weak saliency-like signal that complements the contrast term.
    5. Fuse the two semantic signals and normalize the result to [0, 1].

    The semantic fusion path then combines this semantic score with the
    existing rule-based score inside `Predictor`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = resolve_semantic_runtime_config(config)
        self.semantic_mode = str(self.config.get("semantic_mode", self.config.get("mode", "dinov2")))
        self.backend = str(self.config.get("backend", "huggingface"))
        self.model_name = str(self.config.get("model_name", "facebook/dinov2-small"))
        self.device_requested = str(self.config.get("device_requested", self.config.get("device", "auto")))
        self.device = self.config["device_resolved"]
        self.use_pretrained = bool(self.config.get("use_pretrained", True))
        self.fallback_enabled = bool(self.config.get("fallback_enabled", True))
        self.require_real_backend = bool(self.config.get("require_real_backend", False))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.model_revision = self.config.get("model_revision")
        self.contrast_weight = float(self.config.get("contrast_weight", 0.8))
        self.norm_weight = float(self.config.get("norm_weight", 0.2))
        self.local_model_path = self.config.get("model_path_resolved") or None
        # Plan item C9. `per_block` is the frozen default so existing results stay
        # reproducible; `image_patches` runs one forward per image instead of one
        # per block. The two are not equivalent (see
        # `encode_image_patches`), so switching is an explicit opt-in.
        self.semantic_encode_mode = str(
            self.config.get("semantic_encode_mode", "per_block")
        ).strip().lower()
        if self.semantic_encode_mode not in {"per_block", "image_patches"}:
            raise ValueError(
                "semantic_encode_mode must be 'per_block' or 'image_patches', "
                f"got {self.semantic_encode_mode!r}"
            )
        self.adapter = self._build_adapter()

    def get_backend_metadata(self) -> Dict[str, Any]:
        """Expose semantic backend metadata for diagnostics and tests."""
        backend_info = self.adapter.backend_info
        return {
            "semantic_mode": backend_info.semantic_mode,
            "backend": backend_info.backend,
            "model_name": backend_info.model_name,
            "active_backend": backend_info.active_backend,
            "fallback_used": backend_info.fallback_used,
            "reason": backend_info.reason,
            "require_real_backend": self.require_real_backend,
            "model_path_resolved": backend_info.model_path_resolved,
            "device_requested": backend_info.device_requested,
            "device_resolved": backend_info.device_resolved,
            "cuda_available": backend_info.cuda_available,
            "gpu_name": backend_info.gpu_name,
            "fallback_reason": backend_info.reason if backend_info.fallback_used else "",
        }

    def _build_adapter(self) -> SemanticEmbeddingAdapter:
        """Instantiate the requested semantic adapter."""
        if self.semantic_mode == "fallback":
            if self.require_real_backend:
                raise RuntimeError(
                    "Real DINOv2 backend is required, but semantic_mode='fallback' was requested."
                )
            return FallbackSemanticAdapter(
                semantic_mode="fallback",
                model_name=self.model_name,
                reason="Explicit fallback mode requested for structure validation.",
                model_path_resolved=self.local_model_path,
                device_requested=self.device_requested,
                device_resolved=self.device,
            )

        if self.semantic_mode != "dinov2":
            if self.require_real_backend:
                raise RuntimeError(
                    f"Real DINOv2 backend is required, but unsupported semantic mode `{self.semantic_mode}` was configured."
                )
            if not self.fallback_enabled:
                raise ValueError(f"Unsupported semantic mode: {self.semantic_mode}")
            return FallbackSemanticAdapter(
                semantic_mode=self.semantic_mode,
                model_name=self.model_name,
                reason=f"Unsupported semantic mode `{self.semantic_mode}`; using fallback backend.",
                model_path_resolved=self.local_model_path,
                device_requested=self.device_requested,
                device_resolved=self.device,
            )

        if self.backend != "huggingface":
            if self.require_real_backend:
                raise RuntimeError(
                    f"Real DINOv2 backend is required, but backend `{self.backend}` is not supported."
                )
            if not self.fallback_enabled:
                raise ValueError(f"Unsupported DINOv2 backend: {self.backend}")
            return FallbackSemanticAdapter(
                semantic_mode="dinov2",
                model_name=self.model_name,
                reason=f"Unsupported DINOv2 backend `{self.backend}`; using fallback backend.",
                model_path_resolved=self.local_model_path,
                device_requested=self.device_requested,
                device_resolved=self.device,
            )

        try:
            return DINOv2HuggingFaceAdapter(
                model_name=self.model_name,
                device_requested=self.device_requested,
                use_pretrained=self.use_pretrained,
                batch_size=self.batch_size,
                use_fp16=bool(self.config.get("use_fp16", False)),
                model_revision=self.model_revision,
                local_model_path=self.local_model_path,
            )
        except Exception as exc:
            if self.require_real_backend:
                raise RuntimeError(
                    f"Real DINOv2 backend is required, but initialization failed for `{self.model_name}` at path `{self.local_model_path}` on device `{self.device}`: {exc}"
                ) from exc
            if not self.fallback_enabled:
                raise
            return FallbackSemanticAdapter(
                semantic_mode="dinov2",
                model_name=self.model_name,
                reason=f"DINOv2 initialization failed: {exc}",
                model_path_resolved=self.local_model_path,
                device_requested=self.device_requested,
                device_resolved=self.device,
            )

    def _encode(
        self,
        blocks: np.ndarray,
        image: Optional[np.ndarray],
        grid_shape: Optional[Tuple[int, int]],
    ) -> Tuple[np.ndarray, str]:
        """Encode blocks via the configured mode, falling back when unavailable.

        Returns:
            ``(embeddings, encode_mode_used)``. The second element records what
            actually ran, which may differ from the request when the adapter or
            the caller cannot supply what ``image_patches`` needs.
        """
        if self.semantic_encode_mode == "image_patches":
            if image is None or grid_shape is None:
                # The caller did not supply the whole image, so the single-forward
                # path is impossible. Fall back rather than fabricate a score.
                return self.adapter.encode_blocks(blocks), "per_block_fallback_no_image"
            if not self.adapter.supports_image_patches():
                return self.adapter.encode_blocks(blocks), "per_block_fallback_unsupported"
            rows, cols = grid_shape
            return (
                self.adapter.encode_image_patches(image, rows, cols),
                "image_patches",
            )
        return self.adapter.encode_blocks(blocks), "per_block"

    def calculate_scores(
        self,
        blocks: np.ndarray,
        return_dict: bool = False,
        image: Optional[np.ndarray] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Calculate semantic scores for image blocks.

        Args:
            blocks: Blocks with shape (N, H, W, C) or (N, H, W).
            return_dict: When True, include semantic diagnostics.
            image: Whole (padded) image. Required only by
                ``semantic_encode_mode="image_patches"``; ignored otherwise.
            grid_shape: ``(block_rows, block_cols)`` for the block grid, also
                only needed by the ``image_patches`` mode.

        Returns:
            A score array or a detail dictionary with scores and backend info.
        """
        if len(blocks) == 0:
            empty_scores = np.array([], dtype=np.float32)
            if return_dict:
                backend_info = self.get_backend_metadata()
                return {
                    "scores": empty_scores,
                    "semantic_score": empty_scores,
                    "global_contrast_score": empty_scores,
                    "embedding_norm_score": empty_scores,
                    "backend": backend_info["active_backend"],
                    "requested_backend": backend_info["backend"],
                    "model_name": backend_info["model_name"],
                    "fallback_used": backend_info["fallback_used"],
                    "backend_reason": backend_info["reason"],
                    "model_path_resolved": backend_info["model_path_resolved"],
                    "device_requested": backend_info["device_requested"],
                    "device_resolved": backend_info["device_resolved"],
                    "cuda_available": backend_info["cuda_available"],
                    "gpu_name": backend_info["gpu_name"],
                    "fallback_reason": backend_info["fallback_reason"],
                }
            return empty_scores

        embeddings, encode_mode_used = self._encode(blocks, image, grid_shape)
        normalized_embeddings = _safe_l2_normalize(embeddings)

        global_embedding = np.mean(normalized_embeddings, axis=0, keepdims=True)
        global_embedding = _safe_l2_normalize(global_embedding)[0]

        # Avoid initializing a second BLAS/OpenMP runtime after Torch inference.
        cosine_similarity = np.clip(
            np.sum(normalized_embeddings * global_embedding, axis=1), -1.0, 1.0
        )
        global_contrast_score = _normalize_array(1.0 - cosine_similarity)

        embedding_norms = np.linalg.norm(embeddings, axis=1)
        embedding_norm_score = _normalize_array(embedding_norms)

        semantic_score = _normalize_array(
            self.contrast_weight * global_contrast_score
            + self.norm_weight * embedding_norm_score
        )

        if return_dict:
            backend_info = self.get_backend_metadata()
            return {
                "scores": semantic_score,
                "semantic_score": semantic_score,
                "global_contrast_score": global_contrast_score,
                "embedding_norm_score": embedding_norm_score,
                "semantic_encode_mode": encode_mode_used,
                "backend": backend_info["active_backend"],
                "requested_backend": backend_info["backend"],
                "model_name": backend_info["model_name"],
                "fallback_used": backend_info["fallback_used"],
                "backend_reason": backend_info["reason"],
                "model_path_resolved": backend_info["model_path_resolved"],
                "device_requested": backend_info["device_requested"],
                "device_resolved": backend_info["device_resolved"],
                "cuda_available": backend_info["cuda_available"],
                "gpu_name": backend_info["gpu_name"],
                "fallback_reason": backend_info["fallback_reason"],
            }
        return semantic_score
