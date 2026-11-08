import numpy as np
from collections.abc import Mapping
from typing import List, Dict, Any, Union
from .leakage_aware_scorer import build_risk_signals
from .rule_based import RuleBasedScorer
from .scorer_registry import build_scorer
from .semantic_scorer import SemanticScorer, resolve_semantic_runtime_config
from .vit_scorer import ViTScorer
from src.preprocess.features import extract_features_batch

# Lazy import for torch to avoid overhead if not used? 
# For now, standard import is fine.
from .cnn_model import CNNScorer, CNNClassifierScorer

class Predictor:
    """
    Unified interface for importance prediction.
    Supports rule-based V1 scoring and additive extension modes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.policy_config = self.config.get("policy", {})
        
        # Initialize scorers lazily or eagerly?
        # Eagerly is safer for now.
        self.rule_scorer = RuleBasedScorer(self.policy_config)
        self.semantic_scorer = None
        self.vit_scorer = None
        
        scorer_mode = self.policy_config.get("scorer", "rule_based")
        
        # Only init CNN if configured, to save resources
        if scorer_mode in ["cnn", "cnn_classifier", "fusion"]:
            # Pass relevant config to CNN
            cnn_cfg = {
                "block_size": self.config.get("image", {}).get("block_size", 16),
                "channels": self.config.get("image", {}).get("channels", 3),
                # Backward-compatible keys
                "model_path": self.policy_config.get("model_path"),
                # Extended CNN config surface
                "checkpoint_path": self.policy_config.get("checkpoint_path", self.policy_config.get("model_path")),
                "use_pretrained": self.policy_config.get("use_pretrained", False),
                "input_channels": self.policy_config.get("input_channels", self.config.get("image", {}).get("channels", 3)),
                "output_dim": self.policy_config.get("output_dim", 1),
                "device": self.policy_config.get("device")
            }
            if scorer_mode == "cnn":
                self.cnn_scorer = CNNScorer(cnn_cfg)
                self.cnn_classifier_scorer = None
            else:
                self.cnn_classifier_scorer = CNNClassifierScorer(cnn_cfg)
                self.cnn_scorer = None
        else:
            self.cnn_scorer = None
            self.cnn_classifier_scorer = None
        
        # Initialize ViT scorer if needed
        if scorer_mode in ["vit", "fusion_vit"]:
            vit_cfg = {
                "model_name": self.policy_config.get("vit_model_name", "vit_small_patch16_224"),
                "use_pretrained": self.policy_config.get("use_pretrained", True),
                "device": self.policy_config.get("device")
            }
            self.vit_scorer = ViTScorer(vit_cfg)
        else:
            self.vit_scorer = None

    def _build_semantic_config(self) -> Dict[str, Any]:
        """
        Build semantic scorer config while preserving backward compatibility.

        Nested `policy.semantic` values take precedence, but top-level semantic
        keys in `policy` are also supported for convenience.
        """
        semantic_cfg = dict(self.policy_config.get("semantic", {}))
        defaults = {
            "semantic_mode": self.policy_config.get("semantic_mode", semantic_cfg.get("semantic_mode", "dinov2")),
            "model_name": self.policy_config.get("model_name", semantic_cfg.get("model_name", "facebook/dinov2-small")),
            "backend": self.policy_config.get("backend", semantic_cfg.get("backend", "huggingface")),
            "device": self.policy_config.get("device", semantic_cfg.get("device")),
            "use_pretrained": self.policy_config.get("use_pretrained", semantic_cfg.get("use_pretrained", True)),
            "fallback_enabled": self.policy_config.get("fallback_enabled", semantic_cfg.get("fallback_enabled", True)),
            "calibrate_semantic": semantic_cfg.get("calibrate_semantic", False),
            "cli_model_path": self.policy_config.get("cli_model_path", semantic_cfg.get("cli_model_path")),
            "model_path": self.policy_config.get("model_path", semantic_cfg.get("model_path")),
            "local_model_path": self.policy_config.get("local_model_path", semantic_cfg.get("local_model_path")),
            "require_real_backend": self.policy_config.get(
                "require_real_backend",
                semantic_cfg.get("require_real_backend", False),
            ),
        }
        for key, value in defaults.items():
            semantic_cfg.setdefault(key, value)
        return resolve_semantic_runtime_config(semantic_cfg)

    def _get_semantic_scorer(self) -> SemanticScorer:
        """Lazily instantiate the semantic scorer to keep V1 startup stable."""
        if self.semantic_scorer is None:
            self.semantic_scorer = SemanticScorer(self._build_semantic_config())
        return self.semantic_scorer

    @staticmethod
    def _calibrate_to_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Z-score calibration: transform source scores to have the same mean and std as target.

        This ensures alpha/beta weights operate on fairly scaled signals.
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.size == 0 or target.size == 0:
            return source.astype(np.float32)
        src_mean = source.mean()
        src_std = source.std()
        tgt_mean = target.mean()
        tgt_std = target.std()
        if src_std < 1e-8:
            return np.full_like(source, tgt_mean, dtype=np.float32)
        calibrated = (source - src_mean) / src_std * tgt_std + tgt_mean
        return calibrated.astype(np.float32)
        
    def predict_block_scores(self, 
                             blocks: np.ndarray, 
                             features: List[Dict[str, float]] = None, 
                             mode: str = None,
                             return_dict: bool = False,
                             *,
                             risk_signal_overrides: Mapping[str, np.ndarray] | None = None,
                             publication_required: bool = False,
                             image: np.ndarray | None = None,
                             grid_shape: tuple[int, int] | None = None) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Predict importance scores for a batch of blocks.
        
        Args:
            blocks: Input blocks (N, H, W, C).
            features: Pre-computed features (optional).
            mode: Scorer mode. If None, uses config default.
            return_dict: If True and mode is 'cnn_classifier', returns dict with classification details.
            
        Returns:
            np.ndarray: Importance scores [0, 1] if return_dict is False.
            Dict: Detailed output if return_dict is True and mode is 'cnn_classifier'.
        """
        # Determine mode
        if mode is None:
            mode = self.policy_config.get("scorer", "rule_based")
            
        # Map legacy names if any (e.g. 'entropy' -> rule_based)
        if mode in ["entropy", "variance", "edge"]:
            mode = "rule_based"
            
        if mode == "rule_based":
            if features is None:
                features = extract_features_batch(blocks)
            scores = self.rule_scorer.calculate_scores(features)
            if return_dict:
                 # Provide a dummy dict for compatibility if asked
                 return {"scores": scores}
            return scores
            
        elif mode == "cnn":
            if self.cnn_scorer is None:
                # Fallback init if not done in __init__
                cnn_cfg = {
                    "block_size": blocks.shape[1],
                    "channels": blocks.shape[3] if blocks.ndim==4 else 1
                }
                self.cnn_scorer = CNNScorer(cnn_cfg)
                
            scores = self.cnn_scorer.calculate_scores(blocks)
            if return_dict:
                return {"scores": scores}
            return scores
            
        elif mode == "cnn_classifier":
            if self.cnn_classifier_scorer is None:
                # Fallback init if not done in __init__
                cnn_cfg = {
                    "block_size": blocks.shape[1],
                    "channels": blocks.shape[3] if blocks.ndim==4 else 1
                }
                self.cnn_classifier_scorer = CNNClassifierScorer(cnn_cfg)
                
            return self.cnn_classifier_scorer.calculate_scores(blocks, return_dict=return_dict)
            
        elif mode == "fusion":
            # 1. Get rule-based scores
            if features is None:
                features = extract_features_batch(blocks)
            rule_scores = self.rule_scorer.calculate_scores(features)
            
            # 2. Get CNN signals
            if self.cnn_classifier_scorer is None:
                cnn_cfg = {
                    "block_size": blocks.shape[1],
                    "channels": blocks.shape[3] if blocks.ndim==4 else 1
                }
                self.cnn_classifier_scorer = CNNClassifierScorer(cnn_cfg)
                
            cnn_res = self.cnn_classifier_scorer.calculate_scores(blocks, return_dict=True)
            cnn_scores = cnn_res["scores"]
            
            # 3. Fusion (alpha parameter from config, default 0.7 for rule-based)
            alpha = self.policy_config.get("fusion_alpha", 0.7)
            final_scores = alpha * rule_scores + (1 - alpha) * cnn_scores
            
            if return_dict:
                return {
                    "scores": final_scores,
                    "rule_scores": rule_scores,
                    "cnn_scores": cnn_scores,
                    "cnn_predicted_class": cnn_res["predicted_class"],
                    "cnn_confidence": cnn_res["confidence"]
                }
            return final_scores

        elif mode == "semantic_fusion":
            if features is None:
                features = extract_features_batch(blocks)

            rule_scores = self.rule_scorer.calculate_scores(features)
            semantic_res = self._get_semantic_scorer().calculate_scores(
                blocks, return_dict=True, image=image, grid_shape=grid_shape
            )
            semantic_scores = semantic_res["scores"]

            semantic_cfg = self._build_semantic_config()
            alpha = float(
                self.policy_config.get(
                    "semantic_alpha",
                    semantic_cfg.get("alpha", 0.8),
                )
            )
            beta = float(
                self.policy_config.get(
                    "semantic_beta",
                    semantic_cfg.get("beta", 0.2),
                )
            )
            enable_calibration = semantic_cfg.get("calibrate_semantic", False)
            if enable_calibration:
                semantic_scores = self._calibrate_to_match(semantic_scores, rule_scores)
            final_scores = np.clip(alpha * rule_scores + beta * semantic_scores, 0.0, 1.0)

            if return_dict:
                return {
                    "scores": final_scores,
                    "final_score": final_scores,
                    "rule_scores": rule_scores,
                    "rule_score": rule_scores,
                    "semantic_scores": semantic_scores,
                    "semantic_score": semantic_scores,
                    "calibrated": enable_calibration,
                    "alpha": alpha,
                    "beta": beta,
                    "semantic_backend": semantic_res["backend"],
                    # Records which encoding actually ran, so a config that
                    # asks for image_patches but silently falls back to
                    # per_block cannot be mistaken for the former.
                    "semantic_encode_mode": semantic_res.get("semantic_encode_mode"),
                    "semantic_requested_backend": semantic_res["requested_backend"],
                    "semantic_model_name": semantic_res["model_name"],
                    "semantic_fallback_used": semantic_res["fallback_used"],
                    "semantic_backend_reason": semantic_res["backend_reason"],
                    "model_path_resolved": semantic_res["model_path_resolved"],
                    "device_requested": semantic_res["device_requested"],
                    "device_resolved": semantic_res["device_resolved"],
                    "cuda_available": semantic_res["cuda_available"],
                    "gpu_name": semantic_res["gpu_name"],
                    "fallback_reason": semantic_res["fallback_reason"],
                }
            return final_scores

        elif mode == "semantic_only":
            semantic_res = self._get_semantic_scorer().calculate_scores(
                blocks, return_dict=True, image=image, grid_shape=grid_shape
            )
            semantic_scores = semantic_res["scores"]

            if return_dict:
                return {
                    "scores": semantic_scores,
                    "semantic_scores": semantic_scores,
                    "semantic_score": semantic_scores,
                    "global_contrast_score": semantic_res["global_contrast_score"],
                    "embedding_norm_score": semantic_res["embedding_norm_score"],
                    "semantic_backend": semantic_res["backend"],
                    # Records which encoding actually ran, so a config that
                    # asks for image_patches but silently falls back to
                    # per_block cannot be mistaken for the former.
                    "semantic_encode_mode": semantic_res.get("semantic_encode_mode"),
                    "semantic_requested_backend": semantic_res["requested_backend"],
                    "semantic_model_name": semantic_res["model_name"],
                    "semantic_fallback_used": semantic_res["fallback_used"],
                    "semantic_backend_reason": semantic_res["backend_reason"],
                    "model_path_resolved": semantic_res["model_path_resolved"],
                    "device_requested": semantic_res["device_requested"],
                    "device_resolved": semantic_res["device_resolved"],
                    "cuda_available": semantic_res["cuda_available"],
                    "gpu_name": semantic_res["gpu_name"],
                    "fallback_reason": semantic_res["fallback_reason"],
                }
            return semantic_scores

        elif mode == "leakage_aware":
            if risk_signal_overrides is None:
                overrides: Mapping[str, np.ndarray] = {}
            elif isinstance(risk_signal_overrides, Mapping):
                overrides = risk_signal_overrides
            else:
                raise TypeError("risk_signal_overrides must be a mapping")

            spatial_names = ("object", "boundary", "context", "uncertainty")
            missing = sorted(name for name in spatial_names if name not in overrides)
            if missing and publication_required:
                raise ValueError(
                    "missing required risk signals: " + ",".join(missing)
                )

            block_count = len(blocks)
            if features is None:
                features = extract_features_batch(blocks)

            rule_scores = self.rule_scorer.calculate_scores(features)
            rule_count = len(np.asarray(rule_scores).reshape(-1))
            if rule_count != block_count:
                raise ValueError(
                    f"rule signal length {rule_count} != block count "
                    f"{block_count}"
                )
            semantic_res = self._get_semantic_scorer().calculate_scores(
                blocks, return_dict=True, image=image, grid_shape=grid_shape
            )
            semantic_scores = semantic_res["scores"]
            semantic_count = len(np.asarray(semantic_scores).reshape(-1))
            if semantic_count != block_count:
                raise ValueError(
                    f"semantic signal length {semantic_count} != block count "
                    f"{block_count}"
                )

            spatial_scores = {
                name: (
                    overrides[name]
                    if name in overrides
                    else np.zeros(block_count, dtype=np.float64)
                )
                for name in spatial_names
            }
            signals = build_risk_signals(
                rule_scores=rule_scores,
                semantic_scores=semantic_scores,
                object_scores=spatial_scores["object"],
                boundary_scores=spatial_scores["boundary"],
                context_scores=spatial_scores["context"],
                uncertainty_scores=spatial_scores["uncertainty"],
            )
            scorer = build_scorer("leakage_aware", self.config)
            fused = scorer.fuse(signals)
            fused_count = len(np.asarray(fused.scores).reshape(-1))
            if fused_count != block_count:
                raise ValueError(
                    f"fused scores length {fused_count} != block count "
                    f"{block_count}"
                )

            scorer_config = self.config["research"]["scorer"]
            provider_metadata: dict[str, dict[str, Any]] = {}
            for config_key in (
                "signal_providers",
                "signal_provider_digests",
            ):
                configured = scorer_config.get(config_key, {})
                if not isinstance(configured, Mapping):
                    raise ValueError(
                        f"config.research.scorer.{config_key} must be a mapping"
                    )
                provider_metadata[config_key] = dict(configured)

            semantic_metadata_keys = (
                "backend",
                "requested_backend",
                "model_name",
                "fallback_used",
                "backend_reason",
                "model_path_resolved",
                "device_requested",
                "device_resolved",
                "cuda_available",
                "gpu_name",
                "fallback_reason",
            )
            semantic_metadata = {
                key: semantic_res.get(key) for key in semantic_metadata_keys
            }
            fallback_reason = (
                "missing_risk_signals:" + ",".join(missing)
                if missing
                else None
            )
            detail = {
                "scores": fused.scores,
                "raw_components": fused.raw_components,
                "normalized_components": fused.normalized_components,
                "normalized_weights": fused.normalized_weights,
                "uncertainty_direction": fused.uncertainty_direction,
                "evidence_status": "PARTIAL" if missing else "FULL",
                "fallback_reason": fallback_reason,
                "semantic_backend_metadata": dict(semantic_metadata),
                "semantic_backend": semantic_metadata["backend"],
                "semantic_requested_backend": semantic_metadata[
                    "requested_backend"
                ],
                "semantic_model_name": semantic_metadata["model_name"],
                "semantic_fallback_used": semantic_metadata["fallback_used"],
                "semantic_backend_reason": semantic_metadata["backend_reason"],
                "model_path_resolved": semantic_metadata["model_path_resolved"],
                "device_requested": semantic_metadata["device_requested"],
                "device_resolved": semantic_metadata["device_resolved"],
                "cuda_available": semantic_metadata["cuda_available"],
                "gpu_name": semantic_metadata["gpu_name"],
                "semantic_model_path_resolved": semantic_metadata[
                    "model_path_resolved"
                ],
                "semantic_device_requested": semantic_metadata[
                    "device_requested"
                ],
                "semantic_device_resolved": semantic_metadata["device_resolved"],
                "semantic_cuda_available": semantic_metadata["cuda_available"],
                "semantic_gpu_name": semantic_metadata["gpu_name"],
                "semantic_fallback_reason": semantic_metadata[
                    "fallback_reason"
                ],
                **provider_metadata,
            }
            return detail if return_dict else fused.scores
        
        elif mode == "vit":
            if self.vit_scorer is None:
                # Fallback init if not done in __init__
                vit_cfg = {
                    "model_name": self.policy_config.get("vit_model_name", "vit_small_patch16_224"),
                    "use_pretrained": self.policy_config.get("use_pretrained", True),
                    "device": self.policy_config.get("device")
                }
                self.vit_scorer = ViTScorer(vit_cfg)
            
            scores = self.vit_scorer.calculate_scores(blocks)
            if return_dict:
                return {"scores": scores}
            return scores
        
        elif mode == "fusion_vit":
            # 1. Get rule-based scores
            if features is None:
                features = extract_features_batch(blocks)
            rule_scores = self.rule_scorer.calculate_scores(features)
            
            # 2. Get ViT scores
            if self.vit_scorer is None:
                vit_cfg = {
                    "model_name": self.policy_config.get("vit_model_name", "vit_small_patch16_224"),
                    "use_pretrained": self.policy_config.get("use_pretrained", True),
                    "device": self.policy_config.get("device")
                }
                self.vit_scorer = ViTScorer(vit_cfg)
            
            vit_res = self.vit_scorer.calculate_scores(blocks, return_dict=True)
            vit_scores = vit_res["scores"]
            
            # 3. Fusion (alpha parameter from config, default 0.7 for rule-based)
            alpha = self.policy_config.get("fusion_alpha", 0.7)
            final_scores = alpha * rule_scores + (1 - alpha) * vit_scores
            
            if return_dict:
                return {
                    "scores": final_scores,
                    "rule_scores": rule_scores,
                    "vit_scores": vit_scores,
                    "vit_model": vit_res["vit_model"],
                    "device": vit_res["device"]
                }
            return final_scores
            
        else:
            raise ValueError(f"Unknown scorer mode: {mode}")
