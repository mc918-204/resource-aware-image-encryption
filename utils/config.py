from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any, Dict

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from src.policy_model.spatial_signal_provider import MAX_BLOCK_SIZE, MAX_MARGIN_PX


AMBIGUOUS_WEIGHT_KEYS = {"alpha", "semantic_alpha", "fusion_alpha"}
SPATIAL_SIGNAL_ARTIFACT_VERSION = "v3_spatial_signals_v1"
SPATIAL_SIGNAL_KEYS = {
    "artifact_version",
    "model_name",
    "revision",
    "local_path",
    "cache_dir",
    "threshold",
    "device",
    "context_margin_px",
    "boundary_width_px",
    "seed",
    "accepted_licenses",
    "label_priors",
}
REDISTRIBUTABLE_MODEL_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
}
V3_LABEL_CONFIG_CONTRACT: Dict[str, Any] = {
    "artifact_version": "external_sensitive_labels_v2",
    "required_for_full": True,
    "protected_levels": ["high", "medium"],
    "overlap_threshold": 0.5,
    "mapping_version": "region_overlap_v1",
    "active_profile": "manual_review",
    "profiles": {
        "manual_review": {
            "mode": "manual_review",
            "label_source": "reviewed_privacy_regions_v1",
            "source_dataset": "publication_manifest_manual_review",
            "source_version": "2026-07-15",
            "license_note": "source licenses recorded in dataset manifest",
            "repeat_review_fraction": 0.1,
            "review_seed": 20260715,
        },
        "openimages_privacy_v1": {
            "mode": "official_openimages",
            "label_source": "open_images_v7_official_box",
            "source_dataset": "openimages_privacy_v1",
            "source_version": "Open Images V7",
            "license_note": (
                "Open Images annotations CC BY 4.0; original image licenses "
                "recorded per image"
            ),
            "selection_seed": 20260715,
            "classes": {
                "Human face": "face",
                "Vehicle registration plate": "license_plate",
            },
            "split_quotas": {
                "train": {
                    "Human face": 720,
                    "Vehicle registration plate": 720,
                },
                "validation": {
                    "Human face": 240,
                    "Vehicle registration plate": 240,
                },
                "test": {
                    "Human face": 240,
                    "Vehicle registration plate": 240,
                },
            },
            "excluded_box_flags": ["IsGroupOf", "IsDepiction"],
            "proxy_url": "http://127.0.0.1:10808",
            "dataset_dir": "data/publication/openimages_privacy_v1",
            "manifest_path": (
                "results/manifests/openimages_privacy_v1_manifest.csv"
            ),
            "split_artifact_path": (
                "results/manifests/openimages_privacy_v1_split.json"
            ),
            "label_artifact_path": (
                "results/labels/openimages_privacy_external_labels_v2.json"
            ),
            "label_audit_path": (
                "results/labels/openimages_privacy_external_labels_v2.audit.json"
            ),
        },
    },
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> Dict[Any, Any]:
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require_mapping(value: Any, path: str) -> Dict[str, Any]:
    """Return a configuration mapping or raise a path-specific error."""
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _validate_exact_contract(value: Any, expected: Any, path: str) -> None:
    """Recursively validate exact types, keys, lengths, and frozen values."""
    if type(value) is not type(expected):
        raise ValueError(f"{path} must have type {type(expected).__name__}")
    if isinstance(expected, dict):
        if set(value) != set(expected):
            missing = sorted(set(expected) - set(value))
            extra = sorted(set(value) - set(expected))
            raise ValueError(
                f"{path} key mismatch; missing={missing}, extra={extra}"
            )
        for key, expected_item in expected.items():
            _validate_exact_contract(value[key], expected_item, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{path} must equal {expected!r}")
        for index, expected_item in enumerate(expected):
            _validate_exact_contract(
                value[index], expected_item, f"{path}[{index}]"
            )
        return
    if value != expected:
        raise ValueError(f"{path} must equal {expected!r}")


def _validate_frozen_label_config(value: Any) -> Dict[str, Any]:
    """Validate the exact common and profile contracts for V3 labels."""
    labels = _require_mapping(value, "research.labels")
    threshold = labels.get("overlap_threshold")
    valid_threshold = (
        type(threshold) is float
        and math.isfinite(threshold)
        and threshold in {0.25, 0.5, 0.75}
    )
    if not valid_threshold:
        raise ValueError(
            "research.labels.overlap_threshold must be a finite numeric value "
            "from 0.25, 0.5, 0.75"
        )
    expected = deepcopy(V3_LABEL_CONFIG_CONTRACT)
    expected["overlap_threshold"] = threshold
    _validate_exact_contract(labels, expected, "research.labels")
    return labels


def _validate_spatial_signal_config(value: Any) -> Dict[str, Any]:
    """Validate the strict but path/model-configurable detector contract."""
    spatial = _require_mapping(value, "research.spatial_signals")
    if set(spatial) != SPATIAL_SIGNAL_KEYS:
        missing = sorted(SPATIAL_SIGNAL_KEYS - set(spatial))
        extra = sorted(set(spatial) - SPATIAL_SIGNAL_KEYS)
        raise ValueError(
            "research.spatial_signals key mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if spatial["artifact_version"] != SPATIAL_SIGNAL_ARTIFACT_VERSION:
        raise ValueError(
            "research.spatial_signals.artifact_version must equal "
            f"{SPATIAL_SIGNAL_ARTIFACT_VERSION!r}"
        )
    for field in ("model_name", "revision"):
        if type(spatial[field]) is not str or not spatial[field].strip():
            raise ValueError(
                f"research.spatial_signals.{field} must be a non-blank string"
            )
    for field in ("local_path", "cache_dir"):
        path_value = spatial[field]
        if path_value is not None and (
            type(path_value) is not str or not path_value.strip()
        ):
            raise ValueError(
                f"research.spatial_signals.{field} must be null or a non-blank path"
            )
    threshold = spatial["threshold"]
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) <= 1.0
    ):
        raise ValueError(
            "research.spatial_signals.threshold must be finite and in (0, 1]"
        )
    device = spatial["device"]
    if type(device) is not str or (
        device not in {"auto", "cpu", "cuda", "mps"}
        and re.fullmatch(r"cuda:\d+", device) is None
    ):
        raise ValueError(
            "research.spatial_signals.device must be auto, cpu, mps, cuda, "
            "or cuda:<index>"
        )
    for field in ("context_margin_px", "boundary_width_px", "seed"):
        number = spatial[field]
        if type(number) is not int or number < 0:
            raise ValueError(
                f"research.spatial_signals.{field} must be a non-negative integer"
            )
    for field in ("context_margin_px", "boundary_width_px"):
        if spatial[field] > MAX_MARGIN_PX:
            raise ValueError(
                f"research.spatial_signals.{field} must not exceed {MAX_MARGIN_PX}"
            )
    licenses = spatial["accepted_licenses"]
    if (
        type(licenses) is not list
        or not licenses
        or any(type(item) is not str or not item.strip() for item in licenses)
        or len(licenses) != len(set(licenses))
        or any(item not in REDISTRIBUTABLE_MODEL_LICENSES for item in licenses)
    ):
        raise ValueError(
            "research.spatial_signals.accepted_licenses must be a non-empty, "
            "unique allowlist of supported redistributable SPDX identifiers"
        )
    priors = _require_mapping(
        spatial["label_priors"],
        "research.spatial_signals.label_priors",
    )
    if not priors:
        raise ValueError(
            "research.spatial_signals.label_priors must not be empty"
        )
    for label, prior in priors.items():
        if type(label) is not str or not label.strip():
            raise ValueError(
                "research.spatial_signals.label_priors keys must be non-blank strings"
            )
        if (
            type(prior) not in (int, float)
            or not math.isfinite(float(prior))
            or not 0.0 <= float(prior) <= 1.0
        ):
            raise ValueError(
                "research.spatial_signals.label_priors values must be finite "
                "numbers in [0, 1]"
            )
    return spatial


def resolve_label_profile(
    config: Dict[str, Any],
    profile_name: str | None = None,
) -> Dict[str, Any]:
    """Return frozen common label settings merged with one selected profile.

    Args:
        config: Loaded V3 research configuration.
        profile_name: Explicit profile name, or ``None`` for ``active_profile``.

    Returns:
        A detached mapping containing common settings, selected profile fields,
        and the resolved ``profile_name``.

    Raises:
        ValueError: If the profile selection or nested label contract is invalid.
    """
    root = _require_mapping(config, "research config root")
    research = _require_mapping(root.get("research", {}), "research")
    labels = _require_mapping(research.get("labels", {}), "research.labels")
    profiles = labels.get("profiles")
    if type(profiles) is not dict:
        raise ValueError("label profiles must be a mapping")
    selected = labels.get("active_profile") if profile_name is None else profile_name
    if type(selected) is not str or not selected.strip():
        raise ValueError("label profile name must be a non-blank string")
    if selected not in profiles:
        raise ValueError(f"unsupported label profile: {selected!r}")
    _validate_frozen_label_config(labels)
    common = {
        key: deepcopy(labels[key])
        for key in (
            "artifact_version",
            "required_for_full",
            "protected_levels",
            "overlap_threshold",
            "mapping_version",
        )
    }
    return {
        **common,
        **deepcopy(profiles[selected]),
        "profile_name": selected,
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.load(f, Loader=_UniqueKeySafeLoader)
            return config
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML config: {e}")


def load_research_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate the frozen V3 research configuration contract.

    Args:
        config_path: Path to a V3 research YAML configuration file.

    Returns:
        Dict[str, Any]: Validated research configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        RuntimeError: If the configuration file contains malformed YAML.
        ValueError: If scorer weight aliases, weight keys, or status values
            violate the frozen V3 contract, or a required section is not a mapping.
    """
    config = _require_mapping(
        load_config(str(config_path)),
        "research config root",
    )

    policy = _require_mapping(config.get("policy", {}), "policy")
    research = _require_mapping(config.get("research", {}), "research")
    scorer = _require_mapping(research.get("scorer", {}), "research.scorer")
    ambiguous = sorted(
        [f"policy.{key}" for key in AMBIGUOUS_WEIGHT_KEYS & set(policy)]
        + [
            f"research.scorer.{key}"
            for key in AMBIGUOUS_WEIGHT_KEYS & set(scorer)
        ]
    )
    if ambiguous:
        raise ValueError(
            f"ambiguous scorer weight key(s): {ambiguous}; "
            "use research.scorer.weights"
        )

    weights = _require_mapping(
        scorer.get("weights", {}),
        "research.scorer.weights",
    )
    required = {"rule", "semantic", "object", "boundary", "context", "uncertainty"}
    if set(weights) != required:
        raise ValueError(
            f"research.scorer.weights must contain exactly {sorted(required)}"
        )
    for name, value in weights.items():
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"research.scorer.weights.{name} must be a finite numeric value in [0, 1]"
            )
    if not math.isclose(
        sum(float(value) for value in weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("research.scorer.weights must sum to 1")

    image = _require_mapping(config.get("image", {}), "image")
    block_size = image.get("block_size")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("image.block_size must be a positive integer")
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(f"image.block_size must not exceed {MAX_BLOCK_SIZE}")

    _validate_frozen_label_config(research.get("labels", {}))
    _validate_spatial_signal_config(research.get("spatial_signals", {}))

    experiment = _require_mapping(config.get("experiment", {}), "experiment")
    statuses = experiment.get("status_values")
    if statuses != ["DEGRADED", "PARTIAL", "FULL"]:
        raise ValueError("experiment.status_values must use the frozen three-state order")

    return config
