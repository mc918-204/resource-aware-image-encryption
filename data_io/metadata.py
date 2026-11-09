from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import json
import base64
import numpy as np

@dataclass
class ImageMetadata:
    """
    Metadata required for decryption and reconstruction.
    """
    original_shape: Tuple[int, ...]
    padded_shape: Tuple[int, ...]
    block_size: int
    num_blocks: int
    channels: int
    padding_info: Dict[str, Any]
    session_id: str
    chaos_backend: str = "logistic"

    # Optional fields for reconstruction
    perm_indices: Optional[List[int]] = None
    block_levels: Optional[List[str]] = None

    # Key-derivation profile (plan item C1). A missing value means the frozen
    # V1 session schedule, so archived packages keep decrypting unchanged.
    # `image_nonce` is the public 16-byte HKDF salt, hex-encoded, and is
    # required by (and only meaningful for) `v2_domain_separated`.
    kdf_profile: str = "v1_session"
    image_nonce: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_shape": self.original_shape,
            "padded_shape": self.padded_shape,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "channels": self.channels,
            "padding_info": self.padding_info,
            "session_id": self.session_id,
            "chaos_backend": self.chaos_backend,
            "perm_indices": self.perm_indices,
            "block_levels": self.block_levels,
            "kdf_profile": self.kdf_profile,
            "image_nonce": self.image_nonce,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageMetadata':
        """Load from dictionary."""
        return cls(
            original_shape=tuple(data["original_shape"]),
            padded_shape=tuple(data["padded_shape"]),
            block_size=data["block_size"],
            num_blocks=data["num_blocks"],
            channels=data["channels"],
            padding_info=data["padding_info"],
            session_id=data["session_id"],
            chaos_backend=data.get("chaos_backend", "logistic"),
            perm_indices=data.get("perm_indices"),
            block_levels=data.get("block_levels"),
            kdf_profile=data.get("kdf_profile") or "v1_session",
            image_nonce=data.get("image_nonce"),
        )

@dataclass
class EncryptedPackage:
    """
    Container for the full encrypted payload.
    """
    metadata: ImageMetadata
    payload: List[Dict[str, Any]] # List of block data
    
    def save(self, path: str):
        """Save package to JSON file (for prototype)."""
        # Convert binary data to base64 for JSON
        serializable_payload = []
        for block in self.payload:
            item = block.copy()
            if isinstance(item.get('data'), bytes):
                item['data'] = base64.b64encode(item['data']).decode('ascii')
            if isinstance(item.get('nonce'), bytes):
                item['nonce'] = base64.b64encode(item['nonce']).decode('ascii')
            serializable_payload.append(item)
            
        data = {
            "metadata": self.metadata.to_dict(),
            "payload": serializable_payload
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'EncryptedPackage':
        """Load package from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        meta = ImageMetadata.from_dict(data['metadata'])
        
        # Decode base64
        payload = []
        for item in data['payload']:
            block = item.copy()
            if 'data' in block:
                block['data'] = base64.b64decode(block['data'])
            if 'nonce' in block:
                block['nonce'] = base64.b64decode(block['nonce'])
            payload.append(block)
            
        return cls(metadata=meta, payload=payload)
