import os
import uuid
from typing import Tuple, Dict
from .key_manager import KeyManager
from .aead import AEADPrimitive

class CryptoSession:
    """
    Manages a secure session for encrypting multiple blocks/messages.
    Derives a session key and manages state (e.g. session_id).
    """
    
    def __init__(self, master_key: bytes, session_id: bytes = None):
        """
        Args:
            master_key: The long-term master key.
            session_id: Optional session identifier. If None, generates random UUID.
        """
        self.key_manager = KeyManager(master_key)
        
        if session_id is None:
            self.session_id = str(uuid.uuid4()).encode('utf-8')
        else:
            self.session_id = session_id
            
        # Derive session keys for different security levels
        # We use context strings to differentiate keys
        # L1 (High) - AES-GCM
        # L2 (Medium) - ChaCha20
        
        key_high = self.key_manager.derive_session_key(self.session_id + b"_HIGH")
        key_medium = self.key_manager.derive_session_key(self.session_id + b"_MED")
        
        self.cipher_high = AEADPrimitive(key_high, "AES-GCM")
        self.cipher_medium = AEADPrimitive(key_medium, "ChaCha20-Poly1305")
        
    def encrypt_block(self, 
                      data: bytes, 
                      level: str, 
                      block_index: int,
                      extra_aad: bytes = b"") -> Tuple[bytes, bytes]:
        """
        Encrypt a single block.
        Binds block_index to AAD to prevent block swapping attacks.
        
        Args:
            data: Block data.
            level: 'high', 'medium', or 'low'.
            block_index: Index of the block.
            extra_aad: Any other context to bind.
            
        Returns:
            (nonce, ciphertext)
        """
        # Bind context to AAD
        # Format: SessionID | BlockIndex | Level | Extra
        aad = self.session_id + block_index.to_bytes(4, 'big') + level.encode() + extra_aad
        
        if level == "high":
            return self.cipher_high.encrypt(data, aad)
        elif level == "medium":
            return self.cipher_medium.encrypt(data, aad)
        else:
            # Low security: No encryption (Pass-through)
            # In a real system, you might want at least integrity, but per requirement:
            # L3 is "Light protection", maybe just chaos. Here we return plaintext.
            return b"", data

    def decrypt_block(self, 
                      nonce: bytes, 
                      ciphertext: bytes, 
                      level: str, 
                      block_index: int,
                      extra_aad: bytes = b"") -> bytes:
        """
        Decrypt a single block.
        """
        aad = self.session_id + block_index.to_bytes(4, 'big') + level.encode() + extra_aad
        
        if level == "high":
            return self.cipher_high.decrypt(nonce, ciphertext, aad)
        elif level == "medium":
            return self.cipher_medium.decrypt(nonce, ciphertext, aad)
        else:
            return ciphertext
            
    def get_session_info(self) -> Dict[str, str]:
        return {
            "session_id": self.session_id.decode('utf-8', errors='ignore'),
            "algorithm_high": "AES-GCM",
            "algorithm_medium": "ChaCha20-Poly1305"
        }
