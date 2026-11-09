import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

class KeyManager:
    """
    Manages key derivation and distribution using HKDF.
    Follows NIST SP 800-108 recommendations for key derivation.
    """
    
    def __init__(self, master_key: bytes):
        """
        Args:
            master_key: 32 bytes (256 bits) master key.
        """
        if len(master_key) != 32:
            raise ValueError("Master key must be exactly 32 bytes (256 bits).")
        self.master_key = master_key
        
    def derive_session_key(self, session_id: bytes, length: int = 32) -> bytes:
        """
        Derive a session-specific key from the master key using HKDF-SHA256.
        
        Args:
            session_id: Unique identifier for the session (salt/context).
            length: Length of the derived key in bytes (default 32 for AES-256).
            
        Returns:
            bytes: The derived session key.
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,  # Optional: in some protocols session_id is salt, in others info.
            info=session_id, # Binding context to key derivation
            backend=default_backend()
        )
        
        return hkdf.derive(self.master_key)

    @staticmethod
    def generate_master_key() -> bytes:
        """Generate a random 256-bit master key."""
        return os.urandom(32)
