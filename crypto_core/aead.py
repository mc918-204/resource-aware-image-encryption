import os
from typing import Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

class AEADPrimitive:
    """
    Wrapper for authenticated encryption primitives.
    Provides a unified interface for AES-GCM and ChaCha20-Poly1305.
    """
    
    def __init__(self, key: bytes, algorithm: str = "AES-GCM"):
        """
        Args:
            key: 32 bytes key.
            algorithm: 'AES-GCM' or 'ChaCha20-Poly1305'.
        """
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes.")
            
        self.algorithm = algorithm
        self.key = key
        
        if algorithm == "AES-GCM":
            self.cipher = AESGCM(key)
        elif algorithm == "ChaCha20-Poly1305":
            self.cipher = ChaCha20Poly1305(key)
        else:
            # Fallback or error
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    def encrypt(self, data: bytes, aad: bytes = b"", nonce: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Encrypt data.
        
        Args:
            data: Plaintext.
            aad: Additional Authenticated Data (integrity protected but not encrypted).
            nonce: Optional 12-byte nonce. If None, generated randomly.
            
        Returns:
            nonce: The used nonce (12 bytes).
            ciphertext: Encrypted data (includes auth tag at the end).
        """
        if nonce is None:
            nonce = os.urandom(12)
            
        if len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes.")
            
        # cryptography library appends tag to ciphertext automatically
        ciphertext = self.cipher.encrypt(nonce, data, aad)
        
        return nonce, ciphertext

    def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        """
        Decrypt data and verify integrity.
        
        Args:
            nonce: 12-byte nonce used during encryption.
            ciphertext: Encrypted data (with tag).
            aad: AAD used during encryption.
            
        Returns:
            bytes: Plaintext.
            
        Raises:
            InvalidTag: If integrity check fails (tampering or wrong key/AAD).
        """
        if len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes.")
            
        return self.cipher.decrypt(nonce, ciphertext, aad)
