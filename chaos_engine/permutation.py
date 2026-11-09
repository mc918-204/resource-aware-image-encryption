import numpy as np
from typing import List, Tuple
from .generator import ChaosGenerator

class ChaosPermutation:
    """
    Handles permutation (shuffling) and diffusion (masking) operations using chaos.
    """
    
    def __init__(self, generator: ChaosGenerator):
        self.generator = generator
        
    def generate_permutation_indices(self, n: int) -> np.ndarray:
        """
        Generate a permutation of indices [0, n-1] using chaotic sort.
        
        Args:
            n: Length of permutation.
            
        Returns:
            np.ndarray: Permuted indices.
        """
        # Generate n chaotic values
        chaos_values = self.generator.generate_sequence(n)
        # Argsort gives the indices that would sort the array
        # This effectively shuffles indices based on chaotic values
        perm_indices = np.argsort(chaos_values)
        return perm_indices
        
    def shuffle_blocks(self, blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shuffle the order of blocks.
        
        Args:
            blocks: Array of blocks (N, ...).
            
        Returns:
            shuffled_blocks: The reordered blocks.
            perm_indices: The indices used for shuffling (needed for inverse).
        """
        n = len(blocks)
        indices = self.generate_permutation_indices(n)
        return blocks[indices], indices
        
    def unshuffle_blocks(self, shuffled_blocks: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
        """
        Restore original order of blocks.
        
        Args:
            shuffled_blocks: The shuffled blocks.
            perm_indices: The indices returned by shuffle_blocks.
            
        Returns:
            restored_blocks: Blocks in original order.
        """
        n = len(shuffled_blocks)
        restored = np.empty_like(shuffled_blocks)
        
        # If perm_indices[i] = k, it means original[k] moved to position i.
        # So restored[k] = shuffled[i]
        # Or: shuffled[i] came from original[perm_indices[i]]
        # So: shuffled = original[perm_indices]
        # To invert: restored[perm_indices] = shuffled
        
        restored[perm_indices] = shuffled_blocks
        return restored

    def diffuse(self, data: np.ndarray, rounds: int = 1) -> np.ndarray:
        """
        Apply XOR diffusion with chaotic mask.
        Used for lightweight visual confusion.
        
        Args:
            data: Input data (numpy array).
            rounds: Number of diffusion rounds.
            
        Returns:
            np.ndarray: Diffused data.
        """
        if rounds <= 0:
            return data.copy()
            
        result = data.copy()
        for _ in range(rounds):
            mask = self.generator.generate_mask(result.shape)
            result = np.bitwise_xor(result, mask)
            
        return result
