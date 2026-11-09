import numpy as np

class LogisticMap:
    """
    Implements the Logistic Map chaotic system: x_{n+1} = r * x_n * (1 - x_n).
    
    This module is used as an auxiliary source of pseudo-randomness for 
    permutation and lightweight masking. It is NOT a cryptographically secure 
    random number generator (CSPRNG).
    """
    
    def __init__(self, x0: float, r: float):
        """
        Initialize the Logistic Map.
        
        Args:
            x0: Initial state (0 < x0 < 1).
            r: Control parameter (3.57 < r <= 4.0 for chaos).
        """
        if not (0.0 < x0 < 1.0):
            raise ValueError(f"Initial value x0 must be in (0, 1), got {x0}")
        if not (0.0 < r <= 4.0):
            raise ValueError(f"Control parameter r must be in (0, 4], got {r}")
            
        self.x0 = x0
        self.r = r
        self.state = x0
        
    def reset(self):
        """Reset the generator to its initial state."""
        self.state = self.x0
        
    def generate(self, n: int) -> np.ndarray:
        """
        Generate n chaotic values in range (0, 1).
        
        Args:
            n: Number of values to generate.
            
        Returns:
            np.ndarray: Array of n float values.
        """
        if n <= 0:
            return np.array([])
            
        sequence = np.empty(n, dtype=np.float64)
        x = self.state
        r = self.r
        
        # Simple loop for iteration. 
        # Numba could optimize this, but standard Python is fine for prototype scale.
        for i in range(n):
            x = r * x * (1.0 - x)
            sequence[i] = x
            
        self.state = x
        return sequence
