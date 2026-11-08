import numpy as np
from typing import List, Dict, Any

class RuleBasedScorer:
    """
    Computes importance scores based on weighted sum of features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Config dict containing 'weights'.
                    Default weights: entropy=0.4, variance=0.2, edge=0.3, texture=0.1
        """
        self.weights = {
            "entropy": 0.4,
            "variance": 0.2,
            "edge_strength": 0.3,
            "texture_score": 0.1
        }
        
        if config and "weights" in config:
            self.weights.update(config["weights"])
            
    def calculate_scores(self, feature_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Calculate raw importance scores.
        
        Args:
            feature_list: List of feature dicts.
            
        Returns:
            np.ndarray: Array of scores (normalized to [0, 1]).
        """
        if not feature_list:
            return np.array([])
            
        scores = []
        
        # Extract individual feature arrays for normalization
        keys = self.weights.keys()
        data = {k: [] for k in keys}
        
        for f in feature_list:
            for k in keys:
                data[k].append(f.get(k, 0.0))
                
        # Normalize each feature to [0, 1] across the batch to ensure fair weighting
        # Otherwise entropy (0-8) and variance (0-0.1) have vastly different scales.
        normalized_data = {}
        for k in keys:
            arr = np.array(data[k])
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val - min_val > 1e-6:
                normalized_data[k] = (arr - min_val) / (max_val - min_val)
            else:
                normalized_data[k] = np.zeros_like(arr)
                
        # Weighted sum
        n = len(feature_list)
        final_scores = np.zeros(n)
        
        for k, w in self.weights.items():
            final_scores += w * normalized_data[k]
            
        # Final normalization to [0, 1]
        s_min, s_max = np.min(final_scores), np.max(final_scores)
        if s_max - s_min > 1e-6:
            final_scores = (final_scores - s_min) / (s_max - s_min)
            
        return final_scores

    def classify_levels(self, scores: np.ndarray, thresholds: List[float] = [0.3, 0.7]) -> List[str]:
        """
        Classify scores into levels based on percentiles.
        Note: This logic is also present in BudgetAllocator, but provided here for completeness if needed.
        """
        # ... logic similar to allocator ...
        # For now, we leave the actual allocation to BudgetAllocator
        pass
