import numpy as np
import torch
from typing import Dict, Any, List
import timm

class ViTScorer:
    """
    Vision Transformer-based block importance scorer.
    Uses timm library to load pretrained ViT models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ViT scorer.
        
        Args:
            config: Configuration dictionary with model settings.
        """
        self.config = config or {}
        
        # Model configuration
        self.model_name = self.config.get("model_name", "vit_small_patch16_224")
        self.use_pretrained = self.config.get("use_pretrained", True)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = timm.create_model(
            self.model_name,
            pretrained=self.use_pretrained,
            num_classes=1
        ).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Preprocessing parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
    
    def preprocess(self, blocks: np.ndarray) -> torch.Tensor:
        """
        Preprocess blocks for ViT input.
        
        Args:
            blocks: Input blocks (N, H, W, C).
        
        Returns:
            torch.Tensor: Preprocessed blocks ready for ViT input.
        """
        # Convert to tensor
        blocks_tensor = torch.from_numpy(blocks).float()
        
        # Reshape to (N, C, H, W)
        if blocks_tensor.ndim == 4:
            blocks_tensor = blocks_tensor.permute(0, 3, 1, 2)
        
        # Move to device before normalization
        blocks_tensor = blocks_tensor.to(self.device)
        
        # Normalize
        blocks_tensor = (blocks_tensor / 255.0 - self.mean) / self.std
        
        # Resize to model input size if needed
        input_size = self.model.default_cfg.get("input_size", (3, 224, 224))
        if blocks_tensor.shape[2:] != input_size[1:]:
            import torch.nn.functional as F
            blocks_tensor = F.interpolate(blocks_tensor, size=input_size[1:], mode="bilinear", align_corners=False)
        
        return blocks_tensor
    
    def calculate_scores(self, blocks: np.ndarray, return_dict: bool = False) -> Dict[str, Any]:
        """
        Calculate importance scores for blocks using ViT.
        
        Args:
            blocks: Input blocks (N, H, W, C).
            return_dict: If True, return detailed dictionary.
        
        Returns:
            np.ndarray: Importance scores [0, 1] if return_dict is False.
            Dict: Detailed output if return_dict is True.
        """
        with torch.no_grad():
            # Preprocess
            preprocessed = self.preprocess(blocks)
            
            # Forward pass
            outputs = self.model(preprocessed)
            
            # Convert to probabilities using sigmoid
            scores = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            # Ensure shape is (N,)
            if scores.ndim == 0:
                scores = np.array([scores])
            
            # Clip to [0, 1]
            scores = np.clip(scores, 0.0, 1.0)
        
        if return_dict:
            return {
                "scores": scores,
                "vit_model": self.model_name,
                "device": self.device
            }
        
        return scores
