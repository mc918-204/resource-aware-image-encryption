import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for block importance scoring (regression).
    
    Architecture:
    Input (C, H, W) -> Conv -> ReLU -> Pool -> Conv -> ReLU -> FC -> Score
    """
    def __init__(self, input_channels: int = 3, block_size: int = 16, output_dim: int = 1):
        super(SimpleCNN, self).__init__()
        
        # Lightweight architecture
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate FC input size
        # block_size=16 -> pool -> 8 -> pool -> 4
        final_size = block_size // 4
        self.fc_input_dim = 32 * final_size * final_size
        
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim) # Output single score [0, 1]
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Reshape: (Batch, Channels*H*W)
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Score between 0 and 1
        return x

class CNNImportanceClassifier(nn.Module):
    """
    CNN model for block importance classification (High, Medium, Low).
    Output is logits for 3 classes.
    """
    def __init__(self, input_channels: int = 3, block_size: int = 16):
        super(CNNImportanceClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        final_size = block_size // 4
        self.fc_input_dim = 32 * final_size * final_size
        
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 3) # 3 classes: Low, Medium, High
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Raw logits
        return x

class CNNClassifierScorer:
    """
    CNN-based classification scorer wrapper.
    Outputs class probabilities, predicted classes, and an expected score.
    Class 0: Low
    Class 1: Medium
    Class 2: High
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.block_size = self.config.get("block_size", 16)
        self.channels = self.config.get("input_channels", self.config.get("channels", 3))
        self.model_path = self.config.get("checkpoint_path", self.config.get("model_path", None))
        self.use_pretrained = bool(self.config.get("use_pretrained", False))
        
        device_cfg = self.config.get("device")
        if isinstance(device_cfg, str):
            self.device = torch.device(device_cfg)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.model = CNNImportanceClassifier(self.channels, self.block_size).to(self.device)
        
        if self.model_path and self.use_pretrained:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"Loaded CNN classifier from {self.model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}")
                print("Using randomly initialized model.")
        else:
            self.model.eval()

    def calculate_scores(self, blocks: np.ndarray, return_dict: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculates classification output.
        Args:
            blocks: Input blocks (N, H, W, C) or (N, H, W)
            return_dict: If True, returns a dict with classes, probabilities, etc.
                         If False, returns expected scores for backward compatibility.
        """
        if len(blocks) == 0:
            if return_dict:
                return {
                    "scores": np.array([]),
                    "predicted_class": np.array([]),
                    "class_probabilities": np.array([]),
                    "confidence": np.array([])
                }
            return np.array([])
            
        if blocks.ndim == 3: 
            tensor = torch.from_numpy(blocks).unsqueeze(1).float() / 255.0
        else: 
            tensor = torch.from_numpy(blocks).permute(0, 3, 1, 2).float() / 255.0
        
        in_c = tensor.shape[1]
        model_c = self.model.conv1.in_channels
        if in_c != model_c:
            if in_c == 3 and model_c == 1:
                tensor = tensor.mean(dim=1, keepdim=True)
            elif in_c == 1 and model_c == 3:
                tensor = tensor.repeat(1, 3, 1, 1)
            else:
                if in_c > model_c:
                    tensor = tensor[:, :model_c]
                else:
                    reps = (model_c + in_c - 1) // in_c
                    tensor = tensor.repeat(1, reps, 1, 1)[:, :model_c]
                    
        with torch.no_grad():
            tensor = tensor.to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            
            # Predict class and confidence
            confidence, predicted_class = torch.max(probs, dim=1)
            
            # Expected score: 0.0 * P(Low) + 0.5 * P(Medium) + 1.0 * P(High)
            # Assuming class 0: Low, class 1: Medium, class 2: High
            weights = torch.tensor([0.0, 0.5, 1.0], device=self.device)
            expected_scores = (probs * weights).sum(dim=1)
            
            scores_np = expected_scores.cpu().numpy()
            
            if return_dict:
                return {
                    "scores": scores_np,
                    "predicted_class": predicted_class.cpu().numpy(),
                    "class_probabilities": probs.cpu().numpy(),
                    "confidence": confidence.cpu().numpy()
                }
            
        return scores_np

    def train_dummy(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved dummy classifier model to {save_path}")

class CNNScorer:
    """
    CNN-based scorer wrapper.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.block_size = self.config.get("block_size", 16)
        # Accept both 'channels' and 'input_channels'
        self.channels = self.config.get("input_channels", self.config.get("channels", 3))
        # Accept both 'model_path' and 'checkpoint_path'
        self.model_path = self.config.get("checkpoint_path", self.config.get("model_path", None))
        self.use_pretrained = bool(self.config.get("use_pretrained", False))
        # Device: explicit > cuda if available > cpu
        device_cfg = self.config.get("device")
        if isinstance(device_cfg, str):
            self.device = torch.device(device_cfg)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Output dim kept as 1 (score)
        output_dim = int(self.config.get("output_dim", 1))
        self.model = SimpleCNN(self.channels, self.block_size, output_dim=output_dim).to(self.device)
        
        if self.model_path and self.use_pretrained:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"Loaded CNN model from {self.model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}")
                print("Using randomly initialized model. NOTE: Untrained model scores are not meaningful.")
        else:
            if self.model_path and not self.use_pretrained:
                print("Info: Checkpoint path is provided but use_pretrained is False; using randomly initialized model.")
            else:
                print("Warning: No model path provided. Using randomly initialized model.")
            print("NOTE: Untrained model is for structure validation only, not for final comparisons.")
            self.model.eval()

    def calculate_scores(self, blocks: np.ndarray) -> np.ndarray:
        """
        Calculate importance scores using CNN.
        
        Args:
            blocks: Numpy array (N, H, W, C) or (N, H, W).
            
        Returns:
            np.ndarray: Scores [0, 1].
        """
        if len(blocks) == 0:
            return np.array([])
            
        # Preprocess for PyTorch: (N, H, W, C) -> (N, C, H, W)
        # Also normalize to [0, 1]
        if blocks.ndim == 3: # (N, H, W) -> (N, 1, H, W)
            tensor = torch.from_numpy(blocks).unsqueeze(1).float() / 255.0
        else: # (N, H, W, C) -> (N, C, H, W)
            tensor = torch.from_numpy(blocks).permute(0, 3, 1, 2).float() / 255.0
        
        # Channel adaptation to match model input
        in_c = tensor.shape[1]
        model_c = self.model.conv1.in_channels
        if in_c != model_c:
            if in_c == 3 and model_c == 1:
                # Convert to grayscale by averaging channels
                tensor = tensor.mean(dim=1, keepdim=True)
            elif in_c == 1 and model_c == 3:
                # Repeat grayscale to 3 channels
                tensor = tensor.repeat(1, 3, 1, 1)
            else:
                # Fallback: project to required channels by slicing or repeating
                if in_c > model_c:
                    tensor = tensor[:, :model_c]
                else:
                    reps = (model_c + in_c - 1) // in_c
                    tensor = tensor.repeat(1, reps, 1, 1)[:, :model_c]
            
        # Handle batching if too large? 
        # For prototype, assume fits in memory or CPU
        
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            scores = output.cpu().numpy().reshape(-1)
            
        return scores

    def train_dummy(self, save_path: str):
        """
        Create a dummy trained model for testing purposes.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved dummy model to {save_path}")
