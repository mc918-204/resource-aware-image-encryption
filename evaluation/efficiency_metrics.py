import time
import numpy as np
from typing import Callable, Any
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class EfficiencyMetrics:
    """
    Metrics related to time and throughput.
    """
    
    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> tuple:
        """
        Measure execution time of a function.
        Returns: (result, time_ms)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, (end - start) * 1000.0

    @staticmethod
    def calculate_throughput(data_size_bytes: int, time_ms: float) -> float:
        """
        Calculate throughput in MB/s.
        """
        if time_ms <= 0:
            return 0.0
        seconds = time_ms / 1000.0
        mb = data_size_bytes / (1024 * 1024)
        return mb / seconds


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between original and reconstructed images.
    """
    try:
        if original.ndim == 3:
            return float(peak_signal_noise_ratio(original, reconstructed, channel_axis=2))
        else:
            return float(peak_signal_noise_ratio(original, reconstructed))
    except Exception:
        return 0.0


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between original and reconstructed images.
    """
    try:
        if original.ndim == 3:
            return float(structural_similarity(original, reconstructed, channel_axis=2, data_range=255))
        else:
            return float(structural_similarity(original, reconstructed, data_range=255))
    except Exception:
        return 0.0
