import numpy as np
from scipy.stats import pearsonr
from skimage.measure import shannon_entropy

class SecurityMetrics:
    """
    Standard security metrics for image encryption.
    """
    
    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """
        Calculate Shannon entropy of the image.
        Ideal value for encrypted image (8-bit) is close to 8.
        """
        if image.ndim == 1:
            return float(shannon_entropy(image))
        if image.ndim == 3:
            # Calculate per channel and average
            entropies = [shannon_entropy(image[:,:,i]) for i in range(image.shape[2])]
            return float(np.mean(entropies))
        else:
            return float(shannon_entropy(image))
            
    @staticmethod
    def calculate_correlation(image: np.ndarray, num_samples: int = 3000) -> dict:
        """
        Calculate correlation coefficient between adjacent pixels in three directions.
        Returns a dictionary with 'horizontal', 'vertical', and 'diagonal' correlations.
        Ideal value for encrypted image is close to 0.
        """
        if image.ndim == 3:
            # Average across channels
            corrs = {'horizontal': [], 'vertical': [], 'diagonal': []}
            for c in range(image.shape[2]):
                c_res = SecurityMetrics._channel_correlation_all(image[:,:,c], num_samples)
                corrs['horizontal'].append(c_res['horizontal'])
                corrs['vertical'].append(c_res['vertical'])
                corrs['diagonal'].append(c_res['diagonal'])
            
            return {
                'horizontal': float(np.mean(corrs['horizontal'])),
                'vertical': float(np.mean(corrs['vertical'])),
                'diagonal': float(np.mean(corrs['diagonal']))
            }
        else:
            channel_result = SecurityMetrics._channel_correlation_all(image, num_samples)
            return float(
                np.mean(
                    [
                        channel_result["horizontal"],
                        channel_result["vertical"],
                        channel_result["diagonal"],
                    ]
                )
            )

    @staticmethod
    def calculate_adjacent_correlations(image: np.ndarray, num_samples: int = 3000) -> dict:
        """Return horizontal/vertical/diagonal correlations for grayscale or RGB inputs."""
        if image.ndim == 1:
            if image.size < 2:
                return {"horizontal": 0.0, "vertical": 0.0, "diagonal": 0.0}
            pair_x = image[:-1].astype(float)
            pair_y = image[1:].astype(float)
            try:
                corr, _ = pearsonr(pair_x, pair_y)
                corr_value = float(corr) if not np.isnan(corr) else 0.0
            except Exception:
                corr_value = 0.0
            return {"horizontal": corr_value, "vertical": corr_value, "diagonal": corr_value}
        if image.ndim == 3:
            corrs = {"horizontal": [], "vertical": [], "diagonal": []}
            for channel_index in range(image.shape[2]):
                current = SecurityMetrics._channel_correlation_all(image[:, :, channel_index], num_samples)
                for key in corrs:
                    corrs[key].append(current[key])
            return {key: float(np.mean(values)) for key, values in corrs.items()}
        return SecurityMetrics._channel_correlation_all(image, num_samples)

    @staticmethod
    def _channel_correlation_all(channel: np.ndarray, num_samples: int) -> dict:
        h, w = channel.shape
        results = {}
        
        # Helper to compute correlation
        def compute_corr(pairs_x, pairs_y):
            if len(pairs_x) < 2: return 0.0
            # Use Pearson correlation
            try:
                val, _ = pearsonr(pairs_x, pairs_y)
                return float(val) if not np.isnan(val) else 0.0
            except:
                return 0.0

        # Horizontal
        x, y = [], []
        for _ in range(num_samples):
            r, c = np.random.randint(0, h), np.random.randint(0, w-1)
            x.append(float(channel[r, c]))
            y.append(float(channel[r, c+1]))
        results['horizontal'] = compute_corr(x, y)
            
        # Vertical
        x, y = [], []
        for _ in range(num_samples):
            r, c = np.random.randint(0, h-1), np.random.randint(0, w)
            x.append(float(channel[r, c]))
            y.append(float(channel[r+1, c]))
        results['vertical'] = compute_corr(x, y)
            
        # Diagonal
        x, y = [], []
        for _ in range(num_samples):
            r, c = np.random.randint(0, h-1), np.random.randint(0, w-1)
            x.append(float(channel[r, c]))
            y.append(float(channel[r+1, c+1]))
        results['diagonal'] = compute_corr(x, y)
            
        return results

    @staticmethod
    def calculate_npcr_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> tuple:
        """
        Calculate NPCR (Number of Pixels Change Rate) and UACI (Unified Average Changing Intensity).
        Requires two ciphertext images from plaintexts differing by 1 bit/pixel.
        
        Args:
            cipher1: First ciphertext image.
            cipher2: Second ciphertext image.
            
        Returns:
            (npcr, uaci)
        """
        if cipher1.shape != cipher2.shape:
            raise ValueError("Images must have same dimensions")
            
        c1 = cipher1.astype(float)
        c2 = cipher2.astype(float)
        
        # NPCR
        diff = c1 != c2
        npcr = np.sum(diff) / diff.size * 100
        
        # UACI
        abs_diff = np.abs(c1 - c2)
        uaci = np.sum(abs_diff) / (abs_diff.size * 255) * 100
        
        return float(npcr), float(uaci)

    @staticmethod
    def calculate_plain_cipher_correlation(plain: np.ndarray, cipher: np.ndarray) -> float:
        """
        Calculate the correlation between plaintext and ciphertext images.
        Ideal value is close to 0.
        """
        if plain.shape != cipher.shape:
            # Resize cipher to match plain if needed (or vice versa), but strictly they should match
            # For this context, we assume they are same size. If not, raise error.
            raise ValueError("Plain and Cipher images must have same dimensions")

        p_flat = plain.flatten().astype(float)
        c_flat = cipher.flatten().astype(float)
        
        # If image is too large, sample it to save time, though full correlation is more accurate
        # Let's use full calculation for accuracy as per requirements
        
        try:
            val, _ = pearsonr(p_flat, c_flat)
            return float(val) if not np.isnan(val) else 0.0
        except:
            return 0.0


def calculate_entropy(image: np.ndarray) -> float:
    return SecurityMetrics.calculate_entropy(image)


def calculate_correlation(image: np.ndarray, num_samples: int = 3000) -> dict:
    return SecurityMetrics.calculate_correlation(image, num_samples)


def calculate_adjacent_correlations(image: np.ndarray, num_samples: int = 3000) -> dict:
    return SecurityMetrics.calculate_adjacent_correlations(image, num_samples)


def calculate_npcr(plain: np.ndarray, cipher: np.ndarray) -> float:
    if plain.shape != cipher.shape:
        raise ValueError("Images must have same dimensions")
    diff = plain.astype(float) != cipher.astype(float)
    return float(np.sum(diff) / diff.size * 100)


def calculate_uaci(plain: np.ndarray, cipher: np.ndarray) -> float:
    if plain.shape != cipher.shape:
        raise ValueError("Images must have same dimensions")
    abs_diff = np.abs(plain.astype(float) - cipher.astype(float))
    return float(np.sum(abs_diff) / (abs_diff.size * 255) * 100)


def safe_calculate_npcr_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> tuple[float | None, float | None]:
    """Return NPCR/UACI or ``(None, None)`` when the inputs are not suitable."""
    try:
        return SecurityMetrics.calculate_npcr_uaci(cipher1, cipher2)
    except Exception:
        return None, None
