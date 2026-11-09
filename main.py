import os
import sys
import argparse
import scipy.linalg  # noqa: F401; initialize MKL before Torch on Windows
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data_io.loader import load_image
from src.data_io.saver import save_image
from src.pipeline import EncryptionPipeline

def main():
    parser = argparse.ArgumentParser(description="Resource-Aware Image Encryption Demo")
    parser.add_argument("--input", type=str, default="data/test_image.png", help="Path to input image")
    parser.add_argument("--output", type=str, default="results/recovered_image.png", help="Path to save recovered image")
    parser.add_argument("--package", type=str, default="results/encrypted_data.json", help="Path to save encrypted package")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--budget", type=float, default=2000.0, help="Time budget (ms) for encryption")
    parser.add_argument("--model-path", type=str, default=None, help="Optional local Hugging Face model snapshot path")
    parser.add_argument("--model-revision", type=str, default=None, help="Optional Hugging Face model commit revision")
    
    args = parser.parse_args()
    
    # Ensure results dir
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 1. Config
    if not os.path.exists(args.config):
        print(f"Error: Config not found at {args.config}")
        # Create default config if missing? Or just exit.
        # For robustness, let's create a minimal config in memory or warn.
        print("Using internal defaults...")
        cfg = {"policy": {"scorer": "rule_based"}, "image": {"block_size": 16}}
    else:
        cfg = load_config(args.config)

    if args.model_path:
        semantic_cfg = cfg.setdefault("policy", {}).setdefault("semantic", {})
        semantic_cfg["model_path"] = args.model_path
    if args.model_revision:
        semantic_cfg = cfg.setdefault("policy", {}).setdefault("semantic", {})
        semantic_cfg["model_revision"] = args.model_revision
    
    # 2. Pipeline
    # Ensure we are using Rule-Based as main method unless config overrides
    print(f"Initializing Pipeline with mode: {cfg.get('policy', {}).get('scorer', 'rule_based')}")
    pipeline = EncryptionPipeline(cfg)
    
    # 3. Load Image
    if not os.path.exists(args.input):
        print(f"Input image not found at {args.input}. Creating dummy image...")
        os.makedirs(os.path.dirname(args.input), exist_ok=True)
        create_dummy_image(args.input)
        
    try:
        # Load
        image_path = args.input
        # Use PIL to load to ensure robust format handling
        pil_img = Image.open(image_path).convert('RGB')
        image = np.array(pil_img)
        print(f"Loaded image: {args.input} {image.shape}")
        
        # 4. Encrypt
        print(f"Encrypting with budget={args.budget} ms...")
        # Measure time
        import time
        t0 = time.perf_counter()
        pkg = pipeline.encrypt(image, budget=args.budget)
        t1 = time.perf_counter()
        duration = (t1 - t0) * 1000.0
        
        # Log status
        # Extract status from metadata if available, else infer
        # Note: metadata might not have status field directly exposed as property, check payload
        # But pipeline logs it.
        print(f"Encryption completed in {duration:.2f} ms")
        
        package_dir = os.path.dirname(args.package)
        if package_dir:
            os.makedirs(package_dir, exist_ok=True)
        pkg.save(args.package)
        print(f"Encrypted package saved to {args.package}")
        
        # 5. Decrypt
        print("Decrypting...")
        recovered_image = pipeline.decrypt(pkg)
        
        # 6. Save & Verify
        save_image(recovered_image, args.output)
        print(f"Recovered image saved to {args.output}")
        
        if np.array_equal(image, recovered_image):
            print("\nSUCCESS: Image recovered perfectly.")
        else:
            diff = np.sum(np.abs(image.astype(int) - recovered_image.astype(int)))
            # Allow small diff for lossy formats? No, we use PNG/BMP usually.
            # But AES is lossless.
            if diff == 0:
                 print("\nSUCCESS: Image recovered perfectly.")
            else:
                 print(f"\nWARNING: Mismatch detected! Total pixel difference: {diff}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

def create_dummy_image(path: str):
    """Create a dummy image for testing."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Gradient
    for i in range(128):
        img[i, :, 0] = i * 2
        img[:, i, 1] = 255 - i * 2
    # Box
    img[30:90, 30:90, 2] = 255 
    Image.fromarray(img).save(path)

if __name__ == "__main__":
    main()
