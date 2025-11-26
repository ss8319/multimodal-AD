import torch
import numpy as np
import random
import os
import sys
import argparse
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
import nibabel as nib
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Fix random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def extract_middle_slice(nii_path, slice_axis=2):
    """
    Extract middle slice from 3D NIfTI image.
    
    Args:
        nii_path: Path to NIfTI file
        slice_axis: Axis along which to extract slice (0=D, 1=H, 2=W)
    
    Returns:
        2D numpy array of the middle slice
    """
    img_nii = nib.load(nii_path)
    img_data = img_nii.get_fdata()
    
    # Get middle slice along specified axis (with -15 offset to match BrainIAC)
    if slice_axis == 0:
        slice_idx = img_data.shape[0] // 2 - 15
        slice_2d = img_data[slice_idx, :, :]
    elif slice_axis == 1:
        slice_idx = img_data.shape[1] // 2 - 15
        slice_2d = img_data[:, slice_idx, :]
    else:  # slice_axis == 2 (default)
        slice_idx = img_data.shape[2] // 2 - 15
        slice_2d = img_data[:, :, slice_idx]
    
    return slice_2d, slice_idx

def preprocess_slice_for_dinov3(slice_2d, target_size=224):
    """
    Preprocess 2D slice for DINOv3 model.
    
    DINOv3 expects:
    - RGB images (3 channels)
    - Size: 224x224 (for ViT-B/16)
    - Normalized pixel values
    
    Args:
        slice_2d: 2D numpy array (grayscale)
        target_size: Target image size (default 224 for ViT-B/16)
    
    Returns:
        Preprocessed tensor ready for DINOv3 [1, 3, H, W]
    """
    # Normalize to [0, 1]
    slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
    
    # Convert to PIL Image
    slice_pil = Image.fromarray((slice_norm * 255).astype(np.uint8))
    
    # Resize to target size
    slice_pil = slice_pil.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to RGB (duplicate grayscale channel to 3 channels)
    slice_rgb = slice_pil.convert('RGB')
    
    # Convert to tensor and normalize
    # DINOv3 uses ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(slice_rgb)
    return tensor.unsqueeze(0)  # Add batch dimension: [1, 3, H, W]

def infer_patch_embeddings(model, processor, image_paths, batch_size=1):
    """
    Extract patch embeddings from DINOv3 model for 2D slices.
    
    Args:
        model: DINOv3 model (from transformers)
        processor: DINOv3 image processor (from transformers)
        image_paths: List of paths to NIfTI files
        batch_size: Batch size for processing
    
    Returns:
        List of patch embeddings, each of shape [n_patches, embed_dim]
    """
    model.eval()
    all_patch_embeddings = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting patch embeddings", unit="image"):
            try:
                # Extract middle slice
                slice_2d, slice_idx = extract_middle_slice(img_path)
                
                # Preprocess for DINOv3
                # Option 1: Use processor (recommended)
                # Convert slice to PIL RGB
                slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
                slice_pil = Image.fromarray((slice_norm * 255).astype(np.uint8))
                slice_rgb = slice_pil.convert('RGB')
                
                # Resize to model's expected size (processor handles this)
                inputs = processor(images=slice_rgb, return_tensors="pt").to(device)
                
                # Forward pass
                outputs = model(**inputs)
                
                # Extract patch embeddings
                # last_hidden_state shape: [batch_size, num_tokens, embed_dim]
                # num_tokens = 1 (CLS) + n_patches
                # For ViT-B/16 with 224x224 input: n_patches = (224/16)^2 = 14^2 = 196
                last_hidden_state = outputs.last_hidden_state  # [1, 197, 768] for ViT-B/16
                
                # Extract patch tokens (exclude CLS token at index 0)
                patch_tokens = last_hidden_state[:, 1:, :]  # [1, 196, 768]
                
                # Remove batch dimension: [n_patches, embed_dim]
                patch_tokens = patch_tokens.squeeze(0)  # [196, 768]
                
                patch_tokens_numpy = patch_tokens.cpu().numpy()
                all_patch_embeddings.append(patch_tokens_numpy)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    return all_patch_embeddings

def main():
    parser = argparse.ArgumentParser(description='Extract DINOv3 patch embeddings from 3D MRI images (using middle slice)')
    parser.add_argument('--model_name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                      help='DINOv3 model name from Hugging Face (default: facebook/dinov3-vitb16-pretrain-lvd1689m)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory to save features (default: XAI/features/)')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                      help='List of image paths (if not provided, uses default ICBM images)')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--slice_axis', type=int, default=2,
                      help='Axis along which to extract middle slice (0=D, 1=H, 2=W, default: 2)')
    
    args = parser.parse_args()
    
    # Default images if not provided
    if args.images is None:
        base_path = "/home/ssim0068/data/multimodal-dataset/all_icbm/images"
        default_images = [
            f"{base_path}/126_S_0606.nii.gz",
            f"{base_path}/031_S_1209.nii.gz",
            f"{base_path}/023_S_0926.nii.gz",
            f"{base_path}/007_S_1206.nii.gz"
        ]
        image_paths = default_images
    else:
        image_paths = args.images
    
    # Set output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "features"
    else:
        output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("DINOv3 Patch Embedding Extraction")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {len(image_paths)}")
    print(f"Slice axis: {args.slice_axis}")
    print("=" * 60)
    
    # Load DINOv3 model and processor
    print(f"\nLoading DINOv3 model: {args.model_name}")
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(
            args.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available() or device.type == 'cuda':
            model = model.to(device)
        model.eval()
        print(f"Model loaded successfully!")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Embedding dimension: {model.config.hidden_size}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Extract patch embeddings
    print("\nExtracting patch embeddings from middle slices...")
    patch_embeddings_list = infer_patch_embeddings(model, processor, image_paths, args.batch_size)
    
    if not patch_embeddings_list:
        print("No patch embeddings were extracted!")
        return
    
    print(f"\nExtracted patch embeddings from {len(patch_embeddings_list)} images")
    for idx, patch_emb in enumerate(patch_embeddings_list):
        img_path = image_paths[idx]
        img_id = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
        print(f"  {img_id}: {patch_emb.shape} (n_patches={patch_emb.shape[0]}, feature_dim={patch_emb.shape[1]})")
    
    # Save patch embeddings
    # Save all patches in one file (for convenience)
    all_patches = np.vstack(patch_embeddings_list)  # Shape: [total_patches, embed_dim]
    all_patches_file = os.path.join(output_dir, "all_dinov3_patch_embeddings.npy")
    np.save(all_patches_file, all_patches)
    print(f"\nAll patch embeddings saved to {all_patches_file}")
    print(f"Total patches: {all_patches.shape[0]}, Feature dim: {all_patches.shape[1]}")
    
    # Save individual patch embedding files
    for idx, patch_emb in enumerate(patch_embeddings_list):
        img_path = image_paths[idx]
        img_id = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
        individual_file = os.path.join(output_dir, f"{img_id}_dinov3_patch_embeddings.npy")
        np.save(individual_file, patch_emb)
    
    print(f"\nIndividual patch embedding files saved to {output_dir}")
    
    # Save metadata about patch structure
    actual_patches = patch_embeddings_list[0].shape[0] if patch_embeddings_list else 0
    patch_info = {
        'model_name': args.model_name,
        'input_size': (224, 224),  # DINOv3 ViT-B/16 default
        'patch_size': 16,  # ViT-B/16 uses 16x16 patches
        'patches_per_slice': actual_patches,  # Typically 196 for 224x224 with 16x16 patches
        'feature_dim': patch_embeddings_list[0].shape[1] if patch_embeddings_list else 0,
        'n_samples': len(patch_embeddings_list),
        'slice_axis': args.slice_axis,
        'note': 'DINOv3 is 2D native - embeddings extracted from middle slice of 3D MRI'
    }
    import json
    info_file = os.path.join(output_dir, "dinov3_patch_embeddings_info.json")
    with open(info_file, 'w') as f:
        json.dump(patch_info, f, indent=2)
    print(f"Patch embeddings metadata saved to {info_file}")
    print(f"  Patches per slice: {actual_patches}")
    print(f"  Feature dimension: {patch_info['feature_dim']}")
    
    print("\n" + "=" * 60)
    print(f"Feature extraction complete!")
    print(f"Extracted patch embeddings from {len(image_paths)} images")
    print(f"Features saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

