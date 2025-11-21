import torch
import numpy as np
import pandas as pd
import random
import os
import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# Add parent directory to path to import BrainIAC modules
sys.path.insert(0, str(Path(__file__).parent.parent / "BrainIAC" / "src"))

from load_brainiac import load_brainiac
from dataset import BrainAgeDataset, get_validation_transform
from get_brainiac_features import infer
from tqdm import tqdm

# Fix random seed (same as get_brainiac_features.py)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set GPU (same as get_brainiac_features.py)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def infer_patch_embeddings(model, test_loader):
    """
    Extract patch embeddings from BrainIAC ViT.
    
    Note: model is ViTBackboneNet, which wraps MONAI ViT.
    - model.forward() returns CLS token only [batch_size, 768]
    - model.backbone is the MONAI ViT, which returns all tokens
    - model.backbone(inputs) returns tuple: (all_tokens, ...)
    - all_tokens shape: [batch_size, num_tokens, hidden_dim] where num_tokens = 1 (CLS) + n_patches
    
    Returns patch embeddings for each sample: [n_patches, 768]
    """
    model.eval()
    all_patch_embeddings = []
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Extracting patch embeddings", unit="batch"):
            inputs = sample['image'].to(device)
            
            # Access MONAI ViT directly (model.backbone) to get all tokens
            # model.backbone is the MONAI ViT, not ViTBackboneNet wrapper
            # MONAI ViT returns tuple: (all_tokens, ...)
            # all_tokens shape: [batch_size, num_tokens, hidden_dim]
            # num_tokens = 1 (CLS) + n_patches (typically 215-216)
            features= model.backbone(inputs)  # Returns tuple from MONAI ViT

            # Extract ALL Patch Embeddings (tokens from index 1 onwards)
            patch_tokens = features[0][:, 1:]  # Shape: [batch_size, num_patches, hidden_dim]
                        
            # For batch_size=1, squeeze batch dimension: [n_patches, 768]
            if patch_tokens.shape[0] == 1:
                patch_tokens = patch_tokens.squeeze(0)
            
            patch_tokens_numpy = patch_tokens.cpu().numpy()
            all_patch_embeddings.append(patch_tokens_numpy)
    
    return all_patch_embeddings

def main():
    parser = argparse.ArgumentParser(description='Extract BrainIAC features from specific images')
    parser.add_argument('--checkpoint', type=str, default='src/mri/BrainIAC/src/checkpoints/BrainIAC.ckpt',
                      help='Path to the BrainIAC checkpoint file')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory to save features (default: XAI/features/)')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                      help='List of image paths (if not provided, uses default ICBM images)')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers for data loading (default: 1)')
    parser.add_argument('--use_patch_embeddings', action='store_true',
                      help='Extract patch embeddings instead of CLS tokens (default: False)')
    
    args = parser.parse_args()
    
    # Default images if not provided
    if args.images is None:
        base_path = "/home/ssim0068/data/multimodal-dataset/all_icbm/images"
        default_images = [
            f"{base_path}/002_S_0413.nii.gz",
            f"{base_path}/002_S_0559.nii.gz",
            f"{base_path}/005_S_0602.nii.gz",
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
    print("BrainIAC Feature Extraction")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {len(image_paths)}")
    print("=" * 60)
    
    # Create temporary CSV with image paths (reuse BrainAgeDataset pattern)
    print("\nCreating temporary CSV for dataset...")
    temp_csv_data = []
    for img_path in image_paths:
        img_id = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
        temp_csv_data.append({
            'mri_path': img_path,
            'label': 0.0  # Dummy label, not used for feature extraction
        })
    
    temp_csv = pd.DataFrame(temp_csv_data)
    temp_csv_path = os.path.join(output_dir, "temp_images.csv")
    temp_csv.to_csv(temp_csv_path, index=False)
    print(f"Temporary CSV created: {temp_csv_path}")
    
    # Setup dataset with validation transforms (same as get_brainiac_features.py)
    print("\nSetting up dataset and dataloader...")
    test_dataset = BrainAgeDataset(
        csv_path=temp_csv_path,
        root_dir="",  # Not needed since mri_path is absolute
        transform=get_validation_transform()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load ViT BrainIAC (same as get_brainiac_features.py)
    model = load_brainiac(args.checkpoint, device)
    
    if args.use_patch_embeddings:
        # Extract patch embeddings instead of CLS tokens
        print("\nExtracting patch embeddings...")
        patch_embeddings_list = infer_patch_embeddings(model, test_loader)
        
        print(f"\nExtracted patch embeddings from {len(patch_embeddings_list)} images")
        for idx, patch_emb in enumerate(patch_embeddings_list):
            img_path = image_paths[idx]
            img_id = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
            print(f"  {img_id}: {patch_emb.shape} (n_patches={patch_emb.shape[0]}, feature_dim={patch_emb.shape[1]})")
        
        # Save patch embeddings
        # Save all patches in one file (for convenience)
        all_patches = np.vstack(patch_embeddings_list)  # Shape: [total_patches, 768]
        all_patches_file = os.path.join(output_dir, "all_patch_embeddings.npy")
        np.save(all_patches_file, all_patches)
        print(f"\nAll patch embeddings saved to {all_patches_file}")
        print(f"Total patches: {all_patches.shape[0]}, Feature dim: {all_patches.shape[1]}")
        
        # Save individual patch embedding files
        for idx, patch_emb in enumerate(patch_embeddings_list):
            img_path = image_paths[idx]
            img_id = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
            individual_file = os.path.join(output_dir, f"{img_id}_patch_embeddings.npy")
            np.save(individual_file, patch_emb)
        
        print(f"\nIndividual patch embedding files saved to {output_dir}")
        
        # Also save metadata about patch structure
        # Use actual number of patches extracted (typically 215 from MONAI ViT)
        actual_patches = patch_embeddings_list[0].shape[0] if patch_embeddings_list else 0
        patch_info = {
            'image_size': (96, 96, 96),
            'patch_size': (16, 16, 16),
            'actual_patches_per_image': actual_patches,  # Actual number extracted (typically 215)
            'feature_dim': 768,
            'n_samples': len(patch_embeddings_list)
        }
        import json
        info_file = os.path.join(output_dir, "patch_embeddings_info.json")
        with open(info_file, 'w') as f:
            json.dump(patch_info, f, indent=2)
        print(f"Patch embeddings metadata saved to {info_file}")
        print(f"  Actual patches per image: {actual_patches}")
        
    else:
        # Extract CLS tokens (original behavior)
        print("\nExtracting CLS tokens...")
        features_df = infer(model, test_loader) #infer function returns CLS token
        
        # Save features (same pattern as get_brainiac_features.py)
        csv_file = os.path.join(output_dir, "features.csv")
        features_df.to_csv(csv_file, index=False)
        print(f"\nViT BrainIAC features saved to {csv_file}")
        print(f"Feature shape: {features_df.shape}")
        print(f"Number of feature dimensions: {features_df.shape[1] - 1}")  # -1 for label column
        
        # Also save as numpy for convenience
        feature_cols = [col for col in features_df.columns if col.startswith('Feature_')]
        features_array = features_df[feature_cols].values
        npy_file = os.path.join(output_dir, "all_features.npy")
        np.save(npy_file, features_array)
        print(f"Features numpy array saved to {npy_file}")
        print(f"Features array shape: {features_array.shape}")
        
        # Save individual feature files
        for idx, row in features_df.iterrows():
            img_path = image_paths[idx]
            img_id = Path(img_path).stem.replace('.nii', '').replace('.gz', '')
            feature_array = row[feature_cols].values
            individual_file = os.path.join(output_dir, f"{img_id}_features.npy")
            np.save(individual_file, feature_array)
        
        print(f"\nIndividual feature files saved to {output_dir}")
    
    # Clean up temporary CSV
    os.remove(temp_csv_path)
    print(f"Temporary CSV removed")
    
    print("\n" + "=" * 60)
    print(f"Feature extraction complete!")
    if args.use_patch_embeddings:
        print(f"Extracted patch embeddings from {len(image_paths)} images")
    else:
        print(f"Extracted CLS tokens from {len(image_paths)} images")
    print(f"Features saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

