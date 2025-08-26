"""
Quality Control utilities for MRI data loading and preprocessing.

This module provides comprehensive QC checks for:
- DICOM volume loading completeness
- Image dimensions and resolution
- Preprocessing quality
- Visualization of raw and processed data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DataQualityController:
    """Comprehensive quality control for MRI data pipeline."""
    
    def __init__(self, save_plots: bool = True, output_dir: Optional[Path] = None):
        """Initialize QC controller.
        
        Args:
            save_plots: Whether to save QC plots to disk
            output_dir: Directory to save QC outputs
        """
        self.save_plots = save_plots
        self.output_dir = Path(output_dir) if output_dir else Path("qc_outputs")
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.qc_results = {}
    
    def check_dicom_volume_loading(self, dicom_folder_path: str, adni_base_path: str) -> Dict:
        """Comprehensive QC for DICOM volume loading.
        
        Args:
            dicom_folder_path: Relative path to DICOM folder
            adni_base_path: Base path to ADNI data
            
        Returns:
            Dictionary with QC results
        """
        full_path = Path(adni_base_path) / dicom_folder_path
        qc_info = {
            'folder_path': str(full_path),
            'folder_exists': full_path.exists(),
            'total_files': 0,
            'dcm_files': 0,
            'valid_dicoms': 0,
            'slice_count': 0,
            'matrix_sizes': [],
            'slice_thicknesses': [],
            'pixel_spacings': [],
            'sort_methods_available': [],
            'volume_shape': None,
            'intensity_range': None,
            'errors': []
        }
        
        if not full_path.exists():
            qc_info['errors'].append(f"Folder not found: {full_path}")
            return qc_info
        
        # Count all files
        all_files = []
        for root, dirs, files in os.walk(full_path):
            all_files.extend(files)
        qc_info['total_files'] = len(all_files)
        
        # Find DICOM files
        dicom_files = []
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        qc_info['dcm_files'] = len(dicom_files)
        
        if not dicom_files:
            qc_info['errors'].append("No DICOM files found")
            return qc_info
        
        # Analyze DICOM files
        valid_dicoms = []
        matrix_sizes = set()
        slice_thicknesses = set()
        pixel_spacings = set()
        sort_methods = set()
        
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'pixel_array'):
                    valid_dicoms.append(ds)
                    
                    # Collect metadata
                    if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                        matrix_sizes.add(f"{ds.Rows}x{ds.Columns}")
                    
                    if hasattr(ds, 'SliceThickness'):
                        slice_thicknesses.add(float(ds.SliceThickness))
                    
                    if hasattr(ds, 'PixelSpacing'):
                        spacing = f"{ds.PixelSpacing[0]:.2f}x{ds.PixelSpacing[1]:.2f}"
                        pixel_spacings.add(spacing)
                    
                    # Check sorting methods
                    if hasattr(ds, 'SliceLocation'):
                        sort_methods.add('SliceLocation')
                    if hasattr(ds, 'ImagePositionPatient'):
                        sort_methods.add('ImagePositionPatient')
                    if hasattr(ds, 'InstanceNumber'):
                        sort_methods.add('InstanceNumber')
                        
            except Exception as e:
                qc_info['errors'].append(f"Failed to read {dcm_file}: {e}")
        
        qc_info['valid_dicoms'] = len(valid_dicoms)
        qc_info['slice_count'] = len(valid_dicoms)
        qc_info['matrix_sizes'] = list(matrix_sizes)
        qc_info['slice_thicknesses'] = list(slice_thicknesses)
        qc_info['pixel_spacings'] = list(pixel_spacings)
        qc_info['sort_methods_available'] = list(sort_methods)
        
        # Create volume if we have valid DICOMs
        if valid_dicoms:
            try:
                # Sort DICOMs (using same logic as dataset)
                dicom_data = []
                for ds in valid_dicoms:
                    sort_key = None
                    if hasattr(ds, 'SliceLocation'):
                        sort_key = float(ds.SliceLocation)
                    elif hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                        sort_key = float(ds.ImagePositionPatient[2])
                    elif hasattr(ds, 'InstanceNumber'):
                        sort_key = int(ds.InstanceNumber)
                    else:
                        sort_key = 0
                    
                    dicom_data.append((sort_key, ds.pixel_array))
                
                dicom_data.sort(key=lambda x: x[0])
                volume = np.stack([data[1] for data in dicom_data], axis=0)
                
                qc_info['volume_shape'] = volume.shape
                qc_info['intensity_range'] = (float(volume.min()), float(volume.max()))
                
            except Exception as e:
                qc_info['errors'].append(f"Failed to create volume: {e}")
        
        return qc_info
    
    def check_preprocessing_quality(self, raw_volume: np.ndarray, 
                                  processed_volume: np.ndarray,
                                  normalization_method: str,
                                  target_size: Tuple[int, int, int]) -> Dict:
        """QC for preprocessing pipeline.
        
        Args:
            raw_volume: Original volume before preprocessing
            processed_volume: Volume after preprocessing
            normalization_method: Normalization method used
            target_size: Target size for resizing
            
        Returns:
            Dictionary with preprocessing QC results
        """
        qc_info = {
            'raw_shape': raw_volume.shape,
            'processed_shape': processed_volume.shape,
            'target_shape': target_size,
            'shape_matches_target': processed_volume.shape == target_size,
            'normalization_method': normalization_method,
            'raw_intensity_range': (float(raw_volume.min()), float(raw_volume.max())),
            'processed_intensity_range': (float(processed_volume.min()), float(processed_volume.max())),
            'raw_mean_std': (float(raw_volume.mean()), float(raw_volume.std())),
            'processed_mean_std': (float(processed_volume.mean()), float(processed_volume.std())),
            'resize_factors': None,
            'intensity_preserved': True,
            'warnings': []
        }
        
        # Calculate resize factors
        resize_factors = [
            processed_volume.shape[i] / raw_volume.shape[i] for i in range(3)
        ]
        qc_info['resize_factors'] = resize_factors
        
        # Check if extreme resizing
        for i, factor in enumerate(resize_factors):
            if factor < 0.25 or factor > 4.0:
                qc_info['warnings'].append(
                    f"Extreme resize factor {factor:.2f} for axis {i}"
                )
        
        # Check normalization quality
        if normalization_method == 'zscore':
            if abs(processed_volume.mean()) > 0.1:
                qc_info['warnings'].append(
                    f"Z-score mean not close to 0: {processed_volume.mean():.3f}"
                )
            if abs(processed_volume.std() - 1.0) > 0.2:
                qc_info['warnings'].append(
                    f"Z-score std not close to 1: {processed_volume.std():.3f}"
                )
        
        # Check for data corruption
        if np.isnan(processed_volume).any():
            qc_info['warnings'].append("NaN values found in processed volume")
        
        if np.isinf(processed_volume).any():
            qc_info['warnings'].append("Infinite values found in processed volume")
        
        return qc_info
    
    def visualize_raw_volume(self, volume: np.ndarray, title: str = "Raw Volume",
                           subject_id: str = "Unknown", save_name: str = None) -> None:
        """Visualize raw loaded volume.
        
        Args:
            volume: 3D volume to visualize
            title: Plot title
            subject_id: Subject identifier
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{title} - {subject_id}\nShape: {volume.shape}, Range: [{volume.min():.1f}, {volume.max():.1f}]')
        
        # Show slices from different views
        mid_slice = volume.shape[0] // 2
        mid_row = volume.shape[1] // 2
        mid_col = volume.shape[2] // 2
        
        # Axial view (top row)
        axes[0, 0].imshow(volume[mid_slice//2], cmap='gray')
        axes[0, 0].set_title(f'Axial (slice {mid_slice//2})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(volume[mid_slice], cmap='gray')
        axes[0, 1].set_title(f'Axial (slice {mid_slice})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(volume[mid_slice + mid_slice//2], cmap='gray')
        axes[0, 2].set_title(f'Axial (slice {mid_slice + mid_slice//2})')
        axes[0, 2].axis('off')
        
        # Sagittal and coronal views (bottom row)
        axes[1, 0].imshow(volume[:, mid_row, :], cmap='gray')
        axes[1, 0].set_title('Sagittal')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(volume[:, :, mid_col], cmap='gray')
        axes[1, 1].set_title('Coronal')
        axes[1, 1].axis('off')
        
        # Intensity histogram
        axes[1, 2].hist(volume.flatten(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Intensity Distribution')
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            save_path = self.output_dir / f"{save_name}_raw.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"     💾 Saved raw volume plot: {save_path}")
        
        plt.show()
    
    def visualize_preprocessing_comparison(self, raw_volume: np.ndarray, 
                                        processed_volume: np.ndarray,
                                        subject_id: str = "Unknown",
                                        save_name: str = None) -> None:
        """Visualize before/after preprocessing comparison.
        
        Args:
            raw_volume: Original volume
            processed_volume: Processed volume
            subject_id: Subject identifier
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Preprocessing Comparison - {subject_id}')
        
        # Get middle slices
        raw_mid = raw_volume.shape[0] // 2
        proc_mid = processed_volume.shape[0] // 2
        
        # Raw volume views
        axes[0, 0].imshow(raw_volume[raw_mid], cmap='gray')
        axes[0, 0].set_title(f'Raw Axial\nShape: {raw_volume.shape}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(raw_volume[:, raw_volume.shape[1]//2, :], cmap='gray')
        axes[0, 1].set_title('Raw Sagittal')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(raw_volume[:, :, raw_volume.shape[2]//2], cmap='gray')
        axes[0, 2].set_title('Raw Coronal')
        axes[0, 2].axis('off')
        
        # Raw histogram
        axes[0, 3].hist(raw_volume.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 3].set_title(f'Raw Intensities\nRange: [{raw_volume.min():.1f}, {raw_volume.max():.1f}]')
        axes[0, 3].set_xlabel('Intensity')
        
        # Processed volume views
        axes[1, 0].imshow(processed_volume[proc_mid], cmap='gray')
        axes[1, 0].set_title(f'Processed Axial\nShape: {processed_volume.shape}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(processed_volume[:, processed_volume.shape[1]//2, :], cmap='gray')
        axes[1, 1].set_title('Processed Sagittal')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(processed_volume[:, :, processed_volume.shape[2]//2], cmap='gray')
        axes[1, 2].set_title('Processed Coronal')
        axes[1, 2].axis('off')
        
        # Processed histogram
        axes[1, 3].hist(processed_volume.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 3].set_title(f'Processed Intensities\nRange: [{processed_volume.min():.1f}, {processed_volume.max():.1f}]')
        axes[1, 3].set_xlabel('Intensity')
        
        # Side-by-side comparison
        axes[2, 0].imshow(np.concatenate([raw_volume[raw_mid], processed_volume[proc_mid]], axis=1), cmap='gray')
        axes[2, 0].set_title('Side-by-side: Raw (left) vs Processed (right)')
        axes[2, 0].axis('off')
        
        # Overlay histograms
        axes[2, 1].hist(raw_volume.flatten(), bins=50, alpha=0.5, color='blue', label='Raw')
        axes[2, 1].hist(processed_volume.flatten(), bins=50, alpha=0.5, color='red', label='Processed')
        axes[2, 1].set_title('Intensity Comparison')
        axes[2, 1].set_xlabel('Intensity')
        axes[2, 1].legend()
        
        # Stats comparison
        stats_text = f"""
        Raw Stats:
        Mean: {raw_volume.mean():.3f}
        Std: {raw_volume.std():.3f}
        Min: {raw_volume.min():.1f}
        Max: {raw_volume.max():.1f}
        
        Processed Stats:
        Mean: {processed_volume.mean():.3f}
        Std: {processed_volume.std():.3f}
        Min: {processed_volume.min():.1f}
        Max: {processed_volume.max():.1f}
        """
        axes[2, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        axes[2, 2].axis('off')
        axes[2, 2].set_title('Statistics')
        
        # Resize factors visualization
        resize_factors = [processed_volume.shape[i] / raw_volume.shape[i] for i in range(3)]
        axes[2, 3].bar(['Depth', 'Height', 'Width'], resize_factors)
        axes[2, 3].set_title('Resize Factors')
        axes[2, 3].set_ylabel('Factor')
        axes[2, 3].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            save_path = self.output_dir / f"{save_name}_preprocessing.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"     💾 Saved preprocessing comparison: {save_path}")
        
        plt.show()
    
    def run_comprehensive_qc(self, dataset, sample_indices: List[int] = None, 
                           max_samples: int = 3) -> Dict:
        """Run comprehensive QC on dataset samples.
        
        Args:
            dataset: MRIDataset instance
            sample_indices: Specific indices to check (random if None)
            max_samples: Maximum number of samples to check
            
        Returns:
            Dictionary with comprehensive QC results
        """
        if sample_indices is None:
            sample_indices = np.random.choice(len(dataset), 
                                            size=min(max_samples, len(dataset)), 
                                            replace=False)
        
        print(f"🔍 Running comprehensive QC on {len(sample_indices)} samples...")
        
        all_qc_results = {
            'samples_checked': len(sample_indices),
            'dicom_loading_results': [],
            'preprocessing_results': [],
            'overall_stats': {
                'successful_loads': 0,
                'failed_loads': 0,
                'common_matrix_sizes': {},
                'common_slice_counts': {},
                'intensity_ranges': []
            }
        }
        
        for i, idx in enumerate(sample_indices):
            print(f"\n📋 QC Sample {i+1}/{len(sample_indices)} (index {idx})")
            row = dataset.df.iloc[idx]
            subject_id = row['Subject']
            group = row['Group']
            
            print(f"   👤 Subject: {subject_id} ({group})")
            
            # 1. Check DICOM loading
            print("   🔍 Checking DICOM loading...")
            dicom_qc = self.check_dicom_volume_loading(
                row['dicom_folder_path'], 
                str(dataset.adni_base_path)
            )
            all_qc_results['dicom_loading_results'].append({
                'subject_id': subject_id,
                'index': idx,
                'results': dicom_qc
            })
            
            # Report DICOM QC
            if dicom_qc['errors']:
                print(f"   ❌ DICOM Loading Issues:")
                for error in dicom_qc['errors']:
                    print(f"      • {error}")
                all_qc_results['overall_stats']['failed_loads'] += 1
                continue
            else:
                print(f"   ✅ DICOM Loading Success:")
                print(f"      • Folder: {dicom_qc['folder_exists']}")
                print(f"      • Total files: {dicom_qc['total_files']}")
                print(f"      • DICOM files: {dicom_qc['dcm_files']}")
                print(f"      • Valid DICOMs: {dicom_qc['valid_dicoms']}")
                print(f"      • Volume shape: {dicom_qc['volume_shape']}")
                print(f"      • Matrix sizes: {dicom_qc['matrix_sizes']}")
                print(f"      • Sort methods: {dicom_qc['sort_methods_available']}")
                
                all_qc_results['overall_stats']['successful_loads'] += 1
                
                # Track common patterns
                if dicom_qc['volume_shape']:
                    slice_count = dicom_qc['volume_shape'][0]
                    matrix_size = f"{dicom_qc['volume_shape'][1]}x{dicom_qc['volume_shape'][2]}"
                    
                    if matrix_size not in all_qc_results['overall_stats']['common_matrix_sizes']:
                        all_qc_results['overall_stats']['common_matrix_sizes'][matrix_size] = 0
                    all_qc_results['overall_stats']['common_matrix_sizes'][matrix_size] += 1
                    
                    if slice_count not in all_qc_results['overall_stats']['common_slice_counts']:
                        all_qc_results['overall_stats']['common_slice_counts'][slice_count] = 0
                    all_qc_results['overall_stats']['common_slice_counts'][slice_count] += 1
                    
                    if dicom_qc['intensity_range']:
                        all_qc_results['overall_stats']['intensity_ranges'].append(dicom_qc['intensity_range'])
            
            # 2. Load and preprocess through dataset
            try:
                print("   🔧 Checking preprocessing...")
                raw_volume = dataset.load_dicom_volume(row['dicom_folder_path'])
                processed_volume = dataset.preprocess_volume(raw_volume)
                
                # QC preprocessing
                prep_qc = self.check_preprocessing_quality(
                    raw_volume, processed_volume, 
                    dataset.normalization_method, dataset.target_size
                )
                all_qc_results['preprocessing_results'].append({
                    'subject_id': subject_id,
                    'index': idx,
                    'results': prep_qc
                })
                
                print(f"   ✅ Preprocessing Success:")
                print(f"      • Raw shape: {prep_qc['raw_shape']}")
                print(f"      • Processed shape: {prep_qc['processed_shape']}")
                print(f"      • Target achieved: {prep_qc['shape_matches_target']}")
                print(f"      • Raw range: [{prep_qc['raw_intensity_range'][0]:.1f}, {prep_qc['raw_intensity_range'][1]:.1f}]")
                print(f"      • Processed range: [{prep_qc['processed_intensity_range'][0]:.3f}, {prep_qc['processed_intensity_range'][1]:.3f}]")
                
                if prep_qc['warnings']:
                    print(f"   ⚠️ Preprocessing Warnings:")
                    for warning in prep_qc['warnings']:
                        print(f"      • {warning}")
                
                # 3. Visualize if requested
                if i < 2:  # Only visualize first 2 samples to avoid too many plots
                    print("   📊 Creating visualizations...")
                    save_name = f"sample_{idx}_{subject_id.replace('_', '-')}"
                    
                    self.visualize_raw_volume(
                        raw_volume, 
                        title="Raw DICOM Volume",
                        subject_id=f"{subject_id} ({group})",
                        save_name=save_name
                    )
                    
                    self.visualize_preprocessing_comparison(
                        raw_volume, processed_volume,
                        subject_id=f"{subject_id} ({group})",
                        save_name=save_name
                    )
                
            except Exception as e:
                print(f"   ❌ Preprocessing failed: {e}")
                all_qc_results['overall_stats']['failed_loads'] += 1
        
        # Summary report
        print(f"\n📊 QC Summary Report:")
        print(f"   • Samples checked: {all_qc_results['samples_checked']}")
        print(f"   • Successful loads: {all_qc_results['overall_stats']['successful_loads']}")
        print(f"   • Failed loads: {all_qc_results['overall_stats']['failed_loads']}")
        
        if all_qc_results['overall_stats']['common_matrix_sizes']:
            print(f"   • Common matrix sizes: {all_qc_results['overall_stats']['common_matrix_sizes']}")
        
        if all_qc_results['overall_stats']['common_slice_counts']:
            print(f"   • Common slice counts: {all_qc_results['overall_stats']['common_slice_counts']}")
        
        return all_qc_results

def plot_3d_volume_slices(volume: np.ndarray, title: str = "3D Volume", 
                         save_path: Optional[Path] = None) -> None:
    """Plot multiple slices of a 3D volume for visualization.
    
    Args:
        volume: 3D numpy array
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title}\nShape: {volume.shape}, Range: [{volume.min():.3f}, {volume.max():.3f}]')
    
    # Show different axial slices
    depth = volume.shape[0]
    slices = [depth//4, depth//2, 3*depth//4]
    
    for i, slice_idx in enumerate(slices):
        axes[0, i].imshow(volume[slice_idx], cmap='gray')
        axes[0, i].set_title(f'Axial slice {slice_idx}')
        axes[0, i].axis('off')
    
    # Show sagittal and coronal views
    axes[1, 0].imshow(volume[:, volume.shape[1]//2, :], cmap='gray')
    axes[1, 0].set_title('Sagittal view')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(volume[:, :, volume.shape[2]//2], cmap='gray')
    axes[1, 1].set_title('Coronal view')
    axes[1, 1].axis('off')
    
    # Histogram
    axes[1, 2].hist(volume.flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('Intensity Distribution')
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory
    else:
        plt.show()
