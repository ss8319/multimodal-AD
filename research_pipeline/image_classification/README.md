# 3D MRI Classification Pipeline

A clean, academic-focused implementation of 3D MRI classification using various deep learning architectures including CNNs, Vision Transformers, and BM-MAE.

## 🏗️ Architecture

This pipeline follows a simplified academic structure with clear separation of concerns:

```
image_classification/
├── __init__.py           # Module initialization and exports
├── config.py             # Configuration management (YAML-based)
├── dataset.py            # MRI dataset loading and preprocessing
├── models.py             # All model architectures in one file
├── training.py           # Training loops and loss functions
├── evaluation.py         # Metrics and visualization
├── main.py               # Main training orchestration
├── example_config.yaml   # Example configuration file
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Create Example Configuration
```bash
cd research_pipeline/image_classification
python main.py --create-config
```

### 2. Edit Configuration
Edit `example_config.yaml` to customize your experiment:
- Choose models to train
- Set hyperparameters
- Configure data paths
- Enable/disable features

### 3. Run Training
```bash
python main.py
```

## 📋 Configuration

The pipeline uses YAML-based configuration with sensible defaults. Key sections:

### Training Configuration
- **Methods**: Choose from `resnet3d`, `alexnet3d`, `convnext3d`, `vit3d`, `bmmae_frozen`, `bmmae_finetuned`
- **Hyperparameters**: Epochs, learning rate, batch size, weight decay
- **Regularization**: Early stopping, gradient clipping
- **Augmentation**: Rotation, intensity scaling, noise

### Data Configuration
- **Paths**: ADNI data and splits folder locations
- **Preprocessing**: Normalization method (zscore, minmax, percentile)
- **Target Size**: 3D volume dimensions (default: 128x128x128)

### Model Configuration
- **BM-MAE**: Pretrained weights path
- **CNN**: Dropout rates
- **Transformer**: Patch size and embedding dimensions

## 🔧 Available Models

### 3D CNNs
- **ResNet3D**: 3D ResNet architecture adapted from torchvision
- **AlexNet3D**: 3D AlexNet-style architecture
- **ConvNeXt3D**: Modern 3D ConvNeXt implementation

### Vision Transformers
- **ViT3D**: 3D Vision Transformer using PyTorch components
- **BM-MAE Frozen**: Brain Masked Autoencoder with frozen encoder
- **BM-MAE Fine-tuned**: BM-MAE with trainable encoder

## 📊 Features

### Data Handling
- **DICOM Loading**: Robust 3D volume reconstruction from DICOM files
- **Preprocessing**: Multiple normalization methods and intelligent resizing
- **Augmentation**: Rotation, intensity scaling, and noise addition
- **Validation Split**: Automatic train/validation splitting

### Training
- **Focal Loss**: Handles class imbalance effectively
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: ReduceLROnPlateau with configurable parameters
- **Gradient Clipping**: Prevents exploding gradients
- **Weight Decay**: Different rates for frozen vs. fine-tuned models

### Evaluation
- **Comprehensive Metrics**: Accuracy, AUC, precision, recall, F1-score
- **Visualization**: Confusion matrices, ROC curves, prediction distributions
- **Results Export**: CSV files for further analysis
- **Training History**: Loss and accuracy plots over epochs

## 📁 Data Structure

Expected data organization:
```
AD_CN/
├── proteomics/
│   └── MRI/
│       ├── ADNI/                    # DICOM files
│       └── splits/                  # Data splits
│           ├── train_split.csv      # Training data
│           └── test_split.csv       # Test data
```

## 🔍 Usage Examples

### Basic Training
```bash
# Train with default configuration
python main.py

# Train with custom config
python main.py
# Enter config file path when prompted
```

### Configuration Management
```bash
# Create example configuration
python main.py --create-config

# Show help
python main.py --help
```

### Model Selection
Edit `example_config.yaml`:
```yaml
training:
  methods: ["resnet3d", "bmmae_frozen"]  # Train specific models
  epochs: 20                              # Custom epochs
  batch_size: 4                           # Custom batch size
```

## 🎯 Academic Benefits

### Code Organization
- **Single Responsibility**: Each file has one clear purpose
- **Easy Navigation**: 7 files vs. complex nested structures
- **Quick Iteration**: Change one component without affecting others
- **Clear Dependencies**: Explicit imports and minimal abstractions

### Research Workflow
- **Reproducibility**: YAML configuration files
- **Experiment Tracking**: Comprehensive logging and saving
- **Easy Comparison**: Train multiple models with same settings
- **Debugging**: Extensive logging and visualization

### Collaboration
- **Simple Sharing**: Easy to share individual files
- **Clear Interfaces**: Well-documented function signatures
- **Minimal Dependencies**: Each module can run independently
- **Academic Standards**: Code that's easy to understand and extend

## 🚧 Dependencies

### Required
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- pydicom
- PyYAML

### Optional
- BM-MAE (for BM-MAE models)
- timm (for advanced ConvNeXt)
- scipy (for better image resizing)

## 🔄 Migration from Old Code

This pipeline replaces the monolithic `image_baseline.py` with a clean, modular structure:

| Old Structure | New Structure |
|---------------|---------------|
| `image_baseline.py` | `main.py` + modules |
| Command-line args | YAML configuration |
| Hardcoded paths | Configurable paths |
| Mixed concerns | Separated concerns |

### Benefits of Migration
- **Easier debugging**: Test components independently
- **Better organization**: Clear module boundaries
- **Faster iteration**: Change one component at a time
- **Academic quality**: Professional, research-grade code

## 📚 Future Extensions

This modular structure makes it easy to extend:

### New Models
- Add new architectures to `models.py`
- Register them in the model factory
- Update configuration options

### New Data Types
- Extend `dataset.py` for new modalities
- Add preprocessing functions
- Maintain consistent interfaces

### New Training Strategies
- Add new loss functions to `training.py`
- Implement new optimizers
- Add custom callbacks

## 🤝 Contributing

When adding new features:
1. **Maintain simplicity**: Keep the flat structure
2. **Single responsibility**: Each file should do one thing well
3. **Clear interfaces**: Document function signatures
4. **Academic focus**: Prioritize clarity over complexity

## 📄 License

This code is part of the AD/CN classification research project.
