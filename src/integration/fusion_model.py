import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusionClassifier(nn.Module):
    """
    Simple concatenation-based fusion classifier
    
    Architecture:
        Protein features (P) + MRI features (768) → Concat (P+768) → FCNN → Output (2)
    """
    
    def __init__(self, protein_dim, mri_dim=768, hidden_dim=128, 
                 num_classes=2, dropout=0.3):
        """
        Args:
            protein_dim: Dimension of protein features
            mri_dim: Dimension of MRI features (default: 768 from BrainIAC)
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (2 for AD/CN)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.input_dim = protein_dim + mri_dim
        
        # Simple FCNN classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"SimpleFusionClassifier:")
        print(f"  Input dim: {self.input_dim} (protein={protein_dim} + mri={mri_dim})")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: Fused features [batch_size, protein_dim + mri_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(x)

def get_model(protein_dim, mri_dim=768, hidden_dim=128, 
              num_classes=2, dropout=0.3):
    """
    Factory function to create fusion model
    
    Args:
        protein_dim: Dimension of protein features
        mri_dim: Dimension of MRI features
        hidden_dim: Hidden layer dimension
        num_classes: Number of classes
        dropout: Dropout rate
    
    Returns:
        model: SimpleFusionClassifier
    """
    return SimpleFusionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the models
    print("Testing fusion models...\n")
    
    batch_size = 4
    protein_dim = 100  # Example
    mri_dim = 768
    
    # Test fusion model
    print("="*60)
    print("1. FUSION MODEL")
    print("="*60)
    fusion_model = get_model(protein_dim=protein_dim)
    
    # Test forward pass
    fused_input = torch.randn(batch_size, protein_dim + mri_dim)
    output = fusion_model(fused_input)
    print(f"Input shape: {fused_input.shape}")
    print(f"Output shape: {output.shape}")

