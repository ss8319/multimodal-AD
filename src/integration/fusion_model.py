import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusionClassifier(nn.Module):
    """
    Simple concatenation-based fusion classifier
    
    IMPROVED ARCHITECTURE (addresses dimension imbalance):
        Protein features (P) + MRI features (768) 
        → Project to shared_dim each
        → Concat (shared_dim*2) → FCNN → Output (2)
    
    Why projection helps:
    - Direct concat [64, 768] gives 92% weight to MRI (dominates gradients)
    - Projection [64→256] + [768→256] balances information flow
    - Both modalities now have equal "voting power"
    """
    
    def __init__(self, protein_dim, mri_dim=768, shared_dim=256, hidden_dim=256, 
                 num_classes=2, dropout=0.3):
        """
        Args:
            protein_dim: Dimension of protein features
            mri_dim: Dimension of MRI features (default: 768 from BrainIAC)
            shared_dim: Dimension to project both modalities to (default: 256)
            hidden_dim: Hidden layer dimension in classifier
            num_classes: Number of output classes (2 for AD/CN)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.shared_dim = shared_dim
        
        # Project protein features to shared dimension
        # Use LayerNorm for stable training across different input scales
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project MRI features to shared dimension
        # MRI already has 768 dims, so this acts as compression+projection
        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Now classifier gets balanced input [shared_dim * 2]
        fused_dim = shared_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"SimpleFusionClassifier (IMPROVED with feature projection):")
        print(f"  Protein: {protein_dim} → {shared_dim}")
        print(f"  MRI: {mri_dim} → {shared_dim}")
        print(f"  Fused input: {fused_dim} (balanced: {shared_dim}+{shared_dim})")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: Fused features [batch_size, protein_dim + mri_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Split modalities
        protein_x = x[:, :self.protein_dim]          # [B, protein_dim]
        mri_x = x[:, self.protein_dim:]               # [B, mri_dim]
        
        # Project to shared dimension
        protein_proj = self.protein_proj(protein_x)  # [B, shared_dim]
        mri_proj = self.mri_proj(mri_x)              # [B, shared_dim]
        
        # Concatenate balanced projections
        fused = torch.cat([protein_proj, mri_proj], dim=1)  # [B, shared_dim*2]
        
        # Classify
        return self.classifier(fused)


def get_model(protein_dim, mri_dim=768, shared_dim=256, hidden_dim=256, 
              num_classes=2, dropout=0.3):
    """
    Factory function to create improved fusion model with feature projection
    
    Args:
        protein_dim: Dimension of protein features
        mri_dim: Dimension of MRI features
        shared_dim: Dimension to project both modalities to
        hidden_dim: Hidden layer dimension
        num_classes: Number of classes
        dropout: Dropout rate
    
    Returns:
        model: SimpleFusionClassifier with balanced feature projection
    """
    return SimpleFusionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )


class WeightedFusionAttentionClassifier(nn.Module):
    """
    Attention-based weighted fusion classifier.
    - Projects protein and MRI features to a shared fusion_dim
    - Learns sample-specific scalar weights for each modality (softmax over 2)
    - Fuses by weighted sum of modality embeddings
    - Classifies with an FCNN head
    """
    
    def __init__(self, protein_dim, mri_dim=768, fusion_dim=128,
                 hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.fusion_dim = fusion_dim
        
        # Modality projections to a shared space
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention over modalities (2-way softmax)
        self.attention_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 2)
        )
        
        # Classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print("WeightedFusionAttentionClassifier:")
        print(f"  Protein dim: {protein_dim}")
        print(f"  MRI dim: {mri_dim}")
        print(f"  Fusion dim: {fusion_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: Concatenated features [batch_size, protein_dim + mri_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Split modalities
        protein_x = x[:, :self.protein_dim]
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]
        
        # Project to shared space
        p_embed = self.protein_proj(protein_x)  # [B, fusion_dim]
        m_embed = self.mri_proj(mri_x)          # [B, fusion_dim]
        
        # Attention weights over modalities
        att_input = torch.cat([p_embed, m_embed], dim=1)  # [B, 2*fusion_dim]
        att_logits = self.attention_mlp(att_input)        # [B, 2]
        att_weights = F.softmax(att_logits, dim=1)        # [B, 2]
        w_p = att_weights[:, 0].unsqueeze(1)
        w_m = att_weights[:, 1].unsqueeze(1)
        
        # Weighted fusion (same dimension as embeddings)
        fused = w_p * p_embed + w_m * m_embed             # [B, fusion_dim]
        
        # Classify
        logits = self.classifier(fused)
        return logits


def get_weighted_fusion_model(protein_dim, mri_dim=768, fusion_dim=128,
                              hidden_dim=128, num_classes=2, dropout=0.3):
    """
    Factory for the attention-based weighted fusion model.
    """
    return WeightedFusionAttentionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        fusion_dim=fusion_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
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

    # Test weighted attention fusion model
    print("\n" + "="*60)
    print("2. WEIGHTED ATTENTION FUSION MODEL")
    print("="*60)
    w_fusion_model = get_weighted_fusion_model(protein_dim=protein_dim, mri_dim=mri_dim, fusion_dim=128)
    w_output = w_fusion_model(fused_input)
    print(f"Weighted fusion output shape: {w_output.shape}")

