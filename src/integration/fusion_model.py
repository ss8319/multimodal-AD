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
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]  # [B, mri_dim]
        
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
        protein_x = x[:, :self.protein_dim]  # [B, protein_dim]
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]     # [B, mri_dim]
        
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

class SimpleCrossModalAttentionClassifier(nn.Module):
    """
    Simplified cross-modal attention fusion classifier.
    
    SIMPLIFIED ARCHITECTURE RATIONALE:
    
    1. **Unidirectional Attention**: Only protein→MRI (not bidirectional)
       - Proteins guide which brain regions to examine
       - Simpler, fewer parameters, less risk of overfitting
    
    2. **Single-Head Attention**: No multi-head complexity
       - One attention pattern instead of multiple heads
       - Easier to interpret, more stable with small data
    
    3. **Strong Residual Connection**: Weighted combination of original + attended
       - `fused = alpha*original + (1-alpha)*attended` where alpha=0.7
       - Prevents attention from being too aggressive
    
    4. **Reduced Fusion Dimension**: 2*shared_dim instead of 4*shared_dim
       - [protein_emb, mri_attended] instead of all 4 combinations
       - Fewer parameters in classifier head
    
    5. **Smaller Embedding Size**: 64-dim instead of 128-dim
       - Reduces parameter count significantly
       - Better suited for small dataset (40 samples)
    
    Why this design for AD classification:
    - Still captures protein→brain attention (most important direction)
    - Much lower parameter count (better for small datasets)
    - Strong residual prevents attention from overfitting
    - More interpretable with single attention pattern
    """
    
    def __init__(self, protein_dim, mri_dim=768, shared_dim=64,
                 hidden_dim=128, num_classes=2, dropout=0.3, residual_alpha=0.7):
        """
        Args:
            protein_dim: Input dimension of protein features
            mri_dim: Input dimension of MRI features (default: 768 from BrainIAC)
            shared_dim: Shared embedding dimension (smaller than full model)
            hidden_dim: Hidden dimension in final classifier
            num_classes: Number of output classes
            dropout: Dropout rate
            residual_alpha: Weight of original features in residual connection
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.shared_dim = shared_dim
        self.residual_alpha = residual_alpha
        
        # Per-modality projections to shared embedding space
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple attention mechanism (no multi-head complexity)
        # Just a single MLP that computes attention scores
        self.attention = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 1)
        )
        
        # Final fusion classifier
        # Input: [protein_emb, mri_attended] = 2 * shared_dim
        fusion_input_dim = 2 * shared_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
        
        print(f"SimpleCrossModalAttentionClassifier:")
        print(f"  Protein: {protein_dim} → {shared_dim}")
        print(f"  MRI: {mri_dim} → {shared_dim}")
        print(f"  Attention: Single-head (protein→MRI)")
        print(f"  Residual strength: {residual_alpha:.1f}")
        print(f"  Fusion input: {fusion_input_dim} (2 × {shared_dim})")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output classes: {num_classes}")
        print(f"  Parameter count: ~{self._count_parameters():,} (vs ~{self._count_full_model_params(shared_dim):,} in full model)")
    
    def _count_parameters(self):
        """Count approximate number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _count_full_model_params(self, shared_dim):
        """Estimate parameters in full cross-modal model for comparison"""
        # This is an approximation of the full model's parameter count
        return self._count_parameters() * 3  # ~3x more parameters in full model
    
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: Concatenated features [batch_size, protein_dim + mri_dim]
            return_attention_weights: If True, return attention weights for analysis
        
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: (optional) Attention weights if requested
        """
        batch_size = x.size(0)

        # Split modalities
        protein_x = x[:, :self.protein_dim]
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]

        # Project to shared embedding space
        protein_emb = self.protein_proj(protein_x)   # [B, shared_dim]
        mri_emb = self.mri_proj(mri_x)               # [B, shared_dim]

        # Per-sample protein-conditioned gating of MRI embedding (no cross-sample attention)
        # Gate is a scalar in [0,1] per sample computed from concat([p_emb, m_emb])
        gate_logits = self.attention(torch.cat([protein_emb, mri_emb], dim=1)).squeeze(1)  # [B]
        gate = torch.sigmoid(gate_logits).unsqueeze(1)  # [B,1]

        # Apply gating: per-sample blend of MRI and protein embeddings in shared space
        # Intuition: when gate≈1 rely more on MRI, when gate≈0 blend toward protein signal
        mri_attended = gate * mri_emb + (1 - gate) * protein_emb  # [B, shared_dim]

        # Store gate as attention weights for interpretability
        attention_weights = gate  # [B,1]
        
        # Strong residual connection to prevent attention from being too aggressive
        mri_combined = self.residual_alpha * mri_emb + (1 - self.residual_alpha) * mri_attended
        
        # Store attention weights for interpretability
        self.last_attention_weights = attention_weights
        
        # Fusion: Concatenate protein embeddings with attended MRI
        fused = torch.cat([protein_emb, mri_combined], dim=1)  # [B, 2*shared_dim]
        
        # Final classification
        logits = self.classifier(fused)
        
        if return_attention_weights:
            return logits, attention_weights
        
        return logits
    
    def get_attention_analysis(self):
        """
        Get interpretable attention analysis from last forward pass.
        
        Returns:
            Dict with attention statistics for interpretability
        """
        if self.last_attention_weights is None:
            return None
            
        gate = self.last_attention_weights.squeeze(1)  # [B]
        # Binary entropy per sample: H(p) = -p log p - (1-p) log(1-p)
        gate_entropy = -(gate * torch.log(gate + 1e-10) + (1 - gate) * torch.log(1 - gate + 1e-10)).mean().item()
        return {
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
            'gate_max': gate.max().item(),
            'gate_min': gate.min().item(),
            'gate_binary_entropy': gate_entropy,
        }


class AsymmetricFusionClassifier(nn.Module):
    """
    Asymmetric fusion classifier - gives MRI more capacity than protein.
    
    DESIGN RATIONALE:
    - MRI has richer spatial information (768-dim) → deserves more capacity
    - Protein is sparser (64-dim) → needs modest expansion, not compression
    - Balanced final fusion without losing MRI's representational power
    
    Architecture:
    - Protein: 64 → 96 (modest expansion)
    - MRI: 768 → 160 (compression but retains more info than protein)
    - Fused: 96 + 160 = 256 (manageable for small dataset)
    """
    
    def __init__(self, protein_dim, mri_dim=768, 
                 protein_proj_dim=96, mri_proj_dim=160,
                 hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.protein_proj_dim = protein_proj_dim
        self.mri_proj_dim = mri_proj_dim
        
        # Asymmetric projections
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, protein_proj_dim),
            nn.BatchNorm1d(protein_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, mri_proj_dim),
            nn.BatchNorm1d(mri_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier on asymmetric fusion
        fused_dim = protein_proj_dim + mri_proj_dim
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
        
        print(f"AsymmetricFusionClassifier:")
        print(f"  Protein: {protein_dim} → {protein_proj_dim}")
        print(f"  MRI: {mri_dim} → {mri_proj_dim}")
        print(f"  Fused input: {fused_dim} ({protein_proj_dim}+{mri_proj_dim})")
        print(f"  Capacity ratio (MRI:Protein): {mri_proj_dim/protein_proj_dim:.1f}:1")
    
    def forward(self, x):
        # Split modalities
        protein_x = x[:, :self.protein_dim]
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]
        
        # Asymmetric projections
        protein_proj = self.protein_proj(protein_x)
        mri_proj = self.mri_proj(mri_x)
        
        # Concatenate and classify
        fused = torch.cat([protein_proj, mri_proj], dim=1)
        return self.classifier(fused)


def get_asymmetric_fusion_model(protein_dim, mri_dim=768, 
                               protein_proj_dim=96, mri_proj_dim=160,
                               hidden_dim=256, num_classes=2, dropout=0.3):
    """Factory function for asymmetric fusion model."""
    return AsymmetricFusionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        protein_proj_dim=protein_proj_dim,
        mri_proj_dim=mri_proj_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )


def get_simple_cross_modal_attention_model(protein_dim, mri_dim=768, shared_dim=64,
                                      hidden_dim=128, num_classes=2, dropout=0.3, residual_alpha=0.7):
    """
    Factory function for simplified cross-modal attention fusion model.
    
    Args:
        protein_dim: Dimension of protein features
        mri_dim: Dimension of MRI features
        shared_dim: Shared embedding dimension (smaller than full model)
        hidden_dim: Hidden dimension in classifier
        num_classes: Number of output classes
        dropout: Dropout rate
        residual_alpha: Weight of original features in residual connection
    
    Returns:
        SimpleCrossModalAttentionClassifier instance
    """
    return SimpleCrossModalAttentionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        residual_alpha=residual_alpha
    )


class CrossTransformerFusionClassifier(nn.Module):
    """
    Cross-transformer-based fusion classifier for multimodal AD classification.
    
    Architecture adapted from: An imaging and genetic-based deep learning network for Alzheimer's disease diagnosis
    
    Design:
    1. Project protein and MRI features to shared embedding dimension
    2. Add CLS tokens to each modality (learnable embeddings)
    3. Two cross-transformer blocks:
       - Block 1: Q from protein, K/V from MRI → f_protein→mri
       - Block 2: Q from MRI, K/V from protein → f_mri→protein
    4. Self-transformer block on cross-transformer outputs
    5. Extract CLS tokens from both modalities
    6. Concatenate CLS tokens → FC layer → classification
    
    Note: Since we have vector features (not sequences), we treat each vector as a 
    single-token sequence, then add CLS token to create [CLS, feature] sequences.
    """
    
    def __init__(
        self,
        protein_dim,
        mri_dim=768,
        embed_dim=256,
        num_heads=4,
        ff_dim=None,
        num_classes=2,
        dropout=0.1
    ):
        """
        Args:
            protein_dim: Input dimension of protein features
            mri_dim: Input dimension of MRI features
            embed_dim: Shared embedding dimension for both modalities
            num_heads: Number of attention heads (default: 4 as in paper)
            ff_dim: Feed-forward dimension (default: 4 * embed_dim)
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim
        
        # Project modalities to shared embedding space
        self.protein_proj = nn.Linear(protein_dim, embed_dim)
        self.mri_proj = nn.Linear(mri_dim, embed_dim)
        
        # CLS tokens (learnable embeddings) - one per modality
        self.protein_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mri_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-attention modules (for cross-modal attention)
        # Block 1: Q from protein, K/V from MRI
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Block 2: Q from MRI, K/V from protein
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms and feed-forward networks for cross-attention blocks
        self.cross_norm1_1 = nn.LayerNorm(embed_dim)
        self.cross_norm1_2 = nn.LayerNorm(embed_dim)
        self.cross_ffn_1 = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.cross_ffn_2 = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.cross_norm2_1 = nn.LayerNorm(embed_dim)
        self.cross_norm2_2 = nn.LayerNorm(embed_dim)
        
        # Self-transformer block (processes cross-transformer outputs)
        self.self_transformer_protein = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.self_transformer_mri = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Classification head
        classifier_input_dim = embed_dim * 2  # Concatenated CLS tokens
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize CLS tokens
        nn.init.normal_(self.protein_cls_token, std=0.02)
        nn.init.normal_(self.mri_cls_token, std=0.02)
        
        print(f"CrossTransformerFusionClassifier:")
        print(f"  Protein dim: {protein_dim} → {embed_dim}")
        print(f"  MRI dim: {mri_dim} → {embed_dim}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Attention heads: {num_heads}")
        print(f"  FFN dim: {self.ff_dim}")
        print(f"  Dropout: {dropout}")
        print(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: Concatenated features [batch_size, protein_dim + mri_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Split modalities
        protein_x = x[:, :self.protein_dim]  # [B, protein_dim]
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]  # [B, mri_dim]
        
        # Project to shared embedding space
        protein_emb = self.protein_proj(protein_x)  # [B, embed_dim]
        mri_emb = self.mri_proj(mri_x)              # [B, embed_dim]
        
        # Reshape to sequences: [B, 1, embed_dim] (single token per modality)
        protein_seq = protein_emb.unsqueeze(1)  # [B, 1, embed_dim]
        mri_seq = mri_emb.unsqueeze(1)          # [B, 1, embed_dim]
        
        # Add CLS tokens
        protein_cls = self.protein_cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        mri_cls = self.mri_cls_token.expand(batch_size, -1, -1)          # [B, 1, embed_dim]
        
        protein_with_cls = torch.cat([protein_cls, protein_seq], dim=1)  # [B, 2, embed_dim]
        mri_with_cls = torch.cat([mri_cls, mri_seq], dim=1)              # [B, 2, embed_dim]
        
        # Cross-transformer block 1: Q from protein, K/V from MRI
        # Pre-LN: normalize before attention
        protein_norm_1 = self.cross_norm1_1(protein_with_cls)
        mri_norm_1 = self.cross_norm1_1(mri_with_cls)
        
        # Cross-attention: protein queries attend to MRI keys/values
        protein_cross_attn, _ = self.cross_attn_1(
            query=protein_norm_1,
            key=mri_norm_1,
            value=mri_norm_1
        )
        # Residual connection
        protein_cross_1 = protein_with_cls + protein_cross_attn
        
        # Feed-forward
        protein_cross_1 = protein_cross_1 + self.cross_ffn_1(self.cross_norm2_1(protein_cross_1))
        
        # Cross-transformer block 2: Q from MRI, K/V from protein
        mri_norm_2 = self.cross_norm1_2(mri_with_cls)
        protein_norm_2 = self.cross_norm1_2(protein_with_cls)
        
        # Cross-attention: MRI queries attend to protein keys/values
        mri_cross_attn, _ = self.cross_attn_2(
            query=mri_norm_2,
            key=protein_norm_2,
            value=protein_norm_2
        )
        # Residual connection
        mri_cross_2 = mri_with_cls + mri_cross_attn
        
        # Feed-forward
        mri_cross_2 = mri_cross_2 + self.cross_ffn_2(self.cross_norm2_2(mri_cross_2))
        
        # Self-transformer blocks (intra-modality attention)
        protein_self = self.self_transformer_protein(protein_cross_1)  # [B, 2, embed_dim]
        mri_self = self.self_transformer_mri(mri_cross_2)               # [B, 2, embed_dim]
        
        # Extract CLS tokens (first token in each sequence)
        protein_cls_feat = protein_self[:, 0, :]  # [B, embed_dim]
        mri_cls_feat = mri_self[:, 0, :]          # [B, embed_dim]
        
        # Concatenate CLS tokens
        fused_cls = torch.cat([protein_cls_feat, mri_cls_feat], dim=1)  # [B, 2*embed_dim]
        
        # Classify
        logits = self.classifier(fused_cls)  # [B, num_classes]
        
        return logits
    


def get_cross_transformer_fusion_model(
    protein_dim,
    mri_dim=768,
    embed_dim=256,
    num_heads=4,
    ff_dim=None,
    num_classes=2,
    dropout=0.1
):
    """
    Factory function for cross-transformer fusion model.
    
    Args:
        protein_dim: Dimension of protein features
        mri_dim: Dimension of MRI features
        embed_dim: Shared embedding dimension
        num_heads: Number of attention heads (default: 4)
        ff_dim: Feed-forward dimension (default: 4 * embed_dim)
        num_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        CrossTransformerFusionClassifier instance
    """
    return CrossTransformerFusionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout
    )


class KroneckerProductFusionClassifier(nn.Module):
    """
    Kronecker product-based fusion classifier for multimodal AD classification.
    
    The Kronecker product captures all pairwise interactions between protein and MRI features.
    For vectors [a1, a2, ..., an] and [b1, b2, ..., bm], the Kronecker product gives:
    [a1*b1, a1*b2, ..., a1*bm, a2*b1, ..., an*bm] (flattened)
    
    This creates a feature space of dimension (protein_dim × mri_dim) that explicitly
    models all cross-modal interactions.
    
    Design:
    1. Compute Kronecker product of protein and MRI features
    2. Optionally project to lower dimension to reduce parameters
    3. Apply classifier with dropout regularization
    """
    
    def __init__(
        self,
        protein_dim,
        mri_dim=768,
        projected_dim=None,
        hidden_dim=256,
        num_classes=2,
        dropout=0.3
    ):
        """
        Args:
            protein_dim: Input dimension of protein features
            mri_dim: Input dimension of MRI features
            projected_dim: Dimension to project Kronecker product to (None = no projection)
            hidden_dim: Hidden layer dimension in classifier
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.mri_dim = mri_dim
        self.kronecker_dim = protein_dim * mri_dim
        self.projected_dim = projected_dim
        
        # Optional projection layer to reduce dimensionality
        if projected_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.kronecker_dim, projected_dim),
                nn.LayerNorm(projected_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = projected_dim
        else:
            self.projection = None
            classifier_input_dim = self.kronecker_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"KroneckerProductFusionClassifier:")
        print(f"  Protein dim: {protein_dim}")
        print(f"  MRI dim: {mri_dim}")
        print(f"  Kronecker product dim: {self.kronecker_dim} ({protein_dim} × {mri_dim})")
        if projected_dim is not None:
            print(f"  Projected to: {projected_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output classes: {num_classes}")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x):
        """
        Args:
            x: Concatenated features [batch_size, protein_dim + mri_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Split modalities
        protein_x = x[:, :self.protein_dim]  # [B, protein_dim]
        mri_x = x[:, self.protein_dim: self.protein_dim + self.mri_dim]  # [B, mri_dim]
        
        # Compute Kronecker product efficiently using broadcasting
        # Reshape for outer product: [B, protein_dim, 1] × [B, 1, mri_dim]
        protein_expanded = protein_x.unsqueeze(-1)  # [B, protein_dim, 1]
        mri_expanded = mri_x.unsqueeze(-2)          # [B, 1, mri_dim]
        
        # Outer product (equivalent to Kronecker product for vectors)
        kronecker = protein_expanded * mri_expanded  # [B, protein_dim, mri_dim]
        
        # Flatten to get Kronecker product vector
        kronecker_flat = kronecker.view(batch_size, -1)  # [B, protein_dim * mri_dim]
        
        # Optional projection to reduce dimensionality
        if self.projection is not None:
            fused = self.projection(kronecker_flat)  # [B, projected_dim]
        else:
            fused = kronecker_flat  # [B, kronecker_dim]
        
        # Classify
        logits = self.classifier(fused)  # [B, num_classes]
        
        return logits


def get_kronecker_product_fusion_model(
    protein_dim,
    mri_dim=768,
    projected_dim=None,
    hidden_dim=256,
    num_classes=2,
    dropout=0.3
):
    """
    Factory function for Kronecker product fusion model.
    
    Args:
        protein_dim: Dimension of protein features
        mri_dim: Dimension of MRI features
        projected_dim: Dimension to project Kronecker product to (None = no projection)
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        KroneckerProductFusionClassifier instance
    """
    return KroneckerProductFusionClassifier(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        projected_dim=projected_dim,
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

    # Test weighted attention fusion model
    print("\n" + "="*60)
    print("2. WEIGHTED ATTENTION FUSION MODEL")
    print("="*60)
    w_fusion_model = get_weighted_fusion_model(protein_dim=protein_dim, mri_dim=mri_dim, fusion_dim=128)
    w_output = w_fusion_model(fused_input)
    print(f"Weighted fusion output shape: {w_output.shape}")

    # Test simple cross-modal attention fusion model
    print("\n" + "="*60)
    print("3. SIMPLE CROSS-MODAL ATTENTION FUSION MODEL")
    print("="*60)
    simple_cross_attn_model = get_simple_cross_modal_attention_model(
        protein_dim=protein_dim, 
        mri_dim=mri_dim, 
        shared_dim=64,
        residual_alpha=0.7
    )
    simple_cross_attn_output = simple_cross_attn_model(fused_input)
    print(f"Simple cross-modal attention output shape: {simple_cross_attn_output.shape}")
    
    # Test with attention weights
    simple_cross_attn_output_with_weights, attention_weights = simple_cross_attn_model(fused_input, return_attention_weights=True)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test attention analysis
    attention_analysis = simple_cross_attn_model.get_attention_analysis()
    print(f"Attention analysis: {attention_analysis}")
    
    # Test cross-transformer fusion model
    print("\n" + "="*60)
    print("4. CROSS-TRANSFORMER FUSION MODEL")
    print("="*60)
    cross_transformer_model = get_cross_transformer_fusion_model(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        embed_dim=256,
        num_heads=4,
        num_classes=2,
        dropout=0.1
    )
    cross_transformer_output = cross_transformer_model(fused_input)
    print(f"Cross-transformer output shape: {cross_transformer_output.shape}")
    
    # Test Kronecker product fusion model
    print("\n" + "="*60)
    print("5. KRONECKER PRODUCT FUSION MODEL")
    print("="*60)
    kronecker_model = get_kronecker_product_fusion_model(
        protein_dim=protein_dim,
        mri_dim=mri_dim,
        projected_dim=512,  # Project from protein_dim*mri_dim to 512
        hidden_dim=256,
        num_classes=2,
        dropout=0.3
    )
    kronecker_output = kronecker_model(fused_input)
    print(f"Kronecker product output shape: {kronecker_output.shape}")
    print(f"  Kronecker product dimension: {protein_dim} × {mri_dim} = {protein_dim * mri_dim}")
    print(f"  Projected to: 512")

