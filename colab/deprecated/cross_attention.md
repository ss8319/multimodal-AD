# Cross-Attention Transformer Architecture for Multimodal AD Classification

## Complete Architecture Flow Diagram

```mermaid
graph TD
    %% Input Layer
    A1["MRI Patches<br/>[B, 64, 768]<br/>From BM-MAE"] --> B1["Linear Projection<br/>[B, 64, 256]"]
    A2["Proteomic Features<br/>[B, 8, 8]<br/>From Autoencoder/TabNet"] --> B2["Linear Projection<br/>[B, 8, 256]"]
    
    %% Positional Encoding
    B1 --> C1["3D Spatial Encoding<br/>Brain coordinates (x,y,z)<br/>→ Embedding space"]
    B2 --> C2["Categorical Encoding<br/>Protein pathways<br/>→ Embedding space"]
    
    %% Self-Attention (PyTorch Built-in)
    C1 --> D1["MRI Self-Attention<br/>nn.TransformerEncoderLayer<br/>Patches attend to patches"]
    C2 --> D2["Protein Self-Attention<br/>nn.TransformerEncoderLayer<br/>Proteins attend to proteins"]
    
    %% Cross-Attention (Custom Orchestration)
    D1 --> E1["MRI → Protein<br/>Cross-Attention<br/>nn.MultiheadAttention"]
    D2 --> E2["Protein → MRI<br/>Cross-Attention<br/>nn.MultiheadAttention"]
    
    %% Bidirectional Flow
    E1 --> F1["Updated MRI Features<br/>[B, 64, 256]"]
    E2 --> F2["Updated Protein Features<br/>[B, 8, 256]"]
    
    %% Second Cross-Attention Layer
    F1 --> G1["MRI → Protein<br/>Layer 2"]
    F2 --> G2["Protein → MRI<br/>Layer 2"]
    
    G1 --> H1["Final MRI Features<br/>[B, 64, 256]"]
    G2 --> H2["Final Protein Features<br/>[B, 8, 256]"]
    
    %% Global Pooling
    H1 --> I1["Global Average Pool<br/>[B, 256]"]
    H2 --> I2["Global Average Pool<br/>[B, 256]"]
    
    %% Fusion and Classification
    I1 --> J["Concatenate<br/>[B, 512]"]
    I2 --> J
    J --> K["MLP Classifier<br/>512 → 256 → 128 → 2"]
    K --> L["Output Logits<br/>[B, 2]<br/>AD vs CN"]
    
    %% Attention Weights for Visualization
    E1 -.-> M1["Attention Weights<br/>[B, H, 64, 8]<br/>MRI-to-Protein"]
    E2 -.-> M2["Attention Weights<br/>[B, H, 8, 64]<br/>Protein-to-MRI"]
    G1 -.-> M3["Layer 2 Weights<br/>[B, H, 64, 8]"]
    G2 -.-> M4["Layer 2 Weights<br/>[B, H, 8, 64]"]
    
    %% Styling
    style A1 fill:#e3f2fd
    style A2 fill:#f3e5f5
    style E1 fill:#fff3e0
    style E2 fill:#fff3e0
    style M1 fill:#e8f5e8
    style M2 fill:#e8f5e8
    style L fill:#ffebee
```

## Key Innovation: Cross-Modal Communication

The core innovation is the **bidirectional cross-attention** mechanism:

1. **MRI → Protein**: "Given this brain pattern, which proteins are most relevant?"
2. **Protein → MRI**: "Given this protein signature, which brain regions matter?"

This allows the model to discover **spatial-molecular associations** that are critical for understanding Alzheimer's disease mechanisms.

## Interpretability Through Attention

The attention weights provide direct insights into:
- Which brain regions correlate with which proteins
- How disease patterns differ from healthy patterns  
- What drives each individual prediction
- Novel biomarker combinations for diagnosis