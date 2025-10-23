"""
Debug script to check attention model training
"""
import numpy as np
import torch
from model import ProteinAttentionPooling

# Create a small synthetic dataset
n_samples = 100
n_features = 50
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, 2, n_samples)

# Create model
model = ProteinAttentionPooling(n_features=n_features, d_model=32, dropout=0.2)

# Check forward pass
X_tensor = torch.FloatTensor(X[:10])
logits, attention = model(X_tensor, return_attention=True)

print("=" * 60)
print("ATTENTION MODEL DIAGNOSTICS")
print("=" * 60)
print(f"\nInput shape: {X_tensor.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Attention weights shape: {attention.shape}")

print(f"\nLogits stats:")
print(f"  Mean: {logits.mean().item():.4f}")
print(f"  Std: {logits.std().item():.4f}")
print(f"  Min: {logits.min().item():.4f}")
print(f"  Max: {logits.max().item():.4f}")

print(f"\nAttention weights stats (should sum to 1.0 per sample):")
print(f"  Mean: {attention.mean().item():.6f}")
print(f"  Std: {attention.std().item():.6f}")
print(f"  Min: {attention.min().item():.6f}")
print(f"  Max: {attention.max().item():.6f}")
print(f"  Sum per sample (should be ~1.0): {attention.sum(dim=1)[:5]}")

# Check if attention is collapsing (too concentrated)
max_attention_per_sample = attention.max(dim=1)[0]
print(f"\nMax attention per sample (if >0.5, might be collapsing):")
print(f"  Mean: {max_attention_per_sample.mean().item():.4f}")
print(f"  Max: {max_attention_per_sample.max().item():.4f}")

# Check predictions
probs = torch.softmax(logits, dim=1)
preds = probs.argmax(dim=1)
print(f"\nPredictions:")
print(f"  Class 0: {(preds == 0).sum().item()}/{len(preds)}")
print(f"  Class 1: {(preds == 1).sum().item()}/{len(preds)}")

print(f"\nProbabilities stats:")
print(f"  Mean prob class 0: {probs[:, 0].mean().item():.4f}")
print(f"  Mean prob class 1: {probs[:, 1].mean().item():.4f}")

# Test training for a few steps with class weights
unique_labels, counts = np.unique(y[:10], return_counts=True)
class_weights = torch.FloatTensor([counts.sum() / (len(unique_labels) * c) for c in counts])
print(f"\nClass weights: {class_weights.numpy()}")

criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Initialize final layer bias to reflect class distribution
if hasattr(model.classifier[-1], 'bias') and model.classifier[-1].bias is not None:
    class_priors = counts / counts.sum()
    model.classifier[-1].bias.data = torch.log(torch.FloatTensor(class_priors))
    print(f"Class priors: {class_priors}")
    print(f"Initialized final bias: {model.classifier[-1].bias.data.numpy()}")

print(f"\n" + "=" * 60)
print("TRAINING TEST (10 steps)")
print("=" * 60)

y_tensor = torch.LongTensor(y[:10])
print(f"True labels: {y_tensor.numpy()}")
print(f"Class distribution in batch: Class 0: {(y_tensor==0).sum().item()}, Class 1: {(y_tensor==1).sum().item()}")

for step in range(20):
    optimizer.zero_grad()
    logits = model(X_tensor)
    loss = criterion(logits, y_tensor)
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    optimizer.step()
    
    preds = logits.argmax(dim=1)
    acc = (preds == y_tensor).float().mean().item()
    
    # Check predictions per class
    probs = torch.softmax(logits, dim=1)
    
    if step % 4 == 0:
        print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc:.2f}, GradNorm={total_grad_norm:.4f}, "
              f"Pred[0]={preds[0].item()}, Prob0={probs[0,0].item():.3f}")

print("\n" + "=" * 60)
print("If you see:")
print("  - Loss decreasing: ✓ Model is learning")
print("  - Accuracy changing: ✓ Model is not collapsed")
print("  - GradNorm > 0: ✓ Gradients are flowing")
print("  - Max attention < 0.5: ✓ Attention is distributed")
print("=" * 60)


