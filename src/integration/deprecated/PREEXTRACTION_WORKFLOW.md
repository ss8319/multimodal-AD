# Protein Latent Pre-extraction Workflow

## Problem
The trained protein models (MLP/Transformer) were saved with a newer NumPy version, causing compatibility issues when loading in the `brainiac` conda environment:
```
ModuleNotFoundError: No module named 'numpy._core'
```

## Solution
Pre-extract protein latents in the `multimodal` uv environment, then load them in the `brainiac` environment for training.

---

## Step 1: Pre-extract Protein Latents (in `multimodal` uv env)

```bash
# Activate multimodal environment
cd /home/ssim0068/multimodal-AD
source .venv/bin/activate  # or however you activate uv env

# Run pre-extraction script
python src/integration/preextract_protein_latents.py
```

**This will:**
- Load the trained MLP model from `src/protein/runs/run_20251003_133215/models/neural_network.pkl`
- Extract `hidden_layer_2` latents for all samples in train.csv and test.csv
- Save to `/home/ssim0068/data/multimodal-dataset/protein_latents/`
  - `train_protein_latents.npy` (shape: [n_train, latent_dim])
  - `test_protein_latents.npy` (shape: [n_test, latent_dim])
  - `train_subjects.npy`, `test_subjects.npy` (for reference)
  - `train_labels.npy`, `test_labels.npy` (for reference)
  - `metadata.json` (model info)

---

## Step 2: Train Fusion Model (in `brainiac` conda env)

```bash
# Activate brainiac environment
conda activate brainiac

# Run training directly
cd /home/ssim0068/multimodal-AD
python src/integration/train_fusion.py

# OR submit SLURM job
sbatch train_fusion_slurm.sh
```

**The training script will:**
- Load pre-extracted protein latents from `/home/ssim0068/data/multimodal-dataset/protein_latents/`
- Extract MRI latents on-the-fly using BrainIAC
- Concatenate protein + MRI latents
- Train fusion classifier

---

## Configuration

### In `train_fusion.py`:
```python
config = {
    'protein_run_dir': None,  # Disable on-the-fly extraction
    'protein_latents_dir': '/home/ssim0068/data/multimodal-dataset/protein_latents',
    'protein_model_type': 'mlp',
    'protein_layer': 'hidden_layer_2',
    ...
}
```

### To switch to Transformer latents:
1. Update `preextract_protein_latents.py`:
   ```python
   protein_model_path = ".../protein_transformer.pth"
   layer_name = "transformer_embeddings"
   ```
2. Re-run Step 1
3. Update `train_fusion.py`:
   ```python
   'protein_model_type': 'transformer',
   'protein_layer': 'transformer_embeddings',
   ```

---

## File Structure

```
/home/ssim0068/
├── multimodal-AD/
│   ├── src/integration/
│   │   ├── preextract_protein_latents.py  # Step 1 script
│   │   ├── train_fusion.py                # Step 2 script
│   │   └── multimodal_dataset.py          # Supports pre-extracted latents
│   └── train_fusion_slurm.sh              # SLURM job for Step 2
└── data/
    └── multimodal-dataset/
        ├── train.csv
        ├── test.csv
        └── protein_latents/                # Pre-extracted latents
            ├── train_protein_latents.npy
            ├── test_protein_latents.npy
            └── metadata.json
```

---

## Advantages
✅ No NumPy version conflicts  
✅ Faster training (no on-the-fly protein inference)  
✅ Reproducible latents  
✅ Can easily switch between MLP/Transformer latents

## Future Work
Once you confirm good results, consider:
1. Merging environments (install all deps in single env)
2. Saving models with compatible NumPy versions
3. Using ONNX for cross-environment model deployment



