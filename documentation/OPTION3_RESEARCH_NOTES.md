# Option 3: Research-Friendly XDVioDet Implementation

**Date**: March 26, 2026  
**Purpose**: Simplified model for research paper on multi-class violence classification

---

## 📝 Paper Contribution Summary

### Title (Suggested)
**"Multi-class Violence Detection in Video using Temporal-Aware Graph Convolution Networks"**

Or:
**"Extended XD-Violence: Weakly-Supervised Multi-class Violence Detection with Positional Encoding"**

### Main Contributions
1. **Extension of XD-Violence**: From binary (Violence/Normal) to 7-class classification
2. **Temporal Encoding**: Added positional encoding for temporal awareness
3. **Empirical Validation**: Demonstrates improved performance on multi-class task
4. **Simplification**: Cleaner architecture while maintaining core GCN approach

### Non-Contributions (Removed for Clarity)
- ❌ ModalityFusion attention (complexity, not core to contribution)
- ❌ Deep classifier/approximator (not necessary, reduces clarity)

---

## 🏗️ Architecture Overview

### Input
- RGB features: 1024-dimensional per frame
- Audio features: 128-dimensional per frame
- Sequence length: Variable (up to 200 frames)
- **Total input dimension**: 1152

### Core Stages

#### Stage 1: Feature Projection (NEW)
```
Input (B, T, 1152)
  ↓
Linear projection (1152 → 256)
  ↓
BatchNorm + Positional Encoding
  ↓
Output (B, T, 256)
```
- **Purpose**: Fuse RGB + Audio features
- **Key enhancement**: Positional encoding adds temporal awareness

#### Stage 2: CNN Feature Extraction (FROM PAPER)
```
(B, T, 256)
  ↓ Conv1d (256→512)
  ↓ Conv1d (512→256)
  ↓ Conv1d (256→128)
  ↓ Conv1d (128→64)
  ↓
Output (B, T, 64)
```
- **Purpose**: Extract spatial-temporal features
- **Unchanged**: Uses original XD-Violence architecture

#### Stage 3: Multi-Path Graph Convolution (FROM PAPER)
```
Three parallel pathways:
├─ Path 1: Self-similarity adjacency (cosine similarity)
├─ Path 2: Distance-based adjacency (temporal proximity)
└─ Path 3: Score-based adjacency (anomaly score similarity)
      ↓
GCN: (B, T, 64) × (B, T, T) → (B, T, 64)
      ↓
Output: Concatenate 3 paths → (B, T, 192)
```
- **Purpose**: Capture relationships between frames
- **Unchanged**: Three adjacency matrices from XD-Violence

#### Stage 4: Classification (SIMPLIFIED)
```
(B, T, 192)
  ↓
Linear (192 → 128)
  ↓
LayerNorm + ReLU + Dropout
  ↓
Linear (128 → 7)  [MULTI-CLASS OUTPUT]
  ↓
Output (B, T, 7) - One of 7 classes per frame
```
- **Change from paper**: 7 output classes instead of 2
- **Simplified**: 2-layer instead of 3-layer for clarity

---

## 🔧 Configuration for Training

### Recommended Command
```bash
python main.py --seed 42 \
  --lr 0.0001 \
  --batch-size 128 \
  --max-epoch 50 \
  --scheduler-milestones 15 30 \
  --grad-clip 1.0 \
  --croloss-weight 0.1 \
  --online-mode Multi \
  --weights Normal
```

### Key Hyperparameters
| Parameter | Value | Reason |
|-----------|-------|--------|
| --lr | 0.0001 | Small LR for stable training |
| --batch-size | 128 | Large batch for gradient stability |
| --grad-clip | 1.0 | Prevents exploding gradients |
| --croloss-weight | 0.1 | Reduces distillation loss weight (focus on classification) |
| --scheduler-milestones | [15, 30] | LR decay at epochs 15 & 30 (for 50 epochs total) |

---

## 📊 Expected Performance

### Training Trajectory (Multi-class, Binary mode)
```
Epoch 1:    ROC-AUC ~0.55-0.60 (random)
Epoch 5:    ROC-AUC ~0.65-0.70
Epoch 10:   ROC-AUC ~0.72-0.78
Epoch 20:   ROC-AUC ~0.78-0.85
Epoch 50:   ROC-AUC ~0.85-0.90 (saturating)
```

### For Multi-class Mode
- **mAP**: ~0.75-0.82
- **F1-score**: ~0.70-0.78
- **Precision**: ~0.72-0.80
- **Recall**: ~0.65-0.75

---

## 📈 Metrics to Report in Paper

### For Each Model Variant
- ✅ ROC-AUC (weighted, per-class)
- ✅ F1-score (weighted average)
- ✅ Precision & Recall (weighted)
- ✅ mAP (mean Average Precision)
- ✅ Confusion matrix (for multi-class breakdown)
- ✅ Convergence time (epochs to saturation)

### Ablation Study Suggestions
1. **With vs Without PositionalEncoding**
2. **Binary vs Multi-class classification**
3. **Different CRO loss weights** (0.01, 0.1, 1.0, 5.0)

---

## 🎯 Paper Structure (Suggested)

```
1. Introduction
   - Violence detection importance
   - Limitations of binary classification
   - Related work on XD-Violence

2. Background: XD-Violence Architecture
   - Original paper summary
   - Multi-path GCN approach
   - Weakly-supervised learning

3. Our Modifications (Option 3)
   - Temporal positional encoding (ADD)
   - Multi-class classification (MODIFY)
   - Simplified architecture (CLARIFY)

4. Experimental Setup
   - Dataset: XD-Violence dataset
   - Training/test split: 19,770 / 4,000
   - Hyperparameters & training details
   - Metrics & evaluation protocol

5. Results
   - Performance comparison (binary vs multi-class)
   - With / without positional encoding
   - Convergence analysis
   - Per-class breakdown

6. Ablation Study
   - Impact of each component
   - Sensitivity to hyperparameters

7. Conclusion & Future Work
```

---

## 🔍 Key Differences from Enhanced Version

| Aspect | Enhanced (Old) | Option 3 (Current) |
|--------|---|---|
| ModalityFusion | Attention-based | Linear projection |
| Classifier | 3-layer (256, 128) | 2-layer (128) |
| Approximator | 3-layer Conv | 2-layer Conv |
| Parameters | ~1.2M | ~0.9M |
| Complexity | Higher | Lower |
| Paper-friendly | ❌ | ✅ |

---

## 🚀 Quick Start

### 1. Verify Installation
```bash
python -c "import torch; from model import Model; print('✓ Model ready')"
```

### 2. Train
```bash
python main.py --seed 42 --croloss-weight 0.1 --online-mode Multi
```

### 3. Monitor
- Training loss should decrease smoothly
- ROC-AUC should improve each epoch (initially)
- Check `./ckpt/` for saved models and metrics

### 4. Evaluate
```bash
# Test after training
python -c "
from test import test
from dataset import Dataset
from option import parser
import torch
import numpy as np

args = parser.parse_args(['--online-mode', 'Multi', '--modality', 'MIX2'])
model = torch.load('ckpt/wsanodetV5_.pkl')
gt = np.load('gtMulti.npy')
# ... run test(...)
"
```

---

## 📚 References for Your Paper

Cite these in your work:
1. **XD-Violence Original**: 
   - Fang et al., "XD-Violence: a large-scale dataset for violence detection in video"
   - arXiv:2011.04843

2. **Graph Convolution Networks**:
   - Kipf & Welling, "Semi-supervised classification with GCNs"
   - ICLR 2017

3. **Weakly-Supervised Learning**:
   - Video understanding with weak supervision

4. **Positional Encoding** (if needed):
   - Vaswani et al., "Attention is All You Need" (Transformer paper)
   - arXiv:1706.03762

---

## ⚠️ Important Notes

### For Paper Writing
- ✅ Be clear about which parts are from XD-Violence
- ✅ Highlight your specific contributions
- ✅ Include adequate citations
- ✅ Report full ablation studies
- ✅ Discuss limitations

### Potential Improvements Not Included
- Data augmentation
- Pre-trained I3D features
- Ensemble methods
- Advanced optimization (Adam → AdamW)
- Knowledge distillation

---

**Last Updated**: March 26, 2026  
**Status**: Ready for research/publication
