# XDVioDet Model Architecture Guide (Option 3 - Simplified for Research Paper)

## 🎯 What is This Model?

**XDVioDet** is a weakly-supervised deep learning model for detecting and classifying violence in video sequences. It combines CNN feature extraction with Graph Convolutional Networks to capture both temporal and relational patterns in video data.

### Real-World Example
A security camera records a 200-frame video with RGB (1024D) and Audio (128D) features:
- Model outputs: Frame-level classification (7 classes: Normal + 6 violence types)

---

## 📊 Problem Statement

**Input**: Video sequences with multiple modalities (vision + audio)
- RGB features: What we see in the video (1024 dimensions)
- Audio features: What we hear (128 dimensions)

**Output**: 
- **Binary mode**: Is there violence? (Yes/No)
- **Multi-class mode**: What type of violence? (Normal, 6 violence types)

**Challenge**: 
- The model must understand temporal patterns (what happens over time)
- It must fuse information from different modalities (seeing + hearing)
- It must localize anomalies precisely in time

---

## 🏗️ Model Architecture Overview

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Video Sequence (200 frames)                                  │
│ ├─ RGB Features:   (200, 1024)  ← What we see                       │
│ └─ Audio Features: (200, 128)   ← What we hear                      │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: FEATURE FUSION & TEMPORAL ENCODING                         │
│ ├─ Attention-based Modality Fusion: Learn weights for RGB vs Audio  │
│ └─ Positional Encoding: Add time information to features            │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓ (200, 256)
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: FEATURE EXTRACTION (Convolutional Network)                 │
│ ├─ Conv1d: 256 → 512 dims + BatchNorm                               │
│ ├─ Conv1d: 512 → 256 dims + BatchNorm                               │
│ ├─ Conv1d: 256 → 128 dims + BatchNorm                               │
│ └─ Conv1d: 128 → 64 dims + BatchNorm                                │
└──────────────────────┬──────────────────────────────────────────────┘
                       ↓ (200, 64)
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                  ↓
┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ GCN Path 1      │ │ GCN Path 2       │ │ GCN Path 3       │
│ (Self-Similarity)│ │ (Distance-based) │ │ (Score-based)    │
│ Adjacency       │ │ Adjacency        │ │ Adjacency        │
│ Matrix 1        │ │ Matrix 2         │ │ Matrix 3         │
└────────┬────────┘ └────────┬─────────┘ └────────┬─────────┘
         ↓                   ↓                    ↓
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: GRAPH CONVOLUTION (3 Parallel Pathways)         │
│ Each pathway learns different relationships:             │
│ ├─ Path 1: Features similar to each other?              │
│ ├─ Path 2: Frames close in time? (temporal distance)    │
│ └─ Path 3: Frames with similar anomaly scores?          │
│ Output: (200, 64) + (200, 64) + (200, 64)               │
└────────────────┬─────────────────────────────────────────┘
                 ↓ Concatenate: (200, 192)
┌─────────────────────────────────────────────────────────┐
│ STAGE 4: CLASSIFICATION                                 │
│ Deep Classifier: (200, 192) → ... → (200, classes)      │
│ Output per frame: Probability of each class             │
└────────────────┬───────────────────────────────────────┘
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
    [Binary Output]  [Multi-class Output]
    (200, 1)         (200, 7)
    Violence?        Which type?
```

### Latest Model Stabilizations (Mar 26, 2026 - Option 3)
- **Simplified for research clarity**: Removed ModalityFusion, simplified classifier to 2 layers, simplified approximator
- **Kept from papers**: Core Conv+GCN architecture, 3-path graph convolution, weakly-supervised learning
- **Added enhancement**: Positional encoding for temporal awareness
- **Key modification**: Binary→Multi-class classification (7 classes: Normal + 6 violence types)

---

## 🔍 Detailed Step-by-Step Breakdown

### STAGE 1: Feature Fusion & Temporal Encoding

#### What Happens?
The raw features from RGB and Audio are combined and enriched with time information.

#### Why?
- Features from different sensors have different scales and meanings
- The model needs to know: "This is frame #50 out of 200" (temporal context)

#### Code Representation:
```python
# Input: (Batch=32, Time=200, Features=1152)  [1024 RGB + 128 Audio]
x = inputs  # Shape: (32, 200, 1152)

# Step 1: Learn which modality to emphasize
x_fused = modality_fusion(x)  # (32, 200, 256)

# Step 2: Add positional encoding (sin/cos waves for time)
x_encoded = positional_encoding(x_fused)  # (32, 200, 256)
# Now the model knows: frame 0 is different from frame 100
```

#### Real Example:
```
Frame 0:   [Red=0.8, Blue=0.2, Audio=0.5] → Fusion → [0.85]
           + Positional encoding (sin/cos of position 0)

Frame 100: [Red=0.8, Blue=0.2, Audio=0.5] → Same features!
           + Different Positional encoding (sin/cos of position 100)
           
→ Model can distinguish frame 0 from frame 100 even with same RGB/Audio
```

---

### STAGE 2: Feature Extraction (Convolutional Neural Network)

#### What Happens?
Convolutional layers learn increasingly abstract features from the fused input.

#### Why?
- Conv1d learns local patterns (what happens in small time windows)
- BatchNorm stabilizes learning and allows faster training
- Dropout prevents overfitting

#### Layer Breakdown:

```
Conv1d Block 1:
├─ Input: (32, 256, 200)
├─ Conv1d: 256 → 512 filters, kernel=1
├─ BatchNorm: Normalize outputs
├─ ReLU: Activation (non-linearity)
├─ Dropout 0.3: Random dropout
└─ Output: (32, 512, 200)

Conv1d Block 2:
├─ Conv1d: 512 → 256 filters
├─ BatchNorm + ReLU + Dropout
└─ Output: (32, 256, 200)

Conv1d Block 3:
├─ Conv1d: 256 → 128 filters
├─ BatchNorm + ReLU + Dropout
└─ Output: (32, 128, 200)

Conv1d Block 4:
├─ Conv1d: 128 → 64 filters
├─ BatchNorm + ReLU + Dropout
└─ Output: (32, 64, 200)  ← Features are now more "abstract"
```

#### What Features Do They Learn?

| Layer | What It Learns |
|-------|----------------|
| Early Layers (Conv1d 1-2) | Low-level patterns: sudden movements, loud sounds |
| Middle Layers (Conv1d 3-4) | Mid-level patterns: punch sequences, shouting patterns |
| Later Stages (GCN) | High-level patterns: typical violence scenario structure |

---

### STAGE 3: Graph Convolution (Multi-Path Processing)

#### What Happens?
The model parallel-processes the sequence using 3 different relationship matrices.

#### Why These 3 Paths?

**Path 1: Self-Similarity Adjacency**
```
Question: Which frames are similar to each other?
Method: Cosine similarity between feature vectors
Result: Frame showing "punch" connects to other "punch" frames
```

**Path 2: Distance-Based Adjacency**
```
Question: Which frames are close in time?
Method: Proximity in sequence (frames 50-51 are neighbors)
Result: Temporal smoothing (what happens next matters)
```

**Path 3: Score-Based Adjacency**
```
Question: Which frames have similar anomaly scores?
Method: Predicted violence score similarity
Result: Frames with "high violence probability" reinforce each other
```

#### Code Representation:

```python
# Start: features from Conv layers
x = features  # Shape: (32, 200, 64)

# Path 1: Self-similarity graph
adj1 = compute_adj(x)  # (32, 200, 200) adjacency matrix
x1 = GraphConv(x, adj1)  # (32, 200, 64) graph convolution

# Path 2: Distance-based graph
adj2 = distance_adjacency()  # (32, 200, 200)
x2 = GraphConv(x, adj2)  # (32, 200, 64)

# Path 3: Score-based graph
adj3 = compute_score_adj(anomaly_scores)  # (32, 200, 200)
x3 = GraphConv(x, adj3)  # (32, 200, 64)

# Combine all paths
x_combined = concat([x1, x2, x3])  # (32, 200, 192)
```

#### Real Example:

Imagine 3 frames with violence predictions:
```
Frame A: "punch" → score = 0.9  ← High violence
Frame B: "run"   → score = 0.2
Frame C: "punch" → score = 0.85 ← High violence

Path 1 (self-similarity):   A and C are close (both punches) → strong connection
Path 2 (distance):          A-B and B-C are temporal neighbors → connections
Path 3 (score-based):       A and C have similar scores → strong connection

Result: Frames A and C reinforce each other's violence predictions
```

---

### STAGE 4: Classification & Output

#### What Happens?
A deep neural network classifier takes the 192-dimensional fused features and predicts class probabilities for each frame.

#### Architecture:

```python
Classifier Network:
├─ Input: (32, 200, 192)
├─ Dense: 192 → 256 + BatchNorm + ReLU + Dropout(0.4)
├─ Dense: 256 → 128 + BatchNorm + ReLU + Dropout(0.3)
└─ Dense: 128 → num_classes (1 or 7)
   └─ Output: (32, 200, num_classes)
```

#### Output Interpretation:

**Binary Mode** (num_classes=1):
```
Output per frame: 0.95 → "95% probability this frame is violent"
```

**Multi-Class Mode** (num_classes=7):
```
Output per frame: 
├─ Normal:  0.05
├─ Violence1: 0.80  ← Highest probability
├─ Violence2: 0.10
├─ Violence3: 0.02
├─ Violence4: 0.01
├─ Violence5: 0.01
└─ Violence6: 0.01
```

---

## 🔄 Complete Forward Pass Example

Let's trace one batch through the entire model:

```
[BATCH INPUT]
├─ 32 videos
├─ 200 frames each
├─ RGB: 1024 dims + Audio: 128 dims = 1152 dims total
└─ Shape: (32, 200, 1152)

[STAGE 1: FUSION & ENCODING]
Fusion Network:
  - Takes (200, 1152) per video
  - Learns attention weights for RGB vs Audio
  - Reduces to (200, 256)
  
Positional Encoding:
  - Adds sin/cos values based on position
  - Frame 0: [features + PE(0)]
  - Frame 100: [features + PE(100)]
  - Each position gets unique encoding

Output: (32, 200, 256)

[STAGE 2: CNN FEATURE EXTRACTION]
4 Conv blocks (each with BatchNorm, ReLU, Dropout):
  256 → 512 → 256 → 128 → 64

Output: (32, 200, 64)
These 64 dimensions are learned features representing:
  - Motion patterns
  - Audio intensity variations  
  - Temporal dynamics
  - Contextual information

[STAGE 3: GRAPH CONVOLUTION]
Path 1 (Similarity): Which frames look similar?
  - Compute cosine similarity between all frame pairs
  - Create adjacency matrix
  - Apply GraphConv

Path 2 (Temporal): Which frames are neighbors in time?
  - Compute distances between frame positions
  - Create distance-based adjacency
  - Apply GraphConv

Path 3 (Score): Which frames have similar violence scores?
  - Compute preliminary anomaly scores
  - Create score-based adjacency
  - Apply GraphConv

Concatenate: (32, 200, 64) + (32, 200, 64) + (32, 200, 64) → (32, 200, 192)

[STAGE 4: CLASSIFICATION]
Dense layers:
  (32, 200, 192) → Dense → (32, 200, 256)
                → Dense → (32, 200, 128)
                → Dense → (32, 200, num_classes)

[FINAL OUTPUT]
Binary Mode:  (32, 200, 1)  ← Violence probability per frame
Multi Mode:   (32, 200, 7)  ← Probability of each violence type
```

---

## 🧠 Key Concepts Explained

### What is Batch Normalization?
```
Without BatchNorm:
├─ Different videos have different feature scales
├─ Layer 1 outputs varied ranges (0-1000)
├─ Layer 2 receives unpredictable inputs
└─ Training is unstable, slow convergence

With BatchNorm:
├─ After each layer, normalize outputs to mean=0, std=1
├─ All videos normalized consistently
├─ Layer 2 receives predictable inputs
└─ Training is stable, faster convergence
```

### What is Graph Convolution?
```
Traditional CNN: A neuron looks at neighbors in time/space
Graph Convolution: A neuron looks at connected nodes in a graph

In our case:
- Nodes = frames in the video
- Edges = relationships (similar features, temporal proximity, similar scores)
- Message passing = frames influence each other through edges

Example:
Frame 50 is connected to Frames 49, 51 (temporal)
                        and Frames 22, 88 (similar violence scores)
Frame 50 gets updated based on these 4 connected frames
```

### What is Positional Encoding?
```
Problem: A CNN doesn't know if frame 0 or frame 199
Solution: Add sin/cos patterns that vary with position

Frame 0:   PE(0) = [sin(0/10000^0), cos(0/10000^0), sin(0/10000^1), cos(0/10000^1), ...]
Frame 100: PE(100) = [sin(100/10000^0), cos(100/10000^0), sin(100/10000^1), ...]

Each position gets a unique encoding
Model learns: frame 0 encoding ≠ frame 100 encoding
Therefore: it can learn temporal patterns
```

---

## 📈 Information Flow Summary

| Stage | Input Shape | Output Shape | Key Operation |
|-------|------------|--------------|----------------|
| Raw Input | (B, T, 1152) | - | RGB + Audio concatenated |
| Fusion | (B, T, 1152) | (B, T, 256) | Attention-based combining |
| Encoding | (B, T, 256) | (B, T, 256) | Add temporal position info |
| Conv Layers | (B, T, 256) | (B, T, 64) | Learn abstract features |
| Path 1 GCN | (B, T, 64) | (B, T, 64) | Process via similarity graph |
| Path 2 GCN | (B, T, 64) | (B, T, 64) | Process via temporal graph |
| Path 3 GCN | (B, T, 64) | (B, T, 64) | Process via score graph |
| Concatenate | 3×(B, T, 64) | (B, T, 192) | Combine all paths |
| Classifier | (B, T, 192) | (B, T, C) | Predict class per frame |

Where: B=Batch size, T=Time steps, C=Number of classes

---

## 🎯 Why This Architecture Works

1. **Multi-Modal Fusion**: Combines RGB (what we see) with Audio (what we hear)
2. **Temporal Awareness**: Positional encoding helps understand time
3. **Multi-Path Processing**: 3 GCN paths capture different relationship types
4. **Batch Normalization**: Stable training, better convergence
5. **Graph Convolution**: Frames influence each other naturally
6. **Deep Classifier**: Can learn complex, non-linear patterns

---

## 🔧 Customization Points

Want to modify the model? Here are key adjustable parameters:

### In model.py:
```python
# Feature dimensions
Conv1d layers: 256 → 512 → 256 → 128 → 64  (change filter sizes)
GCN output: 64 dimensions (change for more/less expressiveness)

# Regularization
Dropout rates: 0.3 to 0.5 (higher = more regularization)

# Positional Encoding
Dimension: 256 (must match fusion output)
Max sequence length: 200 (change for longer videos)
```

### In option.py:
```python
--lr: Learning rate (default 0.0001)
--batch-size: Batch size (default 128)
--weight-decay: L2 regularization (default 0.0001)
--scheduler-milestones: When to decay LR (default [15, 30])
```

---

## 📚 Reading Order for Understanding

1. **Start here**: This file (MODEL_ARCHITECTURE_GUIDE.md)
2. **Then read**: QUICK_REFERENCE.md (training commands)
3. **Then read**: IMPROVEMENTS.md (detailed implementation)
4. **Finally check**: layers.py and model.py (actual code)

---

## ❓ Common Questions

**Q: Why use Graph Convolution instead of just CNN?**
A: Frames can have long-range dependencies. A punch at second 5 might influence the score at second 10. GCN handles these long-range relationships better than temporal convolutions alone.

**Q: Why 3 different adjacency matrices?**
A: Different relationships matter:
- Similarity: Similar actions should have similar labels
- Distance: Nearby frames are temporally related
- Score: Frames with similar predictions should reinforce each other

**Q: Why reduce features from 1152 to 256 to 64?**
A: Progressive abstraction:
- 1152: Raw features (high-dimensional, hard to learn)
- 256: After fusion (multimodal interaction learned)
- 64: After CNN (abstract patterns extracted)
- This pyramid helps learn hierarchical representations

**Q: What does BatchNorm actually do in training vs inference?**
A: Training: Normalizes each batch to mean=0, std=1
   Inference: Uses running statistics from training

**Q: Can I use this for other datasets?**
A: Yes! If you have:
1. Multi-modal features (or adapt single-modal)
2. Sequential data (videos, time-series)
3. Frame-level labels
Just retrain with your data.

---

**Created**: March 25, 2026  
**For**: XDVioDet Violence Detection Model  
**Audience**: Anyone new to the project
