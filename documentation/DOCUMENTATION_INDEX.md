# 📚 XDVioDet Documentation Index

Welcome to the XDVioDet Violence Detection Model! This is your guide to understanding the project.

## 🗂️ Documentation Files Overview

### 1. **MODEL_ARCHITECTURE_GUIDE.md** ← **START HERE**
**Purpose**: Understand what the model is and how it works  
**Audience**: Anyone new to the project, ML beginners  
**Key Sections**:
- What is this model? (Real-world examples)
- Architecture overview (High-level flow diagram)
- Detailed step-by-step breakdown of each stage
- Key concepts explained (BatchNorm, Graph Convolution, etc.)
- Complete forward pass example

**Read this if**: You want to understand HOW the model works

---

### 2. **QUICK_REFERENCE.md**
**Purpose**: Quick lookup guide for training and debugging  
**Audience**: Developers who want to start training quickly  
**Key Sections**:
- Model upgrades summary (Before/After comparison)
- Training configuration options
- Common training commands
- Performance expectations
- Quick debug checklist

**Read this if**: You want copy-paste commands and quick answers

---

### 3. **IMPROVEMENTS.md**
**Purpose**: Detailed explanation of model optimization & training improvements  
**Audience**: Developers who want in-depth technical details  
**Key Sections**:
- Changes made to model.py (7 major improvements)
- Model architecture comparison
- Training recommendations & best practices
- Hyperparameter tuning guide
- Advanced architectures to try

**Read this if**: You want deep technical understanding or want to modify the model

---

## 🎓 Recommended Reading Order

### For ML Beginners
1. **MODEL_ARCHITECTURE_GUIDE.md** - Understand what the model does
2. **QUICK_REFERENCE.md** - See training commands
3. **IMPROVEMENTS.md** - Learn about optimizations
4. Look at actual code: `model.py`, `layers.py`

### For Experienced ML Engineers
1. **QUICK_REFERENCE.md** - Summary of changes
2. **IMPROVEMENTS.md** - Technical details
3. **MODEL_ARCHITECTURE_GUIDE.md** - For reference if needed
4. Code: `main.py`, `model.py`, `train.py`

### For Just Getting Started
1. **QUICK_REFERENCE.md** - Training commands
2. Run: `python main.py --seed 42 --max-epoch 50`
3. Check results in `./ckpt/`
4. Read other docs as questions come up

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
- Place features in `Features/` directory (I3D-features provided)
- Update modality list files in `list/` directory
- Ground truth should be in `.npy` files

### 3. Start Training
```bash
# Basic training with defaults
python main.py

# Training with custom config (recommended)
python main.py --seed 42 --lr 0.0001 --batch-size 128 --max-epoch 50

# Different operating mode
python main.py --online-mode Binary --weights Inverse

# Tuned for smaller memory
python main.py --batch-size 64 --lr 0.0005
```

### 4. Monitor Training
- Check `./ckpt/` for saved models
- Metrics saved as: `roc_auc_*.npy`, `f1_*.npy`, `precision_*.npy`, etc.
- Plot these files to visualize learning curves

---

## 🔍 Navigation by Question

**Q: How does the model work step-by-step?**  
→ Read: [MODEL_ARCHITECTURE_GUIDE.md](MODEL_ARCHITECTURE_GUIDE.md) (Sections: "Detailed Step-by-Step Breakdown")

**Q: How do I train the model?**  
→ Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (Section: "⚙️ Training Configuration")

**Q: What changed in the model compared to the original?**  
→ Read: [IMPROVEMENTS.md](IMPROVEMENTS.md) (Section: "Changes Made to model.py")

**Q: What are the hyperparameters and what do they do?**  
→ Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (Section: "Training Configuration")  
→ Then: [IMPROVEMENTS.md](IMPROVEMENTS.md) (Section: "Hyperparameter Tuning")

**Q: How do I debug if training isn't working?**  
→ Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (Section: "🧪 Debugging Checklist")

**Q: What's the optimizer doing with parameter groups?**  
→ Read: [IMPROVEMENTS.md](IMPROVEMENTS.md) (Section: "Optimizer Configuration Details")

**Q: Should I use BatchNorm/Dropout/Attention?**  
→ Read: [MODEL_ARCHITECTURE_GUIDE.md](MODEL_ARCHITECTURE_GUIDE.md) (Section: "Key Concepts Explained")

**Q: What output should I expect?**  
→ Read: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (Section: "📈 Expected Performance Trajectory")

**Q: How do I customize the model?**  
→ Read: [MODEL_ARCHITECTURE_GUIDE.md](MODEL_ARCHITECTURE_GUIDE.md) (Section: "🔧 Customization Points")

---

## 📊 Project Structure

```
XDVioDet/
├── model.py                      ← Neural network architecture
├── layers.py                     ← Custom layers (GCN, GraphAttention)
├── dataset.py                    ← Data loading & preprocessing
├── train.py                      ← Training loop
├── test.py                       ← Evaluation & testing
├── main.py                       ← Main entry point to start training
├── option.py                     ← Argument parser & configuration
│
├── 📚 DOCUMENTATION
├── MODEL_ARCHITECTURE_GUIDE.md   ← How the model works
├── QUICK_REFERENCE.md            ← Training commands (copy-paste)
├── IMPROVEMENTS.md               ← Technical improvements (detailed)
├── DOCUMENTATION_INDEX.md        ← This file
│
├── 📁 Features/                  ← Pre-extracted I3D features
├── 📁 list/                      ← Data list files
├── 📁 ckpt/                      ← Saved models & metrics
└── 📁 __pycache__/               ← Python cache (ignore)
```

---

## 🚀 Common Workflows

### Workflow 1: Understanding the Model
```
1. Read MODEL_ARCHITECTURE_GUIDE.md
2. View model.py lines 50-80 (Model __init__)
3. View model.py lines 170-200 (forward pass)
4. Check layers.py (understand GCN)
```

### Workflow 2: Training & Debugging
```
1. Read QUICK_REFERENCE.md
2. Run: python main.py --seed 42
3. Monitor: Check ./ckpt/ for metrics
4. Debug: Use "Debugging Checklist" if underperforming
5. Adjust: Read IMPROVEMENTS.md for hyperparameter ideas
```

### Workflow 3: Experimenting & Tuning
```
1. Current best: python main.py --seed 42 --batch-size 128
2. Try: python main.py --seed 42 --batch-size 64 --lr 0.0005
3. Compare: Check AUC-ROC in both experiments
4. Iterate: Use IMPROVEMENTS.md hyperparameter table
```

### Workflow 4: Understanding an Error
```
1. Google the error message
2. Check QUICK_REFERENCE.md "Debugging" section
3. If related to model: Check MODEL_ARCHITECTURE_GUIDE.md
4. If related to training: Check IMPROVEMENTS.md
5. If related to config: Check option.py
```

---

## 🎯 Success Metrics

After implementing improvements, you should see:

| Metric | Expected | Timeline |
|--------|----------|----------|
| Training speed | 20-30% faster | Epoch 1-5 |
| AUC-ROC improvement | +3-5% | Epoch 10 |
| Convergence | Smoother curves | Epoch 1-50 |
| Generalization | Smaller train-val gap | Epoch 20+ |

---

## 💡 Pro Tips

1. **Always use `--seed` for reproducibility**
   ```bash
   python main.py --seed 42
   ```

2. **Save experiment configs for comparison**
   ```bash
   # Good naming convention
   python main.py --seed 42 --batch-size 128 --model-name exp1_batch128
   python main.py --seed 42 --batch-size 64 --model-name exp1_batch64
   ```

3. **Monitor early (first 5 epochs)**
   - If loss isn't decreasing → Learning rate too low
   - If loss spikes → Learning rate too high
   - Fix before wasting 50 epochs

4. **Use the 3-command pattern for tuning**
   ```bash
   # Baseline
   python main.py --seed 42
   
   # Variant 1
   python main.py --seed 42 --batch-size 64
   
   # Variant 2
   python main.py --seed 42 --weight-decay 0.0005
   ```

5. **Check ./ckpt/ regularly**
   ```bash
   ls -lt ./ckpt/  # See latest saved models
   ```

---

## 🆘 Getting Help

| Problem | Solution |
|---------|----------|
| Don't understand model | Read MODEL_ARCHITECTURE_GUIDE.md |
| Want to train | Check QUICK_REFERENCE.md → Common Commands |
| Training is slow/wrong | Check QUICK_REFERENCE.md → Debugging |
| Want to modify model | Read IMPROVEMENTS.md + model.py |
| Config questions | Check option.py comments |
| Data format issues | Check dataset.py |
| Results not matching | Check for randomness with --seed |

---

## 📈 Learning Path for This Project

**Week 1: Understand**
- [ ] Read MODEL_ARCHITECTURE_GUIDE.md
- [ ] Read QUICK_REFERENCE.md  
- [ ] Understand model.py structure
- [ ] Understand what these files do: dataset.py, train.py, test.py

**Week 2: Implement**
- [ ] Run basic training: `python main.py`
- [ ] Monitor metrics
- [ ] Try different batch sizes
- [ ] Read IMPROVEMENTS.md for advanced options

**Week 3: Experiment**
- [ ] Compare different configs (use --seed for reproducibility)
- [ ] Tune hyperparameters
- [ ] Debug issues using checklists
- [ ] Document best settings

**Week 4: Deploy/Publish**
- [ ] Save best model
- [ ] Create evaluation notebook
- [ ] Document results
- [ ] Prepare for testing on new data

---

## 📝 File Purposes at a Glance

| File | Lines | Purpose |
|------|-------|---------|
| model.py | 280 | Neural network architecture (Model class) |
| layers.py | 190 | Graph convolution & special layers |
| dataset.py | 120 | Load and preprocess video features |
| train.py | 100 | Training loop with loss calculation |
| test.py | 150 | Evaluation metrics (AUC, F1, Precision, Recall) |
| main.py | 150 | Entry point, argument handling, setup |
| option.py | 35 | Command-line arguments & defaults |

---

## 🎓 Key Takeaways

1. **The Model**: Detects violence in videos using RGB + Audio features
2. **Key Innovation**: 3-path Graph Convolution for different relationships
3. **Key Improvements**: BatchNorm, Positional Encoding, Modality Attention
4. **Training**: Use main.py with various --options for customization
5. **Documentation**: Start with MODEL_ARCHITECTURE_GUIDE.md, then QUICK_REFERENCE.md

---

## 📌 Current Implementation: Option 3 (Simplified for Research Paper)

**Date**: March 26, 2026  
**Architecture**: XD-Violence core + PositionalEncoding + Multi-class classification  
**Key Characteristics**:
- Simplified feature fusion (linear projection instead of attention)
- 2-layer classifier and approximator (reduced complexity)
- Retained core 3-path GCN from original paper
- Added temporal positional encoding
- **Primary contribution**: Binary→Multi-class violence classification

**File Summary**:
| File | Status | Purpose |
|------|--------|---------|
| model.py | ✅ FINAL | Option 3 (simplified, paper-friendly) |
| train.py | ✅ FINAL | Enhanced with error handling & monitoring |
| test.py | ✅ FINAL | Includes mAP metric |
| main.py | ✅ FINAL | Full pipeline with logging |
| option.py | ✅ FINAL | All hyperparameters configurable |

---
