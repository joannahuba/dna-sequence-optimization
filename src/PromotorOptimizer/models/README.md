# Models Module — Structure Overview

This module contains everything required to:

* define genomic ML models
* load pretrained checkpoints
* encode DNA datasets
* run batch inference
* provide a unified scoring interface for optimization

Structure:

```
models/
│
├── architectures/        → MODEL DEFINITIONS (PURE PYTORCH)
│   ├── deepstarr.py              # DeepSTARR CNN architecture
│   └── genomic_model_zero.py     # All genomic CNN variants + wrappers
│
├── loaders/              → CHECKPOINT LOADING (WEIGHTS → MODEL)
│   ├── deepstarr_loader.py       # loads DeepSTARR checkpoint
│   ├── deep_starr_second.py      # loads second DeepSTARR variant
│   ├── original_modified.py      # loads modified baseline model
│   └── zero_loader.py            # loads GenomicModelZeroAdjusted
│
├── datasets/             → DATA PREPROCESSING (DNA → TENSORS)
│   ├── dna_dataset.py            # basic one-hot DNA dataset
│   └── dna_no_adapters.py       # DNA dataset with adapter trimming
│
├── model_manager.py      → INFERENCE ENGINE (MAIN ENTRY POINT)
│   # runs models, batching, scoring, ensemble statistics
│
├── registry.py           → MODEL FACTORY (NAME → MODEL SETUP)
│   # selects architecture + loader + dataset per model name
│
└── base_model_manager.py → INTERFACE (ABSTRACT CONTRACT)
    # defines required inference methods
```

It is divided into 5 functional layers:

---

# 1. `/architectures` — Model Definitions

### What this folder contains

Raw PyTorch model architectures.

### Responsibility

Defines **how DNA is processed into predictions**.

### Contents

* `deepstarr.py`
  DeepSTARR-like CNN architecture for regulatory sequence prediction

* `genomic_model_zero.py`
  Base and experimental genomic CNN models:

  * minimal baseline (`GenomicModelZero`)
  * reverse-complement wrappers
  * DeepSTARR variants
  * adjusted production model (`GenomicModelZeroAdjusted`)

### Key rule

No loading, no data handling, no inference orchestration.

Only:

> “define neural network forward pass”

---

# 2. `/loaders` — Checkpoint Loading Layer

### What this folder contains

Functions that reconstruct trained models from disk.

### Responsibility

Turn:

> checkpoint → ready-to-run PyTorch model

### Contents

* `deepstarr_loader.py`
* `deep_starr_second.py`
* `original_modified.py`
* `zero_loader.py`

### What loaders do

Each loader:

* builds architecture
* loads `.pth` weights
* moves model to device (CPU/GPU)
* sets `eval()` mode

### Key rule

No inference logic, no batching, no dataset handling.

Only:

> “restore model exactly as trained”

---

# 3. `/datasets` — Data Encoding Layer

### What this folder contains

DNA sequence dataset implementations.

### Responsibility

Convert raw biological data → model input tensors.

### Contents

* `dna_dataset.py`

  * TSV loader
  * one-hot encoding (A/C/G/T → 4 channels)
  * supports training + inference modes

* `dna_no_adapters_dataset.py`

  * removes constant prefix/suffix adapters
  * used to reduce experimental bias in sequences

### Output format

```text
(batch, 4, sequence_length)
```

### Key rule

No models, no inference, no optimization logic.

Only:

> “prepare DNA for models”

---

# 4. `/model_manager.py` — Inference Engine

### What this file contains

Central execution layer for all models.

### Responsibility

Provide unified API for:

* multi-model inference
* batching optimization
* ensemble prediction aggregation
* fitness scoring

---

### Core functions

#### `predict_sequences()`

Runs inference for multiple sequences and models:

* encodes sequences once
* runs all models
* returns structured dictionary output

#### `predict_tensor()`

Direct tensor-based inference (used internally by optimizers)

#### `score_sequence()`

Computes:

* mean prediction
* standard deviation
* final fitness score

#### `ensemble_predict()`

Returns ensemble statistics:

* mean across models
* variance
* min/max spread

---

### Key rule

This is the **ONLY layer allowed to call models during inference**

Everything else should go through it.

---

# 5. `/registry.py` — Model Factory

### What this file contains

Central mapping between:

> model name → model + dataset class

---

### Responsibility

Defines:

* which models exist in the system
* how they are loaded
* which dataset they require

---

### Example mapping

| Model name          | Loader                   | Dataset              |
| ------------------- | ------------------------ | -------------------- |
| `deepstarr`         | DeepSTARR loader         | DNADataset           |
| `deepstarr_second`  | second DeepSTARR loader  | DNADatasetNoAdapters |
| `original_modified` | modified baseline loader | DNADatasetNoAdapters |

---

### Output structure

```python
{
  "model_name": {
    "model": torch.nn.Module,
    "dataset_class": Dataset
  }
}
```

---

### Key rule

Registry is the **only entry point for model creation**s

---

# Overall Module Flow

```text id="flow_models"
registry.py
    ↓
loaders/
    ↓
architectures/
    ↓
model_manager.py
    ↓
trajectory / optimization system
```

---

# Responsibility Boundaries (Important)

| Layer         | Responsibility      | Must NOT do               |
| ------------- | ------------------- | ------------------------- |
| architectures | define networks     | load weights, handle data |
| loaders       | load checkpoints    | run inference loops       |
| datasets      | encode DNA          | run models                |
| model_manager | inference + scoring | define architectures      |
| registry      | model selection     | compute predictions       |
