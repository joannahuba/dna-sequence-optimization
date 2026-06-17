# Deep Regulatory Optimization

**Modular deep learning framework for regulatory DNA sequence optimization**

---

## Key Features

The system supports:

* optimization of DNA sequences using ML models
* reconstruction and design of regulatory elements
* interpretation of model decisions via attribution methods
* biological validation using TF motif databases (JASPAR / TOMTOM / MOODS)
* visualization of motif and importance landscapes across sequences

It is **model-agnostic** and can be connected to any PyTorch-based sequence model through a unified interface.

---

Good — this is already close to production, you just need to **normalize it into a clean UV + Snakemake onboarding flow** and remove ambiguity.

Below is a **production-ready rewrite of your Usage section + Quick Start**, adapted to `uv`.

---

## Quick start

### Installation (uv-based setup)

This project uses **uv** for environment and dependency management.

#### 1. Initialize environment

```bash
uv venv
```

#### 2. Activate environment

```bash
source .venv/bin/activate
```

#### 3. Sync dependencies

```bash
uv sync
```

---

#### Optional: editable install (development mode)

If you want to install the package in editable mode:

```bash
pip install -e .
```

> Note: PyTorch must be installed separately depending on your system (CPU/GPU version).

---

### Data Setup

All required datasets can be downloaded using:

```bash
python scripts/download/download_repository_data.py
```

This script populates the required directories for experiments and preprocessing.

---

### Quick Execution

All experiments are executed through a **Snakemake-based workflow graph**.

#### Run a single experiment target

```bash
snakemake -s workflow/Snakefile.smk \
  results/<config_experiment_name>/<input_file>_trajectory.json \
  --cores 1
```

#### Example

```bash
snakemake -s workflow/Snakefile.smk \
  results/test01_reconstruction/reconstruction_input_1_2_trajectory.json \
  --cores 1
```

---

#### Run full experiment suite

```bash
snakemake -s workflow/Snakefile.smk --cores 1
```

---

### Project Entry Points

* **Pipeline entry:**
  `PromotorOptimizer.pipeline`

* **Workflow engine:**
  `workflow/Snakefile.smk`

* **Experiment configs:**
  `config/experiments/`

---

### Reproducibility Guide

For full workflow semantics, caching behavior, and crash recovery details, see:

```
docs/how_to/snakemake_reproducibility.md
```

---

## About the Project

This project frames regulatory DNA sequence analysis as a **continuous optimization problem in sequence space**, where DNA is treated as a structured, searchable, and optimizable design manifold rather than a static biological object. The goal is to couple predictive deep learning models of gene regulation with algorithmic sequence design, forming a closed loop that **optimizes, interprets, and validates** regulatory DNA sequences.

At the core of the system is a learned function that approximates regulatory activity:

$$
f_m: \mathbb{R}^{L \times 4} \rightarrow \mathbb{R}
$$

which maps a one-hot encoded DNA sequence (X \in {0,1}^{L \times 4}) to a continuous regulatory score (e.g., expression proxy). Given this predictive landscape, the framework performs structured search over sequence space using gradient-based and stochastic optimization methods, guided by attribution signals:

$$
I \in \mathbb{R}^{L \times 4}, \quad I_{i,b} = \text{importance of nucleotide } b \text{ at position } i
$$

Rather than optimizing sequences blindly, the system couples three interacting components:

1. **Prediction:** estimating regulatory activity using deep neural networks trained on regulatory genomics data
2. **Interpretation:** computing nucleotide-level attribution maps (saliency, Integrated Gradients, in-silico mutagenesis)
3. **Optimization:** iteratively modifying sequences using beam search, stochastic sampling, and annealing-based methods

This forms a feedback loop in which predictions guide sequence edits, while attribution constrains exploration toward biologically meaningful regions.

Formally, each optimization step is expressed as a transition in sequence space:

$$
X_{t+1} = \mathcal{O}(X_t, f_m(X_t), I(X_t))
$$

where (\mathcal{O}) is a deterministic or stochastic search operator conditioned on predicted fitness and interpretability structure.

To ensure biological plausibility, the system integrates post-hoc validation against transcription factor binding motif space:

$$
\text{TFBS}(X) \approx \text{match}(X, \mathcal{D}_{motif})
$$

Motif databases are used to evaluate whether optimized sequences remain consistent with known regulatory grammar.

The framework supports:

* optimization of individual regulatory sequences (forward design)
* reconstruction of disrupted regulatory elements (inverse design)
* trajectory tracking across multiple optimization runs
* comparison across model architectures
* interpretability-driven analysis of learned regulatory logic

All experiments, including sequence optimization and reconstruction tasks, are documented in `resources/report/final_report.ipynb`, which serves as the main analysis notebook.

In summary, the system functions as a **closed-loop regulatory design engine**:

$$
\text{Design DNA} \rightarrow \text{Predict activity} \rightarrow \text{Explain importance} \rightarrow \text{Optimize sequence} \rightarrow \text{Validate biology}
$$

This enables both high-performance sequence optimization and mechanistic interpretation of what deep learning models learn about cis-regulatory code, bridging predictive genomics with interpretable sequence design.

---

## Architecture

The system implements a **closed-loop optimization and interpretation pipeline** operating directly on DNA sequences treated as a structured search space.

Deep learning models estimate regulatory activity, while attribution methods identify nucleotide-level contributions to predictions. These signals are passed to an optimization engine that iteratively modifies sequences to improve predicted biological function.

In parallel, motif analysis modules evaluate transcription factor binding patterns to enforce biological plausibility. All optimization steps are logged as full trajectories, enabling post-hoc analysis of sequence evolution, model behavior, and optimization dynamics.

---

```Mermaid
graph TD
    %% Nodes definition
    Config[Configuration Layer<br/>YAML / CLI Parameters]
    Runner[PipelineRunner<br/>System initialization, factory loading]
    Wrapper[SequencePredictorModelWrapper<br/>The primary orchestration loop]
    Opt[Task Mode: Optimization<br/>Maximize raw predictions]
    Recon[Task Mode: Reconstruction<br/>Minimize error distance to target]
    Manager[Interpreter Manager Step<br/>Extract position sensitivity matrix]
    Loop[Optimizer Generation Loop]
    Validator[SequenceValidator<br/>Regex prune]
    File[In-Memory Virtual File<br/>Bypasses disk I/O bottlenecks]
    Parallel[Parallel PyTorch Batching]
    Eval[Ensemble Score Evaluation]

    %% Relationships
    Config --> Runner
    Runner --> Wrapper
    Wrapper --> Opt
    Wrapper --> Recon
    Opt --> Manager
    Recon --> Manager
    Manager --> Loop
    Loop <--> Validator
    Loop --> File
    File --> Parallel
    Parallel <--> Eval
```

---

### System Modules — Quick Navigation Layer

This section provides a **high-level entry map of all core modules**, their responsibility, and where to start reading.

---

## `core/` — Execution Engine

**Path:** `PromotorOptimizer/core/`

**Role:**
Runs full optimization trajectories over DNA sequences.

**What to read:**
 `core/README.md`

**What it does:**

* Executes iterative optimization loops
* Coordinates models, optimizers, interpreters
* Tracks full trajectory history
* Supports ensemble and single-model modes

---

## `pipeline/` — Experiment Orchestration Layer

**Path:** `PromotorOptimizer/pipeline/`

**Role:**
Top-level entry point for experiments.

**What to read:**
 `pipeline/README.md`

**What it does:**

* Loads configs and CLI arguments
* Builds full system configuration
* Instantiates core engine
* Runs end-to-end experiments

---

## `models/` — ML Model Layer

**Path:** `PromotorOptimizer/models/`

**Role:**
Defines and manages all neural network models.

**What to read:**
 `models/README.md`

**What it does:**

* Defines genomic CNN architectures
* Loads pretrained checkpoints
* Encodes DNA datasets
* Provides unified inference interface

---

## `optimizers/` — Sequence Search Algorithms

**Path:** `PromotorOptimizer/optimizers/`

**Role:**
Implements mutation-based sequence optimization strategies.

**What to read:**
 `optimizers/README.md`

**What it does:**

* Beam search variants
* Stochastic optimization (Boltzmann, MH)
* Mutation generation logic
* Selection and update rules

---

## `interpreters/` — Attribution & Explainability

**Path:** `PromotorOptimizer/interpreters/`

**Role:**
Explains model predictions at nucleotide level.

**What to read:**
 `interpreters/README.md`

**What it does:**

* Saliency maps
* Integrated gradients
* In-silico mutagenesis
* Model interpretation manager

---

## `loaders/` — Post-processing & Analytics

**Path:** `PromotorOptimizer/loaders/`

**Role:**
Turns raw optimization logs into analytical datasets.

**What to read:**
 `loaders/README.md`

**What it does:**

* Parses JSON trajectories
* Computes metrics
* Re-scores sequences
* Produces DataFrames for analysis

---

## `analysis/` — Visualization & Reporting

**Path:** `PromotorOptimizer/analysis/`

**Role:**
Generates plots and analysis reports from results.

**What it does:**

* Convergence plots
* Trajectory visualization
* IQR / variance analysis
* Best-sequence summaries

---

## `extensions/MotifAnalysis/` — Biological Annotation Layer

**Path:** `PromotorOptimizer/extensions/MotifAnalysis/`

**Role:**
Detects and visualizes transcription factor motifs in sequences.

**What to read:**
 `extensions/MotifAnalysis/README.md`

**What it does:**

* PWM scanning (MOODS)
* JASPAR motif matching
* TF annotation (TOMTOM)
* Genomic visualization

---

## `loss_functions/` — Objective Definitions

**Path:** `PromotorOptimizer/loss_functions/`

**Role:**
Defines fitness functions used in optimization.

**What it does:**

* Promoter scoring functions
* Reconstruction losses
* De-novo analysis objectives

---

## `utils/` — Shared Infrastructure

**Path:** `PromotorOptimizer/utils/`

**Role:**
Common utilities used across all modules.

**What it does:**

* Logging
* IO / JSON schema handling
* Preprocessing (DNA encoding)
* Validation utilities

---

## Reproducibility Guide: Snakemake Workflow Operations

The system uses a **Snakemake-based orchestration layer** to separate execution tracking, experiment configuration, and core Python logic. This ensures **reproducibility, fault tolerance, and deterministic recovery** across large-scale sequence optimization runs.

---

### Core Execution Model

Snakemake executes workflows using a **Directed Acyclic Graph (DAG)**. Instead of running procedural code step-by-step, it resolves dependencies based on declared `input` and `output` files in each rule.

This design guarantees:

* deterministic execution order
* reproducible experiment runs
* automatic caching of completed steps
* safe parallelization across compute resources

---

### Fault Tolerance and Recovery

The workflow is designed for long-running optimization tasks that may fail due to GPU or system constraints.

#### Crash Recovery

If an experiment is interrupted (e.g., GPU OOM, system shutdown, manual stop), intermediate outputs such as:

```
*_trajectory.json
```

may be missing or incomplete.

#### Resume Behavior

When rerun, Snakemake:

* inspects existing output files on disk
* identifies completed steps
* skips finished computations
* resumes only from the last valid checkpoint

This ensures **no recomputation of completed optimization trajectories**, preserving expensive compute resources.

---

### Running Experiments

All executions require the workflow entry point:

```
workflow/Snakefile.smk
```

---

#### Single Experiment Run

Run a specific target output:

```bash
snakemake -s workflow/Snakefile.smk \
  results/exp01_reconstruction_12/reconstruction_input_1_2_trajectory.json \
  --cores 1
```

---

#### Batch Execution

Run all experiments defined in the configuration directory:

```bash
snakemake -s workflow/Snakefile.smk --cores 1
```

Snakemake will automatically resolve all experiment targets defined in `config/experiments/`.

---

### Configuration System

The pipeline uses a **two-layer configuration architecture**:

---

#### 1. Global Configuration

File:

```
config/global_config.yaml
```

Defines system-wide defaults shared across all experiments.

Example:

```yaml
paths:
  input_directory: "data/preprocessed"
  checkpoint_directory: "data/checkpoints"
  results_directory: "data/results"

default_validation_config:
  max_homopolymer_at: 10
  max_homopolymer_gc: 7
  gc_percent_range: [0.25, 0.65]
```

**Purpose:**

* defines dataset locations
* controls global biological constraints
* sets shared experiment infrastructure paths

---

#### 2. Experiment Definitions

Location:

```
config/experiments/
```

Each YAML file defines a **fully independent experiment configuration**.

Example:

```yaml
objective: reconstruction

input_files:
  - reconstruction_input_1_2.tsv

iterations: 40
mutation_budget: 40

models:
  - original_modified

optimizers:
  - beam_search

interpreters:
  - integrated_gradients

optimizer_overrides:
  BeamSearchOptimizer:
    beam_width: 30
    top_k_positions: 40

interpreter_overrides:
  IntegratedGradientsStrategy:
    steps: 30
```

---

## Execution Semantics

Each experiment definition controls:

* objective type (optimization or reconstruction)
* dataset selection
* iteration budget
* model selection
* optimizer selection
* interpretability method
* hyperparameter overrides

Adding a new YAML file in `config/experiments/` automatically registers a new workflow target. No changes to Python core code are required.

---

## Summary

Snakemake acts as the **execution backbone** of the system, transforming declarative experiment definitions into reproducible, checkpointed optimization pipelines over regulatory DNA sequence space.
