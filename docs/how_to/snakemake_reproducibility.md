

# Reproducibility Guide: Snakemake Workflow Operations

The <library_name> isolates execution tracking paths from core python assets via an automated Snakemake orchestration layer. This ensures reproducibility, idempotent resource management, and fault-tolerant computing profiles across diverse sequence datasets.

---

## Core Execution Mechanics

Snakemake evaluates execution tracks using an explicit Directed Acyclic Graph (DAG). Instead of scanning parameters dynamically on every step, it checks the files declared inside the `input` and `output` constraints of active rules.

### Idempotency and Crash Recovery
* **Timestamp and State Verification**: If a sequence optimization step terminates unexpectedly (e.g., due to VRAM out-of-memory, machine shutdown, or manual termination), the target file `*_trajectory.json` is not written or finalized.
* **Short-Circuit Re-execution**: Upon calling the workflow engine again, Snakemake cross-checks the storage disk state. It skips all completed sequence targets that contain complete output markers and restarts calculations exclusively from the last unfinalized step, protecting expensive GPU resource allocations.

---

## Target Optimization Calls

All workflow routing requires an explicit definition of the configuration blueprint file using the `-s` or `--snakefile` parameter flag.

### Single Targeted Run Execution
To launch a specific experiment against a targeted biological sample dataset, provide the explicit path of the desired output file:
```bash
snakemake -s workflow/Snakefile.smk results/exp01_reconstruction_12/reconstruction_input_1_2_trajectory.json --cores 1

```

### Complete Batch Run Execution

To process all active declarative experiments matching files defined within the `config/experiments/` catalog concurrently, use the global evaluation target:

```bash
snakemake -s workflow/Snakefile.smk --cores 1
```

---

## Configuration Parameter Infrastructure

The pipeline parameters are split into two decoupled layers located inside the `config/` namespace:

### 1. Global System Boundaries (`config/global_config.yaml`)

Stores immutable background paths and default biological sequence rule boundaries that apply to all experiments unless overridden:

```yaml
paths:
  # folder with input files
  input_directory: "data/preprocessed"
  # directory with prelearned models 
  checkpoint_directory: "data/checkpoints"
  # folder folder connected to experiments will be created
  results_directory: "data/results"

# boundary condition for promotors
default_validation_config:
  max_homopolymer_at: 10
  max_homopolymer_gc: 7
  gc_percent_range: [0.25, 0.65]
```

### 2. Declarative Experiment Layouts (`config/experiments/`)

Each individual YAML file defines a single distinct experiment branch. It isolates the choice of objective from the library registry, limits iterations, and controls algorithmic tuning parameters inside dedicated class override structures.

```yaml
# Target objective configuration name mapping
objective: "reconstruction"

# Define explicit input files to target, or set to null to process all files in data/preprocessed/
input_files:
  - "reconstruction_input_1_2.tsv"

iterations: 40
mutation_budget: 40

models:
  - "original_modified"
optimizers:
  - "beam_search"
interpreters:
  - "integrated_gradients"

# Specific component hyperparameter tuning maps
optimizer_overrides:
  BeamSearchOptimizer:
    beam_width: 30
    top_k_positions: 40

interpreter_overrides:
  IntegratedGradientsStrategy:
    steps: 30

```

By adding a new experiment configuration file to `config/experiments/`, Snakemake dynamically builds matching file rules without requiring manual additions to the core python script infrastructure.
