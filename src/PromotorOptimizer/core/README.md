# Core Module — Trajectory Execution Engine

## Overview

The `core/` module implements the central execution engine of the PromotorOptimizer system. It orchestrates end-to-end optimization of DNA sequences..

The module does not define models, optimizers, or loss functions. Instead, it integrates these components into a unified optimization workflow that generates and evaluates sequence trajectories over multiple iterations.

The primary goal is to optimize sequences with respect to a biological fitness function (e.g. `rna_dna_ratio`) using model-guided mutation search.

---

## Responsibilities

The core module is responsible for:

* Executing optimization trajectories over input sequences
* Coordinating multiple optimizers and interpretability methods
* Managing single-model and multi-model (ensemble) execution modes
* Evaluating sequences using a pluggable objective function
* Tracking full optimization history per sequence
* Persisting incremental checkpoints during execution

It acts as the **runtime orchestration layer** of the system.

---

Here is a **short, concrete, responsibility-focused version** that removes abstraction and makes it obvious what each component does in practice.

---

## Architecture

### 1. SequencePredictorModelWrapper

This is the **experiment-level controller**.

It is responsible for:

* iterating over input DNA sequences
* running all combinations of optimizers and interpreters
* calling `TrajectoryOrchestrator` for each configuration
* storing and saving full experiment results (trajectories)

**In short:** it runs and organizes *all experiments*.

---

### 2. TrajectoryOrchestrator

This is the **single-run optimization engine**.

It is responsible for:

* running the iterative optimization loop for one sequence
* calling models to get predictions
* computing attribution maps for each model
* generating mutated candidate sequences via optimizer
* scoring candidates using the objective function
* updating the search state (beam / best sequence)

It operates in two modes:

* **Aggregated mode**: all models influence one shared optimization run
* **Discrete mode**: each model runs its own separate optimization run

**In short:** it performs *one full evolutionary optimization process*.

---

## Execution Flow

### Step 1 — Input Initialization

The system expects a dictionary of sequences:

* `sequence`: DNA string
* `mutation_budget`: optional constraint
* `target_value`: optional optimization target

Each sequence is processed independently.

---

### Step 2 — Experiment Configuration

The wrapper is initialized with:

* `ModelManager`: provides access to registered models
* `optimizers_list`: mutation/search strategies
* `interpreters_list`: attribution methods
* `objective`: fitness function implementing `BaseLossObjective`
* optional boundary constraints (`prefix_len`, `suffix_len`)

---

### Step 3 — Execution Loop (Per Sequence)

For each sequence:

1. Initialize result tracking structure
2. Iterate over all optimizers
3. Iterate over all interpreters
4. Build runtime execution context
5. Instantiate `TrajectoryOrchestrator`
6. Execute trajectory in selected mode

---

## Execution Modes

### Aggregated Mode (Ensemble Execution)

Triggered when `interpreter.resolve_output_schema(...).is_aggregated == True`

| Aspect                     | Behavior                                                      |
| -------------------------- | ------------------------------------------------------------- |
| Model execution            | All registered models run in the same optimization loop       |
| Predictions                | Computed for every model at each iteration                    |
| Attribution                | Computed per model independently                              |
| Candidate generation input | Combined `{model → attribution_map}` dictionary               |
| Optimization signal        | Merged multi-model importance maps                            |
| Search process             | Single shared search state                                    |
| Trajectories               | One trajectory shared across all models                       |
| Output structure           | One sequence evolution history containing all models' results |

#### What this means in practice

* One sequence is optimized once
* All models evaluate it at every step
* Mutation decisions are influenced by **all models together**
* The optimizer sees a **multi-model importance map at each step**

#### Used for

* sequence design that generalizes across models
* consensus-driven optimization
* ensemble-level biological objective alignment

#### Output

* single optimization trajectory
* nested model-level metrics inside each step

---

### Discrete Mode (Single-Model Execution)

Triggered when `is_aggregated == False`

| Aspect                     | Behavior                                          |
| -------------------------- | ------------------------------------------------- |
| Model execution            | One model per full optimization run               |
| Predictions                | Only target model is evaluated                    |
| Attribution                | Computed for a single model                       |
| Candidate generation input | Single `{model → attribution_map}`                |
| Optimization signal        | Model-specific importance map                     |
| Search process             | Independent search state per model                |
| Trajectories               | Separate trajectory per model                     |
| Output structure           | Multiple independent trajectories (one per model) |

#### What this means in practice

* Each model gets its own optimization run
* No sharing of search state between models
* Each model produces its own “best sequence path”
* Results reflect **model-specific preferences**

#### Used for

* comparing model behavior
* debugging model biases
* interpreting individual model decisions

#### Output

* one trajectory per model
* independent optimization histories

---


## Optimization Loop (TrajectoryOrchestrator)

The `TrajectoryOrchestrator` executes an iterative sequence optimization loop by coordinating model inference, attribution, candidate generation, and objective scoring.

It relies on the following components:

### Dependencies

| Component             | Role in execution                                                                  |
| --------------------- | ---------------------------------------------------------------------------------- |
| `ModelManager`        | Provides model registry and runs inference on sequences                            |
| `optimizers`          | Generates candidate mutations and maintains search state (beam, temperature, etc.) |
| `interpreters`        | Computes attribution maps (sequence importance signals)                            |
| `BaseLossObjective`   | Evaluates fitness of candidate sequences                                           |
| `utils.preprocessing` | Encodes DNA sequences into tensor representations                                  |
| `utils.logger`        | Records execution progress and debugging information                               |

---

## Iterative Optimization Flow

For each sequence trajectory, the system repeats the following loop for `N` iterations:

---

### 1. Sequence Evaluation

**Uses:** `ModelManager`, `utils.preprocessing`

**What happens:**

* Current DNA sequence is encoded into tensor format
* Sequence is passed through one or more models
* Each model produces a prediction score

**Output:**

* `{model → prediction score}` for current sequence

---

### 2. Attribution Computation

**Uses:** `interpreters`, `ModelManager`, encoded tensor input

**What happens:**

* Each model is paired with an interpreter
* Attribution maps are computed (importance per nucleotide)
* These maps indicate which sequence positions influence predictions most

**Output:**

* `{model → importance_map}`

---

### 3. Candidate Generation

**Uses:** `optimizers`, attribution maps, search state

**What happens:**

* Optimizer generates mutated sequence candidates
* Mutations are guided by:

  * attribution maps (important positions prioritized)
  * current search state (beam, temperature, history)

**Output:**

* list of candidate sequences

---

### 4. Fitness Evaluation

**Uses:** `ModelManager`, `BaseLossObjective`

**What happens:**

* Each candidate is evaluated using model predictions
* Objective function computes final fitness score using:

  * prediction values
  * sequence content
  * runtime metadata (e.g. mutation budget, constraints)

**Output:**

* `{candidate → fitness score}`

---

### 5. Search State Update

**Uses:** `optimizers`

**What happens:**

* Candidates are ranked by fitness
* Optimizer updates internal state:

  * selects best sequences (beam update)
  * updates exploration parameters (e.g. temperature)
  * stores mutation history

**Output:**

* updated `search_state`
* updated `best_sequence`

---

### 6. Logging & Trajectory Recording

**Uses:** `utils.logger`

**What happens:**

* Full state of each iteration is stored:

  * current sequence
  * predictions
  * attribution maps
  * beam state
  * optimizer parameters

**Purpose:**

* reconstruct full optimization trajectory
* enable debugging and reproducibility

---

## Summary of Data Flow per Iteration

```text
Sequence
  ↓ (ModelManager)
Predictions
  ↓ (Interpreters)
Attribution Maps
  ↓ (Optimizers)
Candidate Sequences
  ↓ (ModelManager + Objective)
Fitness Scores
  ↓ (Optimizer)
Updated Search State
  ↓
Next Iteration
```

---

### Output Structure

The module returns a nested dictionary:

```text
sequence_id
  └── optimizer_name
        └── interpreter_name
              └── model(s)
                    └── steps[]
```

Each step contains:

* iteration index
* sequence state
* model predictions
* attribution maps
* beam population
* optimizer metadata

---

## Usage

The core module is not executed directly. It is invoked via the pipeline layer.

```python
from PromotorOptimizer.core.wrapper import SequencePredictorModelWrapper

wrapper = SequencePredictorModelWrapper(
    model_type="single",
    sequences=sequences,
    model_manager=model_manager,
    optimizers_list=optimizers,
    interpreters_list=interpreters,
    objective=objective,
)

results = wrapper.ExecuteTrajectories(
    output_path="outputs/trajectory.json"
)
```

---

## Summary

The `core/` module is the orchestration backbone of the system. It transforms independent ML components into a unified evolutionary optimization pipeline that iteratively refines DNA sequences under a defined biological objective.

