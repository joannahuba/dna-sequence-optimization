# Module Architecture and Class Registry Specification

This folder details detailed structural layout of the <library_name>, explaining workflow and how to modify library. 

---

## Core Infrastructure Layout

The python package codebase under `src/PromotorOptimizer/` separates pipeline orchestration logic from algorithmic details using strict object-oriented interface patterns:

* **[pipeline_overview.md](./pipeline_overview.md)**: Explains the top-level orchestration mechanics driven by `PipelineRunner` and the execution abstraction layer contained within `SequencePredictorModelWrapper`.
* **[optimizers.md](./optimizers.md)**: Details the uniform blueprint interface `BaseOptimizer` and maps structural validation checks handled through the regular expression parsing engine `SequenceValidator`.
* #TODO ... pozostałe fragmenty któe można edytować (score function, interpreters)

---

## Component Directories and Submodules

### [`core/`](./core.md)
Contains the structural wrappers managing in-memory telemetry, sequence streaming via virtual RAM files (`tempfile`), and batch scheduling routines.

### `interpreters/`
Manages gradient tracking and differential extraction strategies. Submodule components route execution through an `InterpreterRegistry` factory pattern to load standalone attribution handlers.

### `loss_functions/`
Isolates evaluation metrics from optimization tracking loops. It exposes a dual interface contract: `evaluate_tensor_loss` for automatic differentiation operations on backpropagation graphs, and `evaluate_numpy_fitness` for host scalar calculations.

### `models/`
Implements the unified `ModelManager` interface layer. It acts as a wrapper for external architectures, formatting inputs into standardized $L \times 4$ floating-point tensors and handling batch inference dispatching.

### `extensions/`
Contains downstream processing tools like **[models.md](extensions/models.md)** and **[motif_databases.md](extensions/motif_databases.md)**, which provide auxiliary facilities for analyzing transcription factor binding site (TFBS) changes during optimization trajectories.