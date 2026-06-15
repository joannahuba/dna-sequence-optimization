# Module Architecture and Class Registry Specification

This folder contains  detailed structural layout of the <library_name>, explaining workflow and how to modify library. 

---

## Core Infrastructure Layout

The python package codebase under `src/PromotorOptimizer/` separates pipeline orchestration logic from algorithmic details using strict object-oriented interface patterns:

* **[Pipeline Overview Module](pipeline_overview.md)**: Global orchestration flow, execution lifecycle tracking via `PipelineRunner`, and in-memory optimization paths managed by the central model wrapper layer.
* **[Optimizers Infrastructure Module](optimizers.md)**: Heuristic space-exploration strategies, sequence token mutation loops, and structural verification constraints.
* **[Loss Functions Module](loss_functions.md)**: Differentiable penalty boundaries on graphs and host CPU fitness calculation metrics.
* **[Interpreters Module](interpreters.md)**: Tensor backpropagation operations and empirical landscape attribution mapping components.
* **[Models Module](models.md)**: High-throughput neural network prediction wrappers and dataset tensor compilation managers.
* **[Extensions Module](extensions/README.md)**: Downstream sequence mining tools, motif database alignments, and post-processing analytical utilities that hook into completed execution traces.

---

## Component Directories and Submodules

### [`core/`](./core.md)
Contains the structural wrappers managing in-memory telemetry, sequence streaming via virtual RAM files (`tempfile`), and batch scheduling routines.

### [`interpreters/`](./interpreters.md)
Manages gradient tracking and differential extraction strategies. Submodule components route execution through an `InterpreterRegistry` factory pattern to load standalone attribution handlers.

### `loss_functions/`
Isolates evaluation metrics from optimization tracking loops. It exposes a dual interface contract: `evaluate_tensor_loss` for automatic differentiation operations on backpropagation graphs, and `evaluate_numpy_fitness` for host scalar calculations.

### `models/`
Implements the unified `ModelManager` interface layer. It acts as a wrapper for external architectures, formatting inputs into standardized $L \times 4$ floating-point tensors and handling batch inference dispatching.

### `extensions/`
Contains downstream processing tools like ***[motif_databases.md](extensions/motif_databases.md)**, which provide auxiliary facilities for analyzing transcription factor binding site (TFBS) changes during optimization trajectories.