# Loaders Architecture

## Structure and Overview

The `loaders` package provides the **post-processing and analytical layer** of the optimization framework. While optimizers generate JSON trajectory logs describing sequence evolution, loaders transform these raw artifacts into structures suitable for downstream analysis, visualization, benchmarking, and reporting.

The package performs three major tasks:

1. **Trajectory loading**

   * Converts nested JSON optimization logs into tabular representations.

2. **Prediction enrichment**

   * Re-evaluates discovered sequences using additional predictive models and injects those scores back into logs.

3. **Metric computation**

   * Calculates trajectory-level analytical indicators describing optimization behavior.

The final output is typically a **pandas DataFrame**, where each row corresponds to a single optimization iteration enriched with metadata, predictions, and derived metrics.

---

# Theoretical Overview

Optimization logs represent trajectories in sequence space:

$$
X^{(0)} \rightarrow X^{(1)} \rightarrow X^{(2)} \rightarrow \cdots \rightarrow X^{(T)}
$$

where

$$
X^{(t)}
$$

denotes the sequence state at iteration (t).

The role of the loader subsystem is to transform this trajectory into an analytical representation:

$$
\mathcal{L}
:
{\text{JSON trajectories}}
\rightarrow
{\text{DataFrames with metrics}}
$$

This transformation allows:

* quantitative comparison between optimization strategies,
* statistical characterization of optimization dynamics,
* retrospective model validation,
* preparation of publication-quality analyses.

---

# Package Structure

```text
loaders/
├── loader.py
├── utils/
│   └── load_json.py
├── add_metrics/
│   ├── base_metric.py
│   ├── exampe.py
│   └── orchestror.py
└── add_scores/
    ├── extractor.py
    ├── updater.py
    └── main_runner.py
```

---

# File-to-Theory Mapping

| File                         | Concept                        | Purpose                                          |
| ---------------------------- | ------------------------------ | ------------------------------------------------ |
| `loader.py`                  | Analytical entry point         | Coordinates loading and metric integration       |
| `utils/load_json.py`         | Trajectory flattening          | Converts hierarchical JSON logs into DataFrames  |
| `add_metrics/base_metric.py` | Metric abstraction             | Defines the contract for analytical metrics      |
| `add_metrics/exampe.py`      | Trajectory statistics          | Implements concrete optimization metrics         |
| `add_metrics/orchestror.py`  | Metric execution engine        | Applies metrics across trajectories              |
| `add_scores/extractor.py`    | Sequence extraction            | Identifies unique sequences requiring evaluation |
| `add_scores/main_runner.py`  | Prediction enrichment pipeline | Re-scores sequences using predictive models      |
| `add_scores/updater.py`      | Result serialization           | Injects predictions back into JSON logs          |

---

# Directory Overview

## `loader.py`

### Responsibility

Acts as the **public API** for the entire loader subsystem.

### Input

```text
Folder containing optimization JSON logs
```

### Output

```text
Metric-enriched pandas DataFrame
```

### Internal workflow

```text
JSON logs
    ↓
parse_json_folder(...)
    ↓
Raw trajectory DataFrame
    ↓
calculate_trajectory_metrics(...)
    ↓
Merge metrics
    ↓
Final analytical DataFrame
```

---

## `utils/load_json.py`

### Responsibility

Transforms deeply nested optimization logs into a flat analytical structure.

### Theory

Optimization results are stored hierarchically:

```text
sequence
    ↓
optimizer
    ↓
interpreter
    ↓
model
    ↓
iterations
```

Analysis, however, requires observations:

```text
one iteration = one row
```

Therefore this module performs:

$$
\text{Nested JSON}
\rightarrow
\text{Long-format table}
$$

---

### Extracted information

For every iteration:

* sequence identifier,
* optimizer name,
* interpreter name,
* model type,
* iteration number,
* sequence state,
* objective score,
* model predictions,
* attribution weights,
* beam populations,
* GC content.

---

# Metrics Subsystem

---

## Theory

Metrics transform trajectories into measurable quantities:

$$
M :
X^{(t)}
\rightarrow
\mathbb{R}
$$

where

$$
X^{(t)}
$$

represents the sequence at iteration (t).

Metrics describe properties such as:

* exploration,
* convergence,
* diversity,
* attribution concentration.

---

# `add_metrics/base_metric.py`

## Responsibility

Defines the metric interface.

Every metric must implement:

```python
prepare(...)
calculate(...)
```

---

### Conceptual role

```text
Trajectory
    ↓
prepare()
    ↓
State initialization
    ↓
calculate()
    ↓
Metric values
```

---

# `add_metrics/exampe.py`

Contains concrete analytical metrics.

---

## HammingTrajectoryCalculator

### Theory

Measures distance from the initial sequence.

Given

$$
X^{(0)}
$$

and

$$
X^{(t)},
$$

the normalized Hamming distance is

$$
H(X^{(0)},X^{(t)})
==================

\frac{1}{L}
\sum_{i=1}^{L}
\mathbf{1}
\left(
X_i^{(0)}\neq X_i^{(t)}
\right)
$$

---

### Interpretation

```text
0.0 → identical to starting sequence

1.0 → every position changed
```

---

## ShannonEntropyCalculator

### Theory

Measures attribution dispersion.

Given attribution weights

$$
w_i,
$$

probabilities are computed as

$$
p_i
===

\frac{|w_i|}
{\sum_j |w_j|}
$$

Entropy is then

$$
H
=

-\sum_i p_i\log_2 p_i
$$

---

### Interpretation

Low entropy:

```text
few nucleotides dominate importance
```

High entropy:

```text
importance distributed broadly
```

---

# `add_metrics/orchestror.py`

## Responsibility

Executes metrics across trajectories.

---

### Flow

```text
DataFrame
    ↓
Group trajectories
    ↓
Initialize metrics
    ↓
Iterate through optimization steps
    ↓
Calculate metric values
    ↓
Collect outputs
```

---

### Grouping principle

Metrics are computed independently for each:

```text
(sequence,
 interpreter,
 optimizer)
```

combination.

This preserves trajectory identity.

---

# Prediction Enrichment Subsystem

---

## Theory

Optimization logs often contain sequences that were evaluated only by the models active during optimization.

The enrichment subsystem performs retrospective validation:

$$
S
:
X
\rightarrow
(f_1(X),f_2(X),...,f_n(X))
$$

where

$$
f_i
$$

are predictive models.

---

# `add_scores/extractor.py`

## Responsibility

Collect all unique sequences requiring evaluation.

---

### Sources of sequences

```text
Current optimization states
```

and

```text
Beam populations
```

---

### Transformation

```text
Nested JSON
    ↓
Unique sequence set
```

---

### Benefit

Avoids redundant inference.

If the same sequence appears multiple times:

```text
evaluate once
reuse everywhere
```

---

# `add_scores/main_runner.py`

## Responsibility

Coordinates the complete rescoring pipeline.

---

### Flow

```text
Discover JSON files
    ↓
Extract sequences
    ↓
Load predictive models
    ↓
Batch inference
    ↓
Update JSON logs
    ↓
Write enriched outputs
```

---

### Production role

Acts as the orchestration layer for large-scale retrospective validation.

---

# `add_scores/updater.py`

## Responsibility

Inject predictions back into optimization logs.

---

### Transformation

```text
Predictions dictionary
```

into

```json
"models_predictions": {
    ...
}
```

---

### Update locations

Predictions are inserted into:

```text
Optimization steps
```

and

```text
Beam population candidates
```

---

# End-to-End Flow

```text
Optimization JSON logs
        ↓
parse_json_folder()
        ↓
Trajectory DataFrame
        ↓
calculate_trajectory_metrics()
        ↓
Metric-enriched DataFrame
```

Optional prediction enrichment:

```text
Optimization JSON logs
        ↓
extract_unique_sequences()
        ↓
ModelManager prediction
        ↓
update_json_with_predictions()
        ↓
Enriched JSON logs
```

---

# Summary

The `loaders` package is the **analytical backend** of the optimization framework.

Its responsibilities are divided into three independent layers:

| Layer            | Responsibility                                |
| ---------------- | --------------------------------------------- |
| Loading          | Convert JSON trajectories into DataFrames     |
| Metrics          | Quantify optimization behavior                |
| Score enrichment | Re-evaluate sequences using additional models |