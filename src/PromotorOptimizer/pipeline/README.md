# Pipeline

## Structure

```text
pipeline/
├── configs.py
├── main.py
└── runner.py
```

---

# Overview

The `pipeline` module is the **top-level orchestration layer** of the framework. It does not perform optimization itself. Instead, it coordinates all lower-level components and defines **how an experiment is executed from start to finish**.

Conceptually, the pipeline performs the following sequence:

```text
Experiment configuration
        ↓
Component loading
(models, objectives, optimizers, interpreters)
        ↓
Sequence loading
        ↓
Wrapper initialization
        ↓
Trajectory execution
        ↓
Result serialization
```

Its role is therefore to transform an experiment specification into reproducible optimization trajectories.

---

# Mapping Theory to Files

| File         | Responsibility                                   | Concept              |
| ------------ | ------------------------------------------------ | -------------------- |
| `configs.py` | Defines the experiment specification             | Configuration layer  |
| `main.py`    | Parses CLI arguments and YAML overrides          | User interface layer |
| `runner.py`  | Instantiates components and executes experiments | Orchestration layer  |

---

# File Overview

## `configs.py`

Defines the `PipelineConfig` dataclass containing all experiment settings:

* input and output paths,
* objective function,
* models,
* optimizers,
* interpreters,
* iteration limits,
* mutation budgets,
* validation rules,
* runtime overrides.

This file represents the **experiment contract**.

---

## `main.py`

Provides the command-line entry point.

Responsibilities:

1. Parse terminal arguments,
2. Load optional YAML configuration files,
3. Build a `PipelineConfig`,
4. Start the pipeline through `PipelineRunner`.

This file forms the **interface between the user and the framework**.

---

## `runner.py`

Coordinates the entire execution process.

Responsibilities:

1. Load objectives from `ObjectiveRegistry`,
2. Load models using `ModelRegistry`,
3. Load optimizers using `OptimizerRegistry`,
4. Load interpreters using `InterpreterRegistry`,
5. Read input sequences,
6. Initialize `SequencePredictorModelWrapper`,
7. Execute optimization trajectories,
8. Save results to disk.

This file acts as the **central orchestration engine**.

---

# Execution Flow

```text
main.py
    ↓
Parse CLI arguments
    ↓
Create PipelineConfig
    ↓
Initialize PipelineRunner
    ↓
Load objectives, models, optimizers, interpreters
    ↓
Load input sequences
    ↓
Initialize SequencePredictorModelWrapper
    ↓
Execute trajectories
    ↓
Save results as JSON
```

---

# Responsibility Summary

| Component    | Responsibility                                               |
| ------------ | ------------------------------------------------------------ |
| `configs.py` | Defines what experiment should run                           |
| `main.py`    | Converts user input into configuration                       |
| `runner.py`  | Builds and coordinates all framework components              |
| Wrapper      | Executes optimization trajectories                           |
| Registries   | Instantiate objectives, models, optimizers, and interpreters |
| Output JSON  | Stores experiment results                                    |

