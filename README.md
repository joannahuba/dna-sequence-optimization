# Documentation

## Code Workflow

1. **Input Normalization:** Accepts promoter sequences as raw strings or nested configurations and standardizes them into a unified format. 
2. **Trajectory Collection:** Passes sequences to optimizers and captures **all intermediate and final variants** (the entire mutation history) generated during the optimization process.
3. **Cascading Validation:** Filters the accumulated sequence pool using fast C-level Regex (`SequenceValidator`). Instantly flags variants that violate structural constraints (length boundaries, GC-content range, or homopolymer repeats).
4. **Virtual RAM Storage:** Streams all unique candidate sequences into an in-memory virtual TSV file (`tempfile`), bypassing disk I/O.
5. **Batch PyTorch Prediction:** Binds the virtual file to model-specific `Dataset` classes and feeds them via `DataLoader` to calculate forward-pass regression metrics (`predicted_rna_dna_ratio`) in parallel batches.
6. **Overfitting Assessment:** Computes a composite `min_model_score`

---

## Minimal Usage Template

```python
import torch
from wrapper import SequencePredictorModelWrapper

# 1. Prepare Target Models & Custom Datasets
loaded_model_1 = torch.load("path_to_model_alpha.pt")
loaded_model_2 = torch.load("path_to_model_beta.pt")

class NativeDNADataset:
    """Your project's Dataset class that tokenizes text into tensors."""
    pass

models_dict = {
    "model_alpha": {
        "model": loaded_model_1,
        "dataset_class": NativeDNADataset  # Pass the class reference, NOT an instance
    },
    "model_beta": {
        "model": loaded_model_2,
        "dataset_class": NativeDNADataset
    }
}

# 2. Setup Active Optimizers & Biological Constraints
sequence_optimizer = CustomEvolutionaryOptimizer()

validation_config = {
    "min_length": 20,
    "max_length": 60,
    "max_homopolymer_at": 4,
    "gc_percent_range": (0.4, 0.6)
}

# 3. Instantiate and Execute Pipeline
wrapper = SequencePredictorModelWrapper(
    models_dict=models_dict,
    optimizers_list=[sequence_optimizer],
    validation_config=validation_config
)

input_data = {
    "promoter_ref_01": "ATCGATCGATCGATCGATCGATCG"
}

results = wrapper.OptimizeSequence(input_data)
```

## Output Schema (proposal_table)
- The method returns a dictionary of lists. Converting a sample result directly to a pandas.DataFrame yields the following columns:
- sequence (str): The specific mutant variant from the optimization timeline.
- original_sequence (str): The wild-type/starting input sequence (baseline).
- optimizer_source (str): Class name of the optimizer that produced the variant.
- optimizer_internal_score (float | None): Metrics returned by the generator itself.
- score_[model_name] (float): Continuous evaluation ratio computed by that individual model.
- min_model_score (float): Core evaluation metric — minimum function over all model predictions.
- is_valid (bool): Physical/synthesis feasibility flag from the validator.

--- 
Asia

# 1. How to Run the Project

## 1.1 Basic Execution

Run the pipeline from `main.py`:

```bash
python main.py \
  --input data/sequences.tsv \
  --output results.json \
  --task-mode constrained_recovery \
  --models deepstarr \
  --optimizers beam_search \
  --interpreters saliency \
  --iterations 100
```

---

## 1.2 Arguments

### Required

* `--input`
  Path to input TSV file:

  ```
  id sequence
  seq1 ATCG...
  seq2 GCTA...
  ```

* `--output`
  Path to output JSON file

* `--task-mode`
  One of:

  * `optimization`
  * `constrained_recovery`

* `--models`
  List of model names (from registry)

* `--optimizers`
  List of optimizer names (from registry)

* `--interpreters`
  List of interpreters (from registry)

---

### Optional

* `--iterations` (default: 50)
  Number of optimization iterations

* `--mutation-budget`
  Used only in constrained recovery mode

---

## 1.3 Example Runs

### Optimization mode

```bash
python main.py \
  --input data/sequences.tsv \
  --output results.json \
  --task-mode optimization \
  --models deepstarr \
  --optimizers beam_search \
  --interpreters saliency \
  --iterations 100
```

### Constrained recovery mode

```bash
PYTHONPATH=src uv run main.py \
  --input data/sequences.tsv \
  --output results.json \
  --task-mode constrained_recovery \
  --models deepstarr \
  --optimizers beam_search \
  --interpreters saliency \
  --mutation-budget 5 \
  --iterations 100
```

---

# 2. Pipeline Architecture

The pipeline is structured as:

```
main.py
  ↓
PipelineRunner
  ↓
ModelRegistry → ModelManager
OptimizerRegistry → Optimizers
InterpreterRegistry → Interpreters
  ↓
SequencePredictorModelWrapper
  ↓
(interpreter → optimizer loop)
  ↓
Sequence optimization results
```

---

## 2.1 Execution Flow

For each input sequence:

1. **Interpreter step**

   * computes importance scores for sequence positions
   * returns `InterpretationResult`

2. **Optimizer step**

   * uses interpretation + model predictions
   * generates mutated sequences
   * filters invalid sequences (SequenceValidator)
   * selects best candidates iteratively

3. **Output**

   * best sequence
   * best score
   * full optimization trajectory

---

# 3. Adding a New Optimizer

## 3.1 Required Interface

All optimizers must implement:

```python
class BaseOptimizer(ABC):

    @abstractmethod
    def optimize(
        self,
        sequence,
        model_manager,
        interpretation,
        config
    ):
        pass
```

---

## 3.2 Implementation Example

Example: Random Search Optimizer

```python
from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator

class RandomSearchOptimizer(BaseOptimizer):

    def __init__(self, validation_config, iterations=50):
        self.validator = SequenceValidator(validation_config)
        self.iterations = iterations

    def optimize(self, sequence, model_manager, interpretation, config):

        best_seq = sequence
        best_score = 0
        trajectory = []

        for i in range(self.iterations):

            candidate = MutationGenerator.random_mutation(sequence)

            if not self.validator.is_valid(candidate):
                continue

            score = model_manager.predict_sequences([candidate])
            score = sum(score[candidate].values())

            if score > best_score:
                best_score = score
                best_seq = candidate

            trajectory.append({
                "iteration": i,
                "sequence": best_seq,
                "score": best_score
            })

        return {
            "best_sequence": best_seq,
            "best_score": best_score,
            "trajectory": trajectory
        }
```

---

## 3.3 Register the Optimizer

Edit:

```
src/PromotorOptimizer/optimizers/registry.py
```

```python
from .beam_search import BeamSearchOptimizer
from .random_search import RandomSearchOptimizer


class OptimizerRegistry:

    @staticmethod
    def load(names, validation_config=None):

        registry = []

        for name in names:

            if name == "beam_search":
                registry.append(
                    BeamSearchOptimizer(validation_config)
                )

            elif name == "random_search":
                registry.append(
                    RandomSearchOptimizer(validation_config)
                )

            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry
```

---

## 3.4 Use it

```bash
--optimizers beam_search random_search
```

---

# 4. Adding a New Interpreter

## 4.1 Base Interface

All interpreters must implement:

```python
class BaseInterpreter(ABC):

    @abstractmethod
    def explain(
        self,
        model_manager,
        sequence: str,
        model_type
    ):
        pass
```

---

## 4.2 Example: Random Importance Interpreter

```python
import torch
import random
from .base import BaseInterpreter
from ..core.types import InterpretationResult

class RandomInterpreter(BaseInterpreter):

    def explain(self, model_manager, sequence, model_type):

        length = len(sequence)

        importance = torch.rand(length)

        return InterpretationResult(
            method_name="random",
            importance_scores=importance,
            sequence=sequence,
            model_scores={}
        )
```

---

## 4.3 Register Interpreter

Edit:

```
src/PromotorOptimizer/interpretation/registry.py
```

```python
from .saliency import SaliencyInterpreter
from .integrated_gradients import IntegratedGradientsInterpreter
from .mutagenesis import InSilicoMutagenesis
from .random_interpreter import RandomInterpreter


class InterpreterRegistry:

    @staticmethod
    def load(names: list):

        registry = []

        for name in names:

            if name == "saliency":
                registry.append(SaliencyInterpreter())

            elif name == "integrated_gradients":
                registry.append(IntegratedGradientsInterpreter())

            elif name == "mutagenesis":
                registry.append(InSilicoMutagenesis())

            elif name == "random":
                registry.append(RandomInterpreter())

            else:
                raise ValueError(f"Unknown interpreter {name}")

        return registry
```

---

## 4.4 Use it

```bash
--interpreters saliency random
```

## TO DO na ten tydzień
skończenie interpreterów, pododawanie opcji ile chce sie mutacji, aby interpretacje były per pozycja zasada a nie per pozycja
dodanie k-merów

dodanie drugiego/innych modelów narazie jest tylko mój

dokonczenie guided mutation (dodanie opcji w optimizerze)

dodanie konfliktów modeli-> chyba w optimizerze zapisywać dla każdego zapisu

