# Pipeline explanation

This document explains pipeline workflow, idea behind and how to implement additional optimizers.

## Workflow graph

```
main.py 
  │
  ▼
pipeline/runner.py ──[Loads Ensembles via]──> models/registry.py ──> ModelManager
  │
  ├──[Loads Interpreters via]──> interpretation/registry.py
  │
  ├──[Loads Optimizers via]──> optimizers/registry.py
  │
  ▼
core/wrapper.py (SequencePredictorModelWrapper)
  │
  ▼ Loop over each Sequence:
  │
  ├──> BaseInterpreter.explain() ──> Returns InterpretationResult (Attribution Maps)
  │      │
  │      ▼ Loop over each Optimizer passed to the wrapper:
  │
  └────────> BaseOptimizer.optimize(interpretation) 
               │
               ├──> MutationGenerator (Generates variations based on attribution)
               │
               └──> SequenceValidator (Filters structural and biological violations)

```

### Step 1: Initialization (`main.py` & `pipeline/runner.py`)

The execution begins at `main.py` which parses command-line parameters and populates the `PipelineConfig` object. This configuration dictates the execution mode (`optimization` or `constrained_recovery`), specifies tracking depths (`--iterations`), and sets path tokens.
`PipelineRunner` catches this configuration object and instantiates the runtime pipeline. It loads specified network blocks via `ModelRegistry`, registers importance tools via `InterpreterRegistry`, and sets up constraints within `OptimizerRegistry`.

* **Key Parameters**:
* `max_homopolymer_at` (int): Maximum consecutive A/T bases allowed (default: 8).
* `max_homopolymer_gc` (int): Maximum consecutive G/C bases allowed (default: 12).
* `gc_percent_range` (tuple): Minimum and maximum acceptable global GC ratio (default: 0.25 to 0.8).
* `min_length` / `max_length` (int): Boundary constraints forcing sequence window invariants (default: 230).



### Step 2: Pipeline Orchestration (`core/wrapper.py`)

`SequencePredictorModelWrapper` intercepts the loaded model manager suite, sequence batches, optimizers, and interpreters. It executes a combinatorial execution pattern across two modes:

* **Optimization Mode (`OptimizeSequences`)**: Maximizes the average ensemble raw prediction score output.
* **Reconstruction Mode (`ReconstructSequences`)**: Inverts targets by minimizing the absolute error distance against the benchmark ground-truth `original_activity` value.

For every single sequence, it runs all loaded interpreters to compute an initial static attribution landscape. It then passes this importance matrix directly into the optimization tracks.

### Step 3: Sequence Interrogation (`interpretation/`)

Interpreters derive feature importance matrices across the $L \times 4$ spatial sequence dimension. The pipeline reads three native methods:

* `SaliencyInterpreter`: Computes a single first-order derivative step backpropagation pass to map instantaneous output sensitivity.
* `IntegratedGradientsInterpreter`: Integrates sequential step-gradients along a linear trajectory between a null zero-baseline matrix and the encoded instance to resolve feature saturation problems.
* `InSilicoMutagenesis`: Systematically scans every single alternate base variant across every index location to empirically document real prediction deviations ($\Delta$).

### Step 4: Iterative Optimization Loops (`optimizers/`)

Optimizers refine the structural contents of the sequences. They read the pre-computed attribution weights to bias mutations toward highly reactive coordinates. During child generation steps, the algorithm hooks into `MutationGenerator` for probabilistic nucleotide changes and immediately routes candidates through `SequenceValidator` to filter out biological or structural rule-breakers before scoring.

---

#TODO  (move to optimizers docs)

## Developer part

### Inserting new optimizers 

All sequence optimizers must inherit from the uniform blueprint interface `BaseOptimizer` located inside `optimizers/base_optimizer.py`. To plug a novel optimization algorithm into the core pipeline, follow this two-step implementation specification.

#### 1. Define the Optimizer Module

Create a new file under the `optimizers/` subdirectory (e.g., `optimizers/simulated_annealing.py`). Implement the class framework.

```python
# optimizers/simulated_annealing.py

import math
import random
import numpy as np
from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .validator import SequenceValidator


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    An algorithmic sequence optimizer implementing discrete Simulated Annealing.

    It escapes local local sub-optimal minima by allowing transient stochastically 
    bounded drops in regulatory objective scores based on a Metropolis acceptance criterion.

    :param validation_config: Configuration keys mapping tracking constraints for validation checks.
    :type validation_config: dict
    :param initial_temperature: The starting scalar control state for the cooling map. Default is 10.0.
    :type initial_temperature: float
    :param cooling_rate: Geometric degradation multiplier applied per iteration loop step. Default is 0.95.
    :type cooling_rate: float
    """

    def __init__(
        self,
        validation_config,
        initial_temperature=10.0,
        cooling_rate=0.95
    ):
        # Resource initialization
        ## Bind short-circuit structural validator
        self.validator = SequenceValidator(validation_config)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def optimize(
        self,
        sequence,
        model_manager,
        interpretation,
        config
    ):
        """
        Executes a stochastically guided cooling track to optimize the DNA input string.

        :param sequence: Target wild-type or broken base string context.
        :type sequence: str
        :param model_manager: Unified ensemble model coordination stack interface.
        :type model_manager: ModelManager
        :param interpretation: Result dataclass matching extracted position attribution arrays.
        :type interpretation: InterpretationResult
        :param config: Runtime dictionary payload mapping target modes, labels, and mutation limits.
        :type config: dict
        :return: Map structure compiling the optimal string sequence and performance log tracks.
        :rtype: dict
        """
        # Parse runtime parameters
        ## Extract configuration modes and bounds
        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)
        iterations = config.get("iterations", 100)

        importance = interpretation.importance_scores

        # Define localized performance evaluators
        ## Raw prediction averaging
        def score(seq):
            result = model_manager.predict_sequences([seq])
            return sum(result[seq].values()) / len(result[seq])

        ## Absolute objective distance evaluation
        def reconstruction_score(seq):
            return -abs(score(seq) - target_expression)

        # Optimization track setup
        ## Initialize baseline states
        current_seq = sequence
        current_score = reconstruction_score(sequence) if method == "reconstruction" else score(sequence)
        
        best_seq = current_seq
        best_score = current_score
        
        temperature = self.initial_temperature
        trajectory = []

        # Core optimization loop execution
        for it in range(iterations):
            ## Generate candidate via directed mutation mutations
            ### Draw mutation step single position parameters
            candidate_seq = MutationGenerator.hybrid_mutation(
                sequence=current_seq,
                importance_scores=importance,
                n_mutations=1
            )

            ## Validate structural and biological properties
            if not self.validator.is_valid(candidate_seq):
                continue

            ## Score candidate variation
            candidate_score = (
                reconstruction_score(candidate_seq)
                if method == "reconstruction"
                else score(candidate_seq)
            )

            ## Evaluate state transition
            delta = candidate_score - current_score

            ### Apply Metropolis-Hastings acceptance conditions
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_seq = candidate_seq
                current_score = candidate_score

                #### Keep track of global historical peak
                if current_score > best_score:
                    best_score = current_score
                    best_seq = candidate_seq

            ## Log execution state metrics
            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq,
                "temperature": temperature
            })

            ## Cool down the system state
            temperature *= self.cooling_rate

        # Assemble result output map
        result_map = {
            "best_sequence": best_seq,
            "trajectory": trajectory
        }

        if method == "reconstruction":
            predicted = score(best_seq)
            result_map["predicted_activity"] = predicted
            result_map["reconstruction_error"] = abs(predicted - target_expression)
        else:
            result_map["best_score"] = best_score

        return result_map

```

#### 2. Register the New Optimizer

Open `optimizers/registry.py` and register your new class matching a literal token identifier inside the dynamic loading sequence loop:

```python
# optimizers/registry.py

from .beam_search import BeamSearchOptimizer
from .simulated_annealing import SimulatedAnnealingOptimizer # Add import line

class OptimizerRegistry:

    @staticmethod
    def load(names, validation_config=None):
        registry = []

        for name in names:
            if name == "beam_search":
                registry.append(BeamSearchOptimizer(validation_config))
            elif name == "simulated_annealing": # Add mapping branch
                registry.append(SimulatedAnnealingOptimizer(validation_config))
            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry

```
