# Sequence Optimizers Module — System Overview

## 1. Purpose of the System

This module implements **sequence-space optimization algorithms** for improving DNA sequences by maximizing a learned scoring function:

$$
X^* = \arg\max_{X \in S} f(X)
$$

Where:

* **X**: DNA sequence (A, C, G, T)
* **S**: discrete combinatorial sequence space
* **f(X)**: ML-based fitness / scoring model

The system solves this via **iterative search over mutation space**, combining:

* Mutation generation (exploration operator)
* Fitness evaluation (external model)
* Selection strategy (optimizer-specific logic)
* State tracking (beam / trajectory memory)

All optimizers share a **unified contract (`BaseOptimizer`)** but differ in:

* candidate generation strategy
* selection mechanism
* exploration vs exploitation control
* trajectory structure (beam vs single path)

---

# 2. High-Level Architecture

## 2.1 Module Structure

```
optimizers/
├── __init__.py
├── registry.py
├── strategies/
│   ├── base_optimizer.py
│   ├── beam_search.py
│   ├── beam_search_stochastic_boltzman.py
│   ├── beam_stochastic_mh.py
└── utils/
    ├── mutation_generator.py
```

---

## 2.2 Core Layers

### Layer 1 — Interface Layer

**File:** `strategies/base_optimizer.py`

Defines the **execution contract** for all optimizers.

### Layer 2 — Optimization Engines

**Files:**

* `beam_search.py` → deterministic beam search
* `beam_search_stochastic_boltzman.py` → Boltzmann sampling beam
* `beam_stochastic_mh.py` → Metropolis-Hastings single trajectory

Implements different **search dynamics over the same abstract interface**.

### Layer 3 — Mutation Engine

**File:** `utils/mutation_generator.py`

Implements:

* random mutation
* guided mutation (importance-aware)
* hybrid mutation (guided + random)
* positional ranking utilities

This is the **exploration kernel used by all optimizers**.

### Layer 4 — Registry / Factory

**File:** `registry.py`

Maps string identifiers → optimizer classes.

Enables runtime selection of optimization strategy.

---

# 3. Core Abstraction (BaseOptimizer)

## File: `strategies/base_optimizer.py`

### Theory Mapping

Defines a **generic iterative optimization process over discrete sequence space**.

Each optimizer must implement:

### 3.1 `initialize_search_state`

#### Theory

Defines initial condition:

$$
\mathcal{B}_0 = {X_0}
$$

plus metadata:

* beam / trajectory state
* hyperparameters
* mutation history

#### Role

Initializes:

* beam structure
* temperature (if stochastic)
* best observed solution
* mutation tracking map

---

### 3.2 `generate_candidate_pool`

#### Theory

Defines mutation operator:

$$
\mathcal{C}_t = \mathcal{M}(\mathcal{B}_t, I)
$$

Where:

* $\mathcal{M}$ = mutation function
* $I$ = importance / attribution map

#### Role

Produces:

* valid mutated sequences
* filtered by biological constraints
* deduplicated candidates

---

### 3.3 `update_generation_step`

#### Theory

Defines selection operator:

$$
\mathcal{B}_{t+1} = \mathcal{S}(\mathcal{C}_t, f)
$$

Where:

* $\mathcal{S}$ = selection rule
* $f$ = fitness function

#### Role

* ranks candidates
* applies deterministic or stochastic selection
* updates search state

---

# 4. Mutation Engine

## File: `utils/mutation_generator.py`

### Theory Role

Implements:

$$
X' = \text{Mutate}(X, P)
$$

Where mutation is controlled by:

* positional importance
* stochastic noise
* structural constraints (prefix/suffix locks)

---

## 4.1 Key Functions

### 4.1.1 Random Mutation

Uniform random substitution:
$$
P(b) = \frac{1}{3}
$$

Used for exploration.

---

### 4.1.2 Guided Mutation

Importance-driven mutation:

$$
P(i) \propto |I_i|
$$

$$
P(b|i) \propto I_{i,b}
$$

Used for exploitation.

---

### 4.1.3 Hybrid Mutation

Combination:

$$
\text{Mutate} = \lambda \cdot \text{guided} + (1-\lambda)\cdot \text{random}
$$

Used in all stochastic optimizers.

---

# 5. Optimizer Implementations (Core Logic)

---

# 5.1 Deterministic Beam Search

## File: `beam_search.py`

### Theory

Maintains top-K deterministic frontier:

$$
\mathcal{B}_{t+1} = \text{TopK}(\mathcal{C}_t)
$$

### Flow

1. Extract top-k important positions:
   $$
   S_i = \sum |I_{i,b}|
   $$

2. Generate all substitutions:
   $$
   X[i \rightarrow b], \forall b \in {A,C,G,T}
   $$

3. Evaluate all candidates

4. Select top-K sequences

---

### Behavior

* fully deterministic
* exhaustive local exploration
* no randomness

---

# 5.2 Stochastic Beam Search (Boltzmann)

## File: `beam_search_stochastic_boltzman.py`

### Theory

Uses Gibbs distribution:

$$
P(X_i) = \frac{e^{f(X_i)/T}}{\sum_j e^{f(X_j)/T}}
$$

### Flow

1. Generate stochastic mutations per parent
2. Pool all candidates globally
3. Compute Boltzmann probabilities
4. Sample K without replacement
5. Apply cooling:

$$
T_{t+1} = \gamma T_t
$$

---

### Behavior

* global competition across beam
* stochastic selection
* avoids local collapse

---

# 5.3 Metropolis-Hastings Optimizer

## File: `beam_stochastic_mh.py`

### Theory

Single trajectory Markov chain:

$$
P(\text{accept}) =
\min(1, e^{\Delta f / T})
$$

Where:

$$
\Delta f = f(X') - f(X)
$$

---

### Flow

1. Generate candidate mutations
2. Select best candidate
3. Compute acceptance probability
4. Accept/reject transition
5. Update temperature

---

### Behavior

* single-path search
* strong exploration early
* greedy convergence late

---

# 6. Registry System

## File: `registry.py`

### Purpose

Maps string identifiers → optimizer classes.

### Mapping

| Name                              | Class               |
| --------------------------------- | ------------------- |
| `beam_search`                     | BeamSearchOptimizer |
| `beam_search_stochastic_boltzman` | Boltzmann Beam      |
| `beam_stochastic_mh`              | Metropolis-Hastings |

---

### Role in System

Enables:

$$
\text{Optimizer} = \text{Factory}(\text{name})
$$

Used for:

* experiment configuration
* runtime switching of strategies
* pipeline automation

---

# 7. End-to-End Execution Flow

This is the **true system lifecycle**.

---

## STEP 1 — Initialization

```text
registry.load(["beam_search"])
```

↓

```text
initialize_search_state(sequence, config)
```

Creates:

* beam or trajectory
* best score tracker
* mutation history

---

## STEP 2 — Candidate Generation

```text
generate_candidate_pool(state, importance_maps)
```

Pipeline:

1. extract beam
2. compute importance aggregation
3. mutate sequences via MutationGenerator
4. validate sequences
5. return candidate pool

Output:

```text
C_t = [X1, X2, ..., Xn]
```

---

## STEP 3 — Scoring (external, not in module)

Each candidate is scored:

```text
f(X_i)
```

Output:

```text
[(X1, s1), (X2, s2), ...]
```

---

## STEP 4 — Selection / Update

```text
update_generation_step(state, scored_candidates)
```

Depends on optimizer:

* Beam Search → top-K
* Boltzmann → sampling
* Metropolis → accept/reject

Output:

```text
state_{t+1}
```

---

## STEP 5 — Repeat Loop

Loop continues until:

* convergence
* max iterations
* score threshold

---

# 8. Conceptual Summary

This system is:

### A discrete evolutionary search engine

| Component         | Meaning              |
| ----------------- | -------------------- |
| MutationGenerator | exploration operator |
| Beam state        | population memory    |
| f(X)              | fitness landscape    |
| update step       | selection operator   |
| temperature       | exploration control  |
| registry          | strategy dispatcher  |

---

# 9. Key Design Idea (Important)

All optimizers implement the same abstraction:

$$
\text{Search} = \text{Mutate} + \text{Score} + \text{Select}
$$

Differences are only in:

* how mutations are generated
* how selection is performed
* how much randomness is injected
