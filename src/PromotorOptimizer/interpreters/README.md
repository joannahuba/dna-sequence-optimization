# 1. High-Level Overview

This module implements **post-hoc attribution methods for DNA sequence models**.

### Core idea

Given:

* Input DNA sequence → encoded as one-hot tensor
  $X \in \mathbb{R}^{L \times 4}$
* Model function
  $f(X) \to \mathbb{R}) (regulatory activity / score$

The system computes:

* Attribution matrix
  $I \in \mathbb{R}^{L \times 4}$

Meaning:

> “Which nucleotide at which position contributed how much to the model output?”

---

# 2. Mapping Theory → Code Structure

## 2.1 File Responsibility Map

### Root module

| File                     | Role                                                           |
| ------------------------ | -------------------------------------------------------------- |
| `interpreter_manager.py` | Orchestrates encoding, model selection, and strategy execution |
| `registry.py`            | Factory that maps string names → interpreter classes           |

---

### Strategy implementations (`strategies/`)


| File                           | Theory method      | Meaning                                      |
|--------------------------------|--------------------|----------------------------------------------|
| `base_attribution_strategy.py` | Abstract interface | Defines contract for all attribution methods |
| `saliency.py`                  | Gradient-based     | Local gradient sensitivity                   |
| `integrated_gradients.py`      | Path integral      | Baseline → input gradient accumulation       |
| `mutagenesis.py`               | Finite differences | Brute-force mutation scoring                 |

Mathematical definitions:

* Saliency:

  $$
  I(X) = \left|\frac{\partial f(X)}{\partial X}\right|
  $$

* Integrated Gradients:

  $$
  IG(X) = (X-X') \int_0^1 \frac{\partial f(X'+\alpha(X-X'))}{\partial X} d\alpha
  $$

* In Silico Mutagenesis:

  $$
  I(i,b) = |f(X[i \leftarrow b]) - f(X)|
  $$

---

# 3. Implementation Mapping

## 3.1 Base Interface (Contract Layer)

### File

`base_attribution_strategy.py`

### Theory role

Defines:

> “Every attribution method must compute (I \in \mathbb{R}^{L \times 4}) from (X, f).”

### Code responsibility

* Ensures uniform API:

```python
compute_tensor_attribution(
    tensor_x,
    model_instance,
    metadata
)
```

### What it enforces

Every method must:

* accept tensor input
* run model
* compute attribution matrix
* return CPU list `[L][4]`

---

## 3.2 Saliency (Local Gradient Method)

### File

`saliency.py`

### Theory

$$
I_{i,b} = \left|\frac{\partial f(X)}{\partial X_{i,b}}\right|
$$

### Code flow

1. Clone input tensor
2. Enable gradients
3. Forward pass
4. Backprop once
5. Take absolute gradient

### Execution steps

```text
X → model → f(X)
        ↓
   backward()
        ↓
    grad(X)
        ↓
|grad| = attribution
```

### Key interpretation

* Measures **instant sensitivity**
* No perturbation
* No baseline
* Single backward pass

---

## 3.3 Integrated Gradients (Path Method)

### File

`integrated_gradients.py`

### Theory

$$
I = (X - X') \int_0^1 \nabla f(X' + \alpha(X - X')) d\alpha
$$

### Code structure

#### Step 1: baseline generation

```python
baseline = np.full(..., 0.25)
```

→ corresponds to biological GC-balanced reference

---

#### Step 2: interpolation path

For each step:

```python
interpolated = baseline + α(X - baseline)
```

---

#### Step 3: gradient accumulation

For each α:

* forward pass
* backward pass
* collect gradients

---

#### Step 4: integration approximation

$$
\frac{1}{M} \sum \nabla f
$$

---

#### Step 5: final attribution

```python
(X - baseline) * avg_grad
```

---

### Interpretation

* Captures **total feature contribution**
* Fixes saturation problem in saliency
* Biologically grounded baseline

---

## 3.4 In Silico Mutagenesis (Finite Difference)

### File

`mutagenesis.py`

### Theory

$$
I_{i,b} = |f(X_{i \to b}) - f(X)|
$$

---

### Code flow

#### Step 1: baseline prediction

```python
base_loss = f(X)
```

---

#### Step 2: generate mutation space

For each position:

* replace A/C/G/T → generate 4×L mutated sequences

---

#### Step 3: batch inference

```text
X_mut_pool → model → predictions
```

---

#### Step 4: compute delta

```python
|f(mut) - f(original)|
```

---

### Interpretation

* Most **biologically faithful**
* No gradients
* Expensive O(4L forward passes)

---

# 4. Interpreter Manager (Orchestration Layer)

## File

`interpreter_manager.py`

---

## Role in theory

This is the **execution controller**:

> “Convert raw sequence + model + strategy → attribution matrix”

---

## Pipeline flow

### Step 1: encode DNA sequence

```python
X_raw = encode_batch([sequence])
```

Maps:

$$
\text{ATGC} \to X \in \mathbb{R}^{L \times 4}
$$

---

### Step 2: model resolution

```python
model = model_manager.get_models()[model_name]
```

Selects correct trained network

---

### Step 3: tensor conversion

```python
X_tensor = torch.tensor(...)
```

Moves into GPU/CPU pipeline

---

### Step 4: strategy execution

```python
strategy.compute_tensor_attribution(...)
```

This is the polymorphic call:

| Strategy | Computation   |
| -------- | ------------- |
| Saliency | gradients     |
| IG       | path integral |
| ISM      | mutation scan |

---

### Step 5: output conversion

```python
attribution_tensor.cpu().numpy().tolist()
```

Final format:

```
[L][4]
```

---

# 5. Registry System (Strategy Factory)

## File

`registry.py`

---

## Role

Maps string names → actual classes:

```python
"saliency" → SaliencyInterpreter
"mutagenesis" → InSilicoMutagenesis
```

---

## Flow

### Step 1: user request

```python
["saliency", "integrated_gradients"]
```

---

### Step 2: registry lookup

```python
strategy_cls = INTERPRETER_MAP[name]
```

---

### Step 3: dependency injection

Each strategy gets:

```python
strategy(objective=BaseLossObjective)
```

---

## Why this matters

This creates:

* plug-and-play interpretability
* shared loss objective contract
* consistent scoring across methods

---

# 6. End-to-End Execution Flow (FULL SYSTEM)

This is the actual runtime pipeline:

---

## INPUT

```
sequence = "ATGCGT..."
model_name = "model_1"
strategy = "integrated_gradients"
```

---

## FLOW

### 1. Manager receives request

```
InterpreterManager.interpret_single_model_step()
```

---

### 2. Encoding

```
DNA → one-hot tensor
```

---

### 3. Model selection

```
ModelManager → model instance
```

---

### 4. Strategy instantiation

```
Registry → IntegratedGradientsStrategy()
```

---

### 5. Attribution computation

Depending on strategy:

| Method   | Internal loop         |
| -------- | --------------------- |
| Saliency | 1 backward            |
| IG       | M interpolation steps |
| ISM      | 4×L forward passes    |

---

### 6. Output

```
[[...L x 4 matrix...]]
```

Each cell:

> importance of nucleotide b at position i

---

# 7. Conceptual Summary (Unified Theory View)

All methods approximate:

$$
I = \text{importance}(X, f)
$$

but differ in approximation:

| Method               | Approximation type |
| -------------------- | ------------------ |
| Saliency             | local derivative   |
| Integrated Gradients | line integral      |
| Mutagenesis          | finite difference  |
