# Theoretical Foundations and Notational Framework

This directory contains the formal mathematical documentation for the attribution models and sequence search heuristics implemented within the <library_name>.

---

## Global Mathematical Conventions

To maintain consistency across all theoretical descriptions, we define a unified coordinate and sequence space:

* Let $\mathcal{A} = \{\text{A}, \text{C}, \text{G}, \text{T}\}$ be the ordered alphabet of biological bases.
* A DNA sequence of length $L$ is represented as a one-hot encoded boolean matrix $X \in \{0, 1\}^{L \times 4}$, where:
  $$X_{i,b} = \begin{cases} 1 & \text{if nucleotide } b \text{ occurs at position } i \\ 0 & \text{otherwise} \end{cases}$$
  for $i \in \{1, \dots, L\}$ and $b \in \{1, 2, 3, 4\}$.
* An individual deep neural network model is defined as a differentiable mapping function $f_m: \mathbb{R}^{L \times 4} \to \mathbb{R}$, evaluating the relative transcription asset output ratio.
* The system configuration pool contains $M$ independent architectures, indexed by $m \in \{1, \dots, M\}$.
* The global importance matrix computed by an interpreter is denoted as $I \in \mathbb{R}^{L \times 4}$, tracking spatial attribution energy values.

---

## Detailed Theoretical Components

The mathematical specifications for each component are isolated into the following files:

### 1. [Attribution and Interpretation Models](interpreters_theoretical.md)
Contains the continuous calculus expressions, baseline sampling methods, and coordinate reduction operations for feature validation:
* **Vanilla Saliency Mapping**: First-order Taylor series approximations tracking instantaneous local linear geometric sensitivity.
* **Integrated Gradients (IG)**: Path-integral approximations over an affine interpolation trajectory utilizing a continuous, GC-matched background reference baseline $X'$.
* **In Silico Mutagenesis (ISM)**: Brute-force empirical derivative simulations analyzing absolute output response deltas $\Delta$.

### 2. [Sequence Space Optimization Heuristics](optimizers_theoretical.md)
Formalizes state updates, probabilistic transitions, and selection distributions for structural searching:
* **Deterministic Beam Search**: Breadth-first graph searches pruning combinations down to a fixed tracking window width $K$.
* **Metropolis-Hastings MCMC Sequences**: Stochastically guided local steps bounded by a geometric cooling matrix schedule $T_t = T_0 \cdot \gamma^t$.
* **Gibbs / Boltzmann Sampling Tracks**: Population-level categorical selection executed without replacement over exponential normalization vectors.


### #TODO - score functions 