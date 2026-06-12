# Sequence Optimization Algorithms Reference

This document provides the formal mathematical and structural descriptions of the sequence search strategies implemented in the SCAN (Stochastic Cooling Track Analysis) framework.

---

## 1. Classical Deterministic Beam Search (`BeamSearchOptimizer`)

### Description
Classical Beam Search is a heuristic, breadth-first graph search algorithm that optimizes a sequence by maintaining a fixed-size pool of the top-$K$ highest-scoring candidates (where $K$ represents the `beam_width`). In each iterative step, the algorithm reduces the dimensionality of the attribution matrix to isolate the top-$P$ most sensitive coordinates (defined by `top_k_positions`). For each position, a systematic, deterministic substitution scan is performed for all alternative nucleotides.

### Mathematical Framework
Let $\mathcal{B}_t = \{X_1, X_2, \dots, X_K\}$ be the beam population at step $t$. For each parent sequence $X \in \mathcal{B}_t$, the position importance vector $S \in \mathbb{R}^L$ is derived from the attribution matrix $I \in \mathbb{R}^{L \times 4}$:

$$S_i = \begin{cases} \max_{b} I_{i,b} & \text{if method is ISM} \\ \sum_{b=1}^4 |I_{i,b}| & \text{if method is Gradient-based} \end{cases}$$

The top-$P$ coordinate spaces are mapped via sorting the indices:

$$\mathcal{P} = \text{argsort}(S)[::-1][:P]$$

The candidate pool $\mathcal{C}$ is generated via deterministic mutation sweeps:

$$\mathcal{C} = \bigcup_{X \in \mathcal{B}_t} \bigcup_{i \in \mathcal{P}} \bigcup_{b \neq X[i]} \{ X[i \leftarrow b] \}$$

The population for the subsequent step is filtered by selecting the top unique elements:

$$\mathcal{B}_{t+1} = \text{top-K} \left( \text{Deduplicate}(\mathcal{C}) \right)$$

---

## 2. Stochastic Beam Search with Metropolis-Hastings (`StochasticBeamSearchMetropolis`)

### Description
This variant introduces local thermodynamic stochasticity into the multi-trajectory beam pool. Instead of evaluating all possible local mutations deterministically, it spawns $N$ stochastic child mutations per parent node. Each child sequence undergoes an isolated transition check against its direct progenitor using the Metropolis-Hastings acceptance criterion scaled by a decaying global temperature parameter $T$.

### Mathematical Framework
For a parent sequence $X_{\text{parent}}$ with fitness score $f(X_{\text{parent}})$, a single nucleotide mutation yields a child $X_{\text{child}}$. The transition probability $P_{\text{accept}}$ governing the lineage persistence is defined as:

$$\Delta = f(X_{\text{child}}) - f(X_{\text{parent}})$$

$$P_{\text{accept}} = \begin{cases} 1 & \text{if } \Delta > 0 \\ e^{\frac{\Delta}{T_t}} & \text{if } \Delta \le 0 \end{cases}$$

Where the system temperature decays geometrically across iterations:

$$T_t = T_0 \cdot \gamma^t, \quad \gamma \in (0, 1)$$

The global beam pool is assembled from the accepted lineage trajectories and ranked to preserve the top-$K$ unique states.

---

## 3. Stochastic Beam Search with Boltzmann Sampling (`StochasticBeamSearchBoltzmann`)

### Description
An ensemble-level stochastic optimization method derived from statistical mechanics. Rather than executing localized independent parent-child battles, all generated children across all beam tracks are pooled into a unified generation layer. The selection of the tracking pool for the next generation is executed concurrently by sampling without replacement from a Gibbs (Boltzmann) categorical probability distribution.

### Mathematical Framework
Let $\mathcal{C} = \{X_1, X_2, \dots, X_M\}$ be the complete pool of valid mutated children generated across the entire beam. The raw fitness scores are normalized using the step maximum to prevent numerical overflow during exponential scaling. The categorical probability $P(X_i)$ of selecting sequence $X_i$ into the next beam layer is formalized as:

$$P(X_i) = \frac{\exp\left(\frac{f(X_i) - \max_j f(X_j)}{T_t}\right)}{\sum_{k=1}^M \exp\left(\frac{f(X_k) - \max_j f(X_j)}{T_t}\right)}$$

The tracking beam $\mathcal{B}_{t+1}$ is sampled stochastically from the distribution:

$$\mathcal{B}_{t+1} = \text{RandomChoice}\left(\mathcal{C}, \text{size}=K, \text{replace}=\text{False}, \mathbf{p}=\mathbf{P}\right)$$

---

## 4. Discrete Simulated Annealing (`SimulatedAnnealingOptimizer`)

### Description
A single-trajectory stochastic search profile modeled after metal annealing processes. It explores the sequence space by evaluating a single mutating coordinate state at a time. The algorithm balances structural exploration and fine-tuning exploitation by transitioning from high-temperature chaotic jumps to low-temperature greedy hill-climbing.

### Mathematical Framework
At step $t$, the system holds a single active state sequence $X_t$. A single-nucleotide substitution is drawn using the background positional saliency map to generate candidate $X_{\text{candidate}}$. The state update follows the classical thermal transition:

$$X_{t+1} = \begin{cases} X_{\text{candidate}} & \text{if } \text{Random}(0,1) < \min\left(1, e^{\frac{f(X_{\text{candidate}}) - f(X_t)}{T_t}}\right) \\ X_t & \text{otherwise} \end{cases}$$

The coordinate mask tracks all accepted historical positions to prevent infinite oscillatory loops across identical nucleotide loci.