# Attribution and Interpretation Algorithms for Nucleic Acid Sequences

This document provides mathematical descriptio and computational examples of the deep learning model interpretation methods implemented in <library_name>

---

## Mathematical Foundations and Space Definitions

Let a DNA nucleotide sequence of length $L$ be represented as a one-hot encoded matrix $X \in \{0, 1\}^{L \times 4}$. The coordinates of the matrix are defined as:

$$X_{i,b} = \begin{cases} 1 & \text{if nucleotide } b \text{ occurs at position } i \\ 0 & \text{otherwise} \end{cases}$$

where the position index is $i \in \{1, \dots, L\}$, and the base index $b \in \{1, 2, 3, 4\}$ corresponds to the ordered set of bases $\mathcal{A} = \{\text{A}, \text{C}, \text{G}, \text{T}\}$.

The prediction of the neural network model (e.g., for regulatory activity) is represented as a differentiable function $f: \mathbb{R}^{L \times 4} \to \mathbb{R}$, which maps the continuous input representation space to a continuous real value (correlated with an expression metric, such as the RNA/DNA ratio).

The task of the interpreter is to determine the feature importance attribution matrix $I \in \mathbb{R}^{L \times 4}$, where the value $I_{i,b}$ defines the relative impact of the presence or modification of base $b$ at position $i$ on the global target function $f(X)$.

---

## 1. Local Sensitivity Maps (`SaliencyInterpreter`)

### Structural Description

The Vanilla Saliency method represents the simplest form of gradient-based attribution, based on the first term of the Taylor series expansion of the function around the input point $X$. It determines the local linear geometric sensitivity of the model to infinitesimal changes in individual input channels.

### Mathematical Formulation

The importance matrix $I_{\text{Saliency}}$ is defined as the magnitude of the partial derivative of the target function with respect to the input tensor:

$$I_{\text{Saliency}}(i, b) = \left| \frac{\partial f(X)}{\partial X_{i,b}} \right|$$

To maintain consistency in ensemble mode, gradients are averaged across $M$ independent predictive architectures before applying the absolute value operation:

$$I_{\text{Saliency}}^{\text{ensemble}}(i, b) = \left| \frac{1}{M} \sum_{m=1}^{M} \frac{\partial f_m(X)}{\partial X_{i,b}} \right|$$

### Computational Example

Consider a micro-sequence of length $L=3$: `ATG`. Its one-hot representation $X$ of dimensions $3 \times 4$ is given by:


$$X = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

Assume that the backward pass of the network returns the following tensor of output gradients $\nabla_X f(X)$:


$$\nabla_X f(X) = \begin{bmatrix} -0.45 & +0.12 & +0.05 & -0.02 \\ +0.10 & -0.05 & +0.22 & -0.85 \\ -0.15 & +0.70 & +0.30 & -0.10 \end{bmatrix}$$

Applying the absolute value operator yields the final importance matrix $I_{\text{Saliency}}$:


$$I_{\text{Saliency}} = \begin{bmatrix} 0.45 & 0.12 & 0.05 & 0.02 \\ 0.10 & 0.05 & 0.22 & 0.85 \\ 0.15 & 0.70 & 0.30 & 0.10 \end{bmatrix}$$

---

## 2. Integrated Gradients with Biological GC Baseline (`IntegratedGradientsInterpreter`)

### Structural Description

The Integrated Gradients (IG) method resolves the fundamental problem of gradient saturation in deep neural networks by integrating the derivative vectors along a linear interpolation path between a reference sequence (baseline) $X'$ and the input sequence $X$.

Unlike standard computer vision implementations where a tensor of all zeros is used as $X'$ (which represents a non-physical state of missing nucleotides in biology), SCAN generates a continuous baseline matched to the actual GC content ($GC\%$) of the evaluated sequence. This prevents the interpolation trajectory from escaping the probability simplex and stabilizes the statistics of the network's normalization layers.

### Mathematical Formulation

The formal continuous definition of the integrated gradient for position $i$ and base $b$ takes the form:

$$\text{IG}_{i,b}(X) = (X_{i,b} - X'_{i,b}) \times \int_{0}^{1} \frac{\partial f\left(X' + \alpha(X - X')\right)}{\partial x_{i,b}} d\alpha$$

Where $X' \in \mathbb{R}^{L \times 4}$ is the compositional background matrix, defined based on the global fraction $p_{\text{GC}}$ of the input sequence with a controlled stochastic fluctuation $\epsilon \sim \mathcal{U}(-0.05, 0.05)$:

$$p_{\text{GC}}^* = \max\left(0.0, \min\left(1.0, \frac{\text{Count}(\text{G}) + \text{Count}(\text{C})}{L} + \epsilon\right)\right)$$

Reference values for each row $i$ are assigned as a probabilistic background:

$$X'_{i,b} = \begin{cases} \frac{p_{\text{GC}}^*}{2} & \text{for } b \in \{\text{C}, \text{G}\} \\ \frac{1.0 - p_{\text{GC}}^*}{2} & \text{for } b \in \{\text{A}, \text{T}\} \end{cases}$$

In a discrete space, the integral is approximated using a Riemann sum across $M$ interpolation steps:

$$\text{IG}_{i,b}^{\text{approx}}(X) = (X_{i,b} - X'_{i,b}) \times \frac{1}{M} \sum_{m=1}^{M} \frac{\partial f\left(X' + \frac{m}{M}(X - X')\right)}{\partial x_{i,b}}$$

The final result is transformed using the absolute value operator: $I_{\text{IG}}(i, b) = |\text{IG}_{i,b}^{\text{approx}}(X)|$.

### Computational Example

Consider a position $i$ where the actual sequence contains the base $\text{G}$ ($X_{i} = [0, 0, 1, 0]$). Assume that the global $GC\%$ of the sequence is $40\%$, and the sampled deviation is $\epsilon = 0.0$, meaning $p_{\text{GC}}^* = 0.4$.

The background reference vector is:


$$X'_{i} = \left[ \frac{1 - 0.4}{2}, \frac{0.4}{2}, \frac{0.4}{2}, \frac{1 - 0.4}{2} \right] = [0.3, 0.2, 0.2, 0.3]$$

The input difference $(X_{i} - X'_{i})$ is:


$$\Delta X_i = [0 - 0.3, 0 - 0.2, 1 - 0.2, 0 - 0.3] = [-0.3, -0.2, +0.8, -0.3]$$

Assume that after performing $M=3$ interpolation steps, the averaged gradient vector $\frac{1}{M}\sum \nabla f$ is:


$$\text{Grad}_{\text{avg}} = [+0.5, -0.1, +2.0, +0.4]$$

We calculate the Hadamard product for this position:


$$\text{IG}_{i}^{\text{approx}} = \Delta X_i \odot \text{Grad}_{\text{avg}} = [-0.3 \times 0.5, -0.2 \times (-0.1), 0.8 \times 2.0, -0.3 \times 0.4] = [-0.15, +0.02, +1.60, -0.12]$$

After applying the absolute value, the position's importance profile is:


$$I_{\text{IG}}(i) = [0.15, 0.02, 1.60, 0.12]$$

---

## 3. In Silico Mutagenesis Mapping (`InSilicoMutagenesis`)

### Structural Description

In Silico Mutagenesis (ISM) is a non-gradient, brute-force attribution method. It represents an empirical approach (finite differences), simulating physical saturation mutagenesis experiments. This method explicitly determines the change in model prediction for every possible single-nucleotide substitution at each position of the sequence.

### Mathematical Formulation

Let $X[i \leftarrow b]$ denote the modified one-hot matrix where the presence of base $b \in \mathcal{A}$ is forced in row $i$. The importance value in the ISM attribution matrix is defined as the absolute difference between the score of the mutated sequence and the score of the wild-type sequence:

$$I_{\text{ISM}}(i, b) = \left| f(X[i \leftarrow b]) - f(X) \right|$$

For positions where $b$ corresponds to the base originally present in the output sequence, the value is identically zero: $I_{\text{ISM}}(i, b) = 0$.

### Computational Example

Consider position $i=5$, where the nucleotide $\text{A}$ occurs in the original sequence. The initial prediction score of the model is $f(X) = 1.250$.

We sequentially perform three independent substitutions in the evaluation loop and pass the sequences forward through the network (forward pass):

1. Substitution of $\text{C}$: $f(X[5 \leftarrow \text{C}]) = 1.450 \implies I_{\text{ISM}}(5, \text{C}) = |1.450 - 1.250| = 0.200$
2. Substitution of $\text{G}$: $f(X[5 \leftarrow \text{G}]) = 0.850 \implies I_{\text{ISM}}(5, \text{G}) = |0.850 - 1.250| = 0.400$
3. Substitution of $\text{T}$: $f(X[5 \leftarrow \text{T}]) = 1.200 \implies I_{\text{ISM}}(5, \text{T}) = |1.200 - 1.250| = 0.050$

For the original base $\text{A}$: $I_{\text{ISM}}(5, \text{A}) = 0$.

The complete attribution vector for position 5 takes the form:


$$I_{\text{ISM}}(5) = [0.000, 0.200, 0.400, 0.050]$$

---

## Method Summary and Operational Comparison

| Method Feature | Vanilla Saliency | Integrated Gradients | In Silico Mutagenesis (ISM) |
| --- | --- | --- | --- |
| **Mathematical Type** | Gradient-based (Local) | Gradient-based (Path) | Non-gradient (Perturbative) |
| **Computational Cost** | Very low (1 backward step) | Medium ($M$ backward steps) | Very high ($3 \times L$ forward steps) |
| **Saturation Phenomenon** | Susceptible | Resilient | Completely resilient |
| **Physical Interpretation** | Direction of local improvement | Total feature contribution energy | Direct fitness gain/loss |