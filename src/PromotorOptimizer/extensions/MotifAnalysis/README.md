# MotifAnalysis Pipeline

---

## Structure

```text id="v9q1lm"
MotifAnalysis/
├── scanner.py
├── pipeline.py
├── jaspar.py
└── plot.py
```

---

# Overview

The MotifAnalysis module implements a **biological sequence annotation pipeline** that detects transcription factor (TF) binding motifs in DNA sequences and maps them to known TF databases.

The system connects three layers:

```text id="m8k3p0"
DNA sequence
    ↓
Motif detection (MOODS + JASPAR PWMs)
    ↓
TF annotation (TOMTOM / JASPAR mapping)
    ↓
Visualization of motif landscape
```

---

## Core theoretical idea

The pipeline implements the following biological abstraction:

1. DNA sequences contain regulatory motifs.
2. Motifs can be represented as Position Weight Matrices (PWMs).
3. PWMs can be scanned across sequences using log-odds scoring.
4. Detected motifs can be mapped to known transcription factors.
5. Motif occurrences can be visualized along genomic coordinates.

Mathematically:

```text id="7v8m2q"
sequence → PWM scan → score matrix → hit locations → TF annotation
```

---

# Mapping Theory to Files

| File          | Responsibility                                 | Theory layer                             |
| ------------- | ---------------------------------------------- | ---------------------------------------- |
| `scanner.py`  | PWM scanning using MOODS + JASPAR              | Motif detection (core signal extraction) |
| `pipeline.py` | Loads sequences and runs scanner               | Execution wrapper for scanning           |
| `jaspar.py`   | Maps motifs → transcription factors via TOMTOM | Biological annotation layer              |
| `plot.py`     | Visualizes motif positions along sequence      | Interpretability / visualization layer   |

---

# File Descriptions

---

## `scanner.py` — Motif detection engine

### Role

This is the **core computational biology engine**.

It performs:

* retrieval of PWMs from JASPAR,
* conversion into log-odds scoring matrices,
* genome-wide scanning using MOODS,
* extraction of motif hits.

---

### Theory

Each motif is represented as a PWM:

```text id="q3kz7d"
P(A), P(C), P(G), P(T)
```

Converted into log-odds:

```text id="2n0c1v"
log(P(base) / background)
```

Scanning computes:

```text id="j0p9aa"
score(position) = sum log-odds over motif length
```

Hits are positions where score exceeds threshold.

---

### Output

Returns a dataframe:

```text id="k3m1qv"
seq_id
TF_name
start
end
score
strand
motif_sequence
```

---

## `pipeline.py` — Execution layer

### Role

This file is a **minimal orchestration wrapper**.

It:

* loads FASTA sequences,
* calls the scanner,
* returns raw motif detection results.

---

### Theory

This is the **data ingestion + scanning pipeline stage**:

```text id="x9k2lm"
FASTA → sequence dictionary → MOODS scanning → motif hits
```

No biological interpretation is done here.

---

### Output

```text id="p4m1aa"
(df motifs, fasta dictionary)
```

---

## `jaspar.py` — TF annotation layer

### Role

This file performs **motif-to-transcription factor mapping** using:

* TOMTOM similarity search
* JASPAR database reference motifs

---

### Theory

Motifs discovered by scanning are not directly TF labels.

So mapping is performed:

```text id="z1v8qq"
MEME motif → similarity search → JASPAR TF match
```

Mathematically:

```text id="v7k2op"
argmin distance(motif, JASPAR_PWM)
```

---

### Functions

* `run_tomtom()` → executes external motif comparison tool
* `parse_tomtom()` → extracts TF mapping results
* `annotate()` → attaches TF names to detected motifs

---

### Output transformation

```text id="m2k9xz"
motif_name → TF_name
```

---

## `plot.py` — Visualization layer

### Role

Transforms raw motif hits into a **genomic landscape plot**.

---

### Theory

Each motif occurrence is represented as:

```text id="b8m1pq"
(start, end, TF, score)
```

Visualization encodes:

* position along DNA axis
* TF identity → color
* score → line thickness
* overlapping motifs → separate tracks

---

### Key idea

Instead of showing raw sequences, the system builds:

```text id="c4k8aa"
1D genomic feature map
```

---

# Execution Flow

## Step 1 — Input loading

```text id="a9m2qv"
FASTA file
```

↓

```python id="m2p9wx"
load_fasta()
```

↓

dictionary:

```text id="k1v8zz"
{seq_id: sequence}
```

---

## Step 2 — Motif scanning (`scanner.py`)

```text id="z9k2lm"
sequence → MOODS scan → hits
```

Output:

* motif positions
* scores
* motif identities

---

## Step 3 — Pipeline wrapper (`pipeline.py`)

* passes FASTA to scanner
* returns raw motif dataframe

No annotation yet.

---

## Step 4 — TF annotation (`jaspar.py`)

Flow:

```text id="q2m8aa"
motif hits
    ↓
TOMTOM similarity search
    ↓
JASPAR TF mapping
    ↓
annotated dataframe
```

Final structure:

```text id="t8m1vv"
motif → TF_name
```

---

## Step 5 — Visualization (`plot.py`)

Flow:

```text id="p9k2qq"
annotated dataframe + sequence
        ↓
position mapping
        ↓
track assignment per TF
        ↓
genomic plot rendering
```

---

# Complete System Flow

```text id="v1k8zz"
FASTA input
    ↓
pipeline.py (load + dispatch)
    ↓
scanner.py (PWM scanning with MOODS)
    ↓
jaspar.py (TF annotation via TOMTOM + JASPAR)
    ↓
plot.py (genomic visualization)
```

---

# Responsibility Summary

| Component     | Role                                |
| ------------- | ----------------------------------- |
| `scanner.py`  | Detect motif occurrences using PWMs |
| `pipeline.py` | Orchestrate scanning pipeline       |
| `jaspar.py`   | Map motifs to transcription factors |
| `plot.py`     | Visualize motif distribution        |
