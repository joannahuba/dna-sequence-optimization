# Deep Learning for Regulatory DNA Sequence Optimization
#TODO - jakąś fajną nazwe tej biblioteki dać, bo to akurat jest akurat dostosowane do dowolnego modelu 

#TODO  bardzo krótki opis co robi ta bibliotea

<libraby name> sevres us ... (czy coś takeigo )

***

## Introduction

In this repostitory we created  computational engineering pipeline designed to restore disrupted enhancer sequences and optimize synthetic promoter activity. Using deep learning models as an *in silico* simulation environment, we navigate the complex cis-regulatory grammar to maximize or recover the `rna_dna_ratio`. 

To achieve this, we have developed a modular, structured library architecture that enables independent manipulation of individual components, facilitating high-throughput testing across diverse models.

### Key Features

#TODO 
- że możemy dowolny model podlączyć i go optymlizozwać 
- że zapewnia zoptymalizowany sposób wyknywana tej optymalizacji na wielu modelach, i jest dostosowana do dowolnego modelu
- zawiera predefinoiwne modlee do znajdywnaia sekwencji promotorów

## Repository Structure

```plaintext
.
├── config/                         # Runtime configuration space
│   ├── experiments/                # Declarative experiment configuration YAML layers
│   └── global_config.yaml          # Immutable environment and path definitions
├── data/                           # Data asset workspace
│   ├── checkpoints/                # Serialized PyTorch network weights (.pth)
│   ├── preprocessed/               # Formatted input sequence datasets (.tsv)
│   └── results_final/              # Final exported optimization output matrices
├── docs/                           # Comprehensive library documentation hierarchy
│   ├── approach/                   # Biological justifications and domain introductions
│   ├── architecture/               # Code structures, managers, and pipeline routing
│   ├── theory/                     # Mathematical proofs, notations, and formal definitions
│   └── how_to/                     # Execution manuals and reproducibility guides
├── src/
│   └── PromotorOptimizer/          # Core library package source tree
│       ├── core/                   # Trajectory orchestrators and sequence wrappers
│       ├── interpreters/           # Gradient backpropagation attribution strategies
│       ├── loss_functions/         # Evaluation objectives and fitness constraints
│       ├── models/                 # Model managers and network registry stacks
│       ├── optimizers/             # Search space exploration strategies
│       └── pipeline/               # CLI definitions, runtime configurations, and entrypoints
└── workflow/                       # Reproducible orchestration layer
    ├── Snakefile.smk               # Master directed acyclic graph (DAG) execution point
    └── rules/                      # Sectional pipeline compilation rule scripts
```

## Documentation Structure

### Description

The complete system documentation is decoupled across specialized categorical subdirectories inside the docs/ workspace:
- [docs/approach/](./docs/approach/promotors_introduction_and_model_justification.md): (#TODO - uzupełnic ) Biological context, enhancer recovery principles, and rationale for using deep learning networks as target simulators.
- [docs/architecture/](./docs/architecture/README.md): Architectural diagrams, module descriptions, class registry configurations, and workflow execution blueprints.
- [docs/theory/](./docs/theory/README.md): Mathematical formalizations of local sensitivity maps, integrated path gradients, and stochastically cooled search heuristics.
- [docs/how_to/](./docs/how_to/snakemake_reproducibility.md): Execution manual for cluster reproducibility using Snakemake pipeline parameter

### Table of contents

#TODO - insert table of contents fo all docs (later) 



## Usage

### Installation

To be able to use library, you need to install it in developer mode

```bash
pip install -e .
```

You also need to install PyTorch library separately 

***

### Quick Execution

The execution graph is managed via Snakemake. To execute an isolated experiment target using the explicit structural path routing:

```bash
# in general
snakemake -s workflow/Snakefile.smk results/<config_experiment_name>/<input_files>.json --cores n
# example
snakemake -s workflow/Snakefile.smk results/test01_reconstruction/reconstruction_input_1_2_trajectory.json --cores 1

```

For complete parameter configurations and workflow controls, refer to [docs/how_to/snakemake_reproducibility.md](./docs/how_to/snakemake_reproducibility.md)`.


***

## Authors

#TODO - wypisać nas i co kto zrobił (zrobie sam )


