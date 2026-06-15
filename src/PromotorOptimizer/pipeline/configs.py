# Heading 1 (Pipeline Configuration Domain)
## Strongly typed structural dataclasses for runtime experiment tracking
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class PipelineConfig:
    """
    Strongly typed runtime configuration layer capturing file coordinates,
    algorithmic selection limits, and objective properties for the pipeline.
```sql
    Example adjusting options via programmatic instantiation:
    >>> config = PipelineConfig(
    ...     input_path="data.tsv",
    ...     output_path="results.json",
    ...     objective="reconstruction"
    ... )
    """
    input_path: str
    output_path: str
    objective: str

    # Algorithmic constraint fields
    mutation_budget: Optional[int] = None
    iterations: int = 50

    # Component registration lists
    models: List[str] = field(default_factory=list)
    optimizers: List[str] = field(default_factory=list)
    interpreters: List[str] = field(default_factory=list)

    # Performance and aggregation settings
    ensemble: bool = True
    seed: int = 42

    # Global structural configuration profiles passed to the objective constructor
    objective_config: Dict[str, Any] = field(default_factory=lambda: {
        "penalty_std": 0.2
    })
    
    # Specific parameter overrides mapped by class names
    optimizer_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    interpreter_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Biological sequence nucleotide validation matrix boundaries
    validation_config: Optional[Dict[str, Any]] = None