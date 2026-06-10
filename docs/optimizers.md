

## Implemented optimizers 

### Beam search

.. class:: optimizers.beam_search.BeamSearchOptimizer(validation_config, beam_width=30, candidates_per_parent=10, iterations=50)

   Bases: :class:`optimizers.base_optimizer.BaseOptimizer`

   An iterative, heuristic sequence optimizer that implements the Beam Search algorithm. 
   It explores the discrete space of DNA sequences by maintaining a fixed-size pool (beam) 
   of the most promising sequences at each step, mitigating the risk of trapping in local minima 
   inherent to pure greedy local searches.

   The optimizer supports two core execution layouts defined via the configuration:
   
   * **Optimization Mode**: Iteratively maximizes the predicted regulatory expression score.
   * **Reconstruction Mode**: Minimizes the absolute error relative to a target reference expression profile.

   :param validation_config: Dictionary containing parameter bounds utilized to initialize the internal sequence validator pipeline.
   :type validation_config: dict
   :param beam_width: The maximum number of active sequences maintained in the tracking pool at the end of each iteration loop. Default is 30.
   :type beam_width: int
   :param candidates_per_parent: The number of alternative single-nucleotide mutated child sequences generated from each parent sequence per step. Default is 10.
   :type candidates_per_parent: int
   :param iterations: The maximum execution depth for sequence modification tracks when not restricted by mutation budgets. Default is 50.
   :type iterations: int

   .. method:: optimize(sequence, model_manager, interpretation, config)

      Executes the multi-trajectory search to optimize the biological sequence toward the specified evaluation target.

      The execution pipeline handles automated sequence parsing, mutation extraction, constraint validation, and performance recording:

      # Execution Pipeline Architecture
      ## Configuration parsing
      The method extracts operational variables including mutation limits, execution mode indicators, and objective targets from the config dictionary.

      ## Adapter domain tracking
      It safely detects if the underlying trained models utilize non-adapter representations by checking the metadata footprint against `DNADatasetNoAdapters`. If matched, it dynamically locks the constant terminal sequences to protect them from downstream mutagenesis.

      ## Multi-parent trajectory exploration
      During each evolutionary loop step:
      
      * Every sequence currently residing inside the beam is evaluated.
      * If valid, multiple child variants are spawned through hybrid importance-score-driven probabilistic substitutions.
      * Offspring that violate biological sequence rules (e.g., structural length mismatch or homopolymer repeats) are instantly pruned out via short-circuit validation evaluation.

      ## Beam ranking and tracking
      All validated children across all lineages are aggregated, evaluated concurrently through the unified model manager suite, sorted, and the top-performing instances are retained to form the next generation beam pool.

      :param sequence: The initial raw wild-type or disrupted DNA nucleotide string to start the refinement process from.
      :type sequence: str
      :param model_manager: The interface coordinating concurrent inference evaluations across the group ensemble models.
      :type model_manager: ModelManager
      :param interpretation: Interpretation object containing computed baseline importance attribution arrays for guided mutation sampling.
      :type interpretation: Interpretation
      :param config: Runtime properties dictionary containing execution values (`method`, `mutation_budget`, `target_expression`, `input_path`).
      :type config: dict
      :return: A structured result map containing the optimized sequence, tracking trajectory logs across iterations, and fitness calculation indicators.
      :rtype: dict