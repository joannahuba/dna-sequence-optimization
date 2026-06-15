# Heading 1 (Optimization Execution Rule)
## Pipeline runner execution mapping parameters onto command line interface arguments
rule run_optimization_task:
    input:
        tsv = os.path.join(GLOBAL_CONFIG["paths"]["input_directory"], "{filename}.tsv"),
        exp_yaml = "config/experiments/{experiment}.yaml"
    output:
        json_out = os.path.join(GLOBAL_CONFIG["paths"]["results_directory"], "{experiment}", "{filename}_trajectory.json")
    log:
        std = os.path.join(GLOBAL_CONFIG["paths"]["results_directory"], "{experiment}", "logs", "{filename}.log")
    resources:
        gpus = 1
    run:
        # Configuration runtime extraction
        ## Fetch active parameters from the current wildcard configuration target
        exp_cfg = get_experiment_config(wildcards.experiment)
        
        models_arg = " ".join(exp_cfg["models"])
        optimizers_arg = " ".join(exp_cfg["optimizers"])
        interpreters_arg = " ".join(exp_cfg["interpreters"])
        
        # Command line instantiation block
        ## Construct execution string routing outputs directly to target results space
        shell(
            "python -m PromotorOptimizer.pipeline.main "
            "--input {input.tsv} "
            "--output {output.json_out} "
            "--objective {exp_cfg[objective]} "
            "--mutation-budget {exp_cfg[mutation_budget]} "
            "--iterations {exp_cfg[iterations]} "
            "--models {models_arg} "
            "--optimizers {optimizers_arg} "
            "--interpreters {interpreters_arg} "
            "--config-path {input.exp_yaml} "
            "> {log.std} 2>&1"
        )