# main.py

import argparse
from PromotorOptimizer.pipeline.configs import PipelineConfig
from PromotorOptimizer.pipeline.runner import PipelineRunner


#TODO runner 
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--output")

    parser.add_argument("--task-mode")
    parser.add_argument("--mutation-budget", type=int, default=None)

    parser.add_argument("--models", nargs="+")
    parser.add_argument("--optimizers", nargs="+")
    parser.add_argument("--interpreters", nargs="+")

    parser.add_argument("--iterations", type=int, default=50)

    return parser.parse_args()


def main():

    args = parse_args()

    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        task_mode=args.task_mode,
        mutation_budget=args.mutation_budget,
        models=args.models,
        optimizers=args.optimizers,
        interpreters=args.interpreters,
        iterations=args.iterations,
    )

    runner = PipelineRunner(config)
    runner.run()


if __name__ == "__main__":
    main()