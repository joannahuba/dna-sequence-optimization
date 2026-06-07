# core/wrapper.py

import time

from typing import Dict, List, Optional, Any, Literal
from ..models.model_manager import ModelManager


class SequencePredictorModelWrapper:

    def __init__(
        self,
        model_type: Literal["ensemble", "single"],
        mode: Literal["optimization", "reconstruction"],
        sequences: List[str],
        model_manager: ModelManager,
        optimizers_list: List[Any],
        interpreters_list: List[Any]
    ):

        self.model_type = model_type
        self.mode = mode
        self.sequences = sequences
        self.model_manager = model_manager
        self.optimizers_list = optimizers_list
        self.interpreters_list = interpreters_list
        self.output = {}

    # -------------------------------------------------
    # OPTIMIZATION
    # -------------------------------------------------

    def OptimizeSequences(
        self,
        config: Optional[Dict] = None
    ):

        print("\n" + "=" * 80)
        print("[PIPELINE] OPTIMIZATION MODE")
        print("=" * 80)

        config = config or {}

        total_sequences = len(self.sequences)

        for idx, (seq_id, seq_data) in enumerate(
            self.sequences.items(),
            start=1
        ):

            start_time = time.time()

            seq = seq_data["sequence"]

            print(
                f"\n[SEQUENCE {idx}/{total_sequences}] "
                f"{seq_id}"
            )

            print(
                f"[INFO] Length = {len(seq)}"
            )

            interpreter_dict = {}

            for interpreter in self.interpreters_list:

                interpreter_name = (
                    interpreter.__class__.__name__
                )

                print(
                    f"[INFO] Running interpreter: "
                    f"{interpreter_name}"
                )

                interpretation = interpreter.explain(
                    model_manager=self.model_manager,
                    sequence=seq,
                    model_type=self.model_type
                )

                optimizer_dict = {}

                for optimizer in self.optimizers_list:

                    optimizer_name = (
                        optimizer.__class__.__name__
                    )

                    print(
                        f"[INFO] Running optimizer: "
                        f"{optimizer_name}"
                    )

                    result = optimizer.optimize(
                        sequence=seq,
                        model_manager=self.model_manager,
                        interpretation=interpretation,
                        config=config
                    )

                    optimizer_dict[
                        optimizer_name
                    ] = result

                    print(
                        f"[INFO] Optimizer finished: "
                        f"{optimizer_name}"
                    )

                interpreter_dict[
                    interpreter_name
                ] = {
                    "interpretation": interpretation,
                    "optimizers_results": optimizer_dict
                }

            self.output[seq_id] = interpreter_dict

            elapsed = (
                time.time() - start_time
            )

            print(
                f"[INFO] Finished sequence "
                f"{seq_id} "
                f"in {elapsed:.2f}s"
            )

        print("\n[PIPELINE] Optimization finished")

        return self.output

    # -------------------------------------------------
    # RECONSTRUCTION
    # -------------------------------------------------

    def ReconstructSequences(
        self,
        reconstruction_config: Optional[Dict] = None,
    ):
        print("\n" + "=" * 80)
        print("[PIPELINE] RECONSTRUCTION MODE")
        print("=" * 80)

        reconstruction_config = reconstruction_config or {}

        self.output = {}

        total_sequences = len(self.sequences)

        for idx, (seq_id, seq_data) in enumerate(
            self.sequences.items(),
            start=1
        ):

            start_time = time.time()

            seq = seq_data["sequence"]
            mutation_budget = seq_data["mutation_budget"]
            original_activity = seq_data["original_activity"]

            print(f"\n[SEQUENCE {idx}/{total_sequences}] {seq_id}")
            print(f"[INFO] Length = {len(seq)}")
            print(f"[INFO] Mutation budget = {mutation_budget}")
            print(f"[INFO] Target activity = {original_activity}")

            # FIX: clean config contract
            config = dict(reconstruction_config)
            config.update({
                "mutation_budget": mutation_budget,
                "target_expression": original_activity,
                "method": "reconstruction",
                "iterations": mutation_budget
            })

            interpreter_dict = {}

            for interpreter in self.interpreters_list:

                interpretation = interpreter.explain(
                    model_manager=self.model_manager,
                    sequence=seq,
                    model_type=self.model_type
                )

                optimizer_dict = {}

                for optimizer in self.optimizers_list:

                    result = optimizer.optimize(
                        sequence=seq,
                        model_manager=self.model_manager,
                        interpretation=interpretation,
                        config=config
                    )

                    optimizer_dict[optimizer.__class__.__name__] = result

                interpreter_dict[interpreter.__class__.__name__] = {
                    "interpretation": interpretation,
                    "optimizers_results": optimizer_dict
                }

            self.output[seq_id] = interpreter_dict

            print(f"[INFO] Finished sequence {seq_id} in {time.time()-start_time:.2f}s")

        print("\n[PIPELINE] Reconstruction finished")
        return self.output