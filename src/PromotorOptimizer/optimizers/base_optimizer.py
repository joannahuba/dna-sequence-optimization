# optimizers/base_optimizer.py

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    @abstractmethod
    def optimize(
        self,
        sequence,
        model_manager,
        interpreter,
        config
    ):
        pass