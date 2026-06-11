from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Metric Calculator Abstract Base & Implementations
## Base definitions and specific implementations for trajectory metrics

class MetricCalculator(ABC):
    """
    Abstract base class defining the interface for trajectory metric computations.
    """

    @abstractmethod
    def prepare(self, trajectory: pd.DataFrame) -> None:
        """
        Pre-compute or extract trajectory-level constants before row-by-row iteration.

        :param trajectory: Sorted DataFrame containing a single trajectory.
        :type trajectory: pd.DataFrame
        """
        pass

    @abstractmethod
    def calculate(self, row: pd.Series) -> dict:
        """
        Compute metric values for a single row within the trajectory context.

        :param row: A single row representing an iteration state.
        :type row: pd.Series
        :return: A dictionary containing the metric name and computed value.
        :rtype: dict
        """
        pass
