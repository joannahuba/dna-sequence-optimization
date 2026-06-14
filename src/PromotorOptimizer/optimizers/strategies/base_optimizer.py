# Heading 1 (Optimization Contract Architecture)
## Infrastructure imports and typing components
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class defining the execution step contract for 
    all trajectory search heuristics and sequence design algorithms.
    """

    @abstractmethod
    def initialize_search_state(
        self, 
        sequence: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Initializes the persistent state tracking dictionary for a sequence trajectory.

        :param sequence: Input wild-type or seed nucleotide sequence string.
        :type sequence: str
        :param config: Extracted runtime parameters matching the optimizer_config block.
        :type config: Dict[str, Any]
        :return: Initialized state dictionary tracking parameters, beams, and lineage maps.
        :rtype: Dict[str, Any]
        """
        pass

    @abstractmethod
    def generate_candidate_pool(
        self, 
        search_state: Dict[str, Any], 
        importance_maps: Dict[str, Any]
    ) -> List[str]:
        """
        Spawns validation-filtered child sequence variants utilizing position importance metrics.

        :param search_state: Current active state dictionary tracking generations.
        :type search_state: Dict[str, Any]
        :param importance_maps: Attribution scores derived from the active interpreter strategy.
        :type importance_maps: Dict[str, Any]
        :return: Unique collection list of validated mutated sequence strings.
        :rtype: List[str]
        """
        pass

    @abstractmethod
    def update_generation_step(
        self, 
        search_state: Dict[str, Any], 
        scored_candidates: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """
        Applies mathematical selection algorithms to filter candidates and transition the search state.

        :param search_state: Current active state dictionary tracking generations.
        :type search_state: Dict[str, Any]
        :param scored_candidates: Collection of generated sequences paired with unified fitness scores.
        :type scored_candidates: List[Tuple[str, float]]
        :return: Updated state dictionary configuration map for the subsequent optimization step.
        :rtype: Dict[str, Any]
        """
        pass