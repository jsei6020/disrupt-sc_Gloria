"""
Disruption Executor for the new clean architecture.

This executor handles single disruption simulations using the new 
separated architecture with distinct components for execution,
data collection, analysis, and export.
"""

from typing import TYPE_CHECKING
import logging

from .base_executor import BaseExecutor
from ..core.interfaces import SimulationRunner, DataCollector
from ..runners.disruption_runner import DisruptionRunner
from ..collection.base_collector import StandardDataCollector

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


class DisruptionExecutor(BaseExecutor):
    """
    Executes single disruption simulation using the new architecture.
    
    This replaces the old DisruptionExecutor with a clean separation:
    - DisruptionRunner: Executes the model.run_disruption() simulation
    - StandardDataCollector: Collects agent data during time steps
    - CompositeAnalyzer: Analyzes the collected data for losses/impacts
    - CompositeExporter: Exports the analysis results (optional)
    """
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        super().__init__(model, parameters)
        self.logger = logging.getLogger(__name__)
        
    def create_runner(self) -> SimulationRunner:
        """Create the DisruptionRunner for disruption simulation execution."""
        return DisruptionRunner(self.model, self.parameters)
        
    def create_collector(self) -> DataCollector:
        """Create the StandardDataCollector for time-series data collection."""
        return StandardDataCollector(self.parameters)