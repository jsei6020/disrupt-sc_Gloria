"""
Initial State Executor for the new clean architecture.

This executor handles initial state (static equilibrium) simulations
using the new separated architecture with distinct components for
execution, data collection, analysis, and export.
"""

from typing import TYPE_CHECKING
import logging

from .base_executor import BaseExecutor
from ..core.interfaces import SimulationRunner, DataCollector
from ..runners.initial_state_runner import InitialStateRunner
from ..collection.base_collector import StandardDataCollector

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


class InitialStateExecutor(BaseExecutor):
    """
    Executes initial state (static equilibrium) simulation using the new architecture.
    
    This replaces the old InitialStateExecutor with a clean separation:
    - InitialStateRunner: Executes the model.run_static() simulation
    - StandardDataCollector: Collects agent data during execution
    - CompositeAnalyzer: Analyzes the collected data
    - CompositeExporter: Exports the analysis results (optional)
    """
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        super().__init__(model, parameters)
        self.logger = logging.getLogger(__name__)
        
    def create_runner(self) -> SimulationRunner:
        """Create the InitialStateRunner for static simulation execution."""
        return InitialStateRunner(self.model, self.parameters)
        
    def create_collector(self) -> DataCollector:
        """Create the StandardDataCollector for data collection."""
        return StandardDataCollector(self.parameters)


class StationaryTestExecutor(BaseExecutor):
    """
    Executes stationary test simulation using the new architecture.
    
    This replaces the old StationaryTestExecutor.
    """
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        super().__init__(model, parameters)
        self.logger = logging.getLogger(__name__)
        
    def create_runner(self) -> SimulationRunner:
        """Create runner for stationary test simulation."""
        from ..runners.base_runner import BaseSimulationRunner
        from ..core.data_structures import SimulationData
        
        class StationaryTestRunner(BaseSimulationRunner):
            def _execute_simulation_logic(self) -> SimulationData:
                # Execute stationary test simulation
                simulation_result = self.model.run_stationary_test()
                
                # Collect data from the simulation result
                collector = StandardDataCollector(self.parameters)
                collector.collect_time_step(self.model, 0)
                
                return collector.get_simulation_data()
        
        return StationaryTestRunner(self.model, self.parameters)
        
    def create_collector(self) -> DataCollector:
        """Create collector for stationary test data."""
        return StandardDataCollector(self.parameters)


class FlowCalibrationExecutor(BaseExecutor):
    """
    Executes flow calibration simulation using the new architecture.
    
    This replaces the old FlowCalibrationExecutor and includes the
    calibration analysis as part of the analysis phase.
    """
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        super().__init__(model, parameters)
        self.logger = logging.getLogger(__name__)
        
    def create_runner(self) -> SimulationRunner:
        """Create runner for flow calibration simulation."""
        from ..runners.base_runner import BaseSimulationRunner
        from ..core.data_structures import SimulationData
        
        class FlowCalibrationRunner(BaseSimulationRunner):
            def _execute_simulation_logic(self) -> SimulationData:
                # Execute static simulation for flow calibration
                simulation_result = self.model.run_static()
                
                # Perform calibration calculations
                from disruptsc.model.utils.functions import mean_squared_distance
                
                calibration_flows = simulation_result.report_annual_flow_specific_edges(
                    self.parameters.flow_data, 
                    self.model.transport_edges,
                    self.parameters.time_resolution, 
                    usd_or_ton='ton'
                )
                
                # Calculate calibration metric
                calibration_metric = mean_squared_distance(calibration_flows, self.parameters.flow_data)
                
                # Log calibration results
                self.logger.info(f"Calibration flows: {calibration_flows}")
                self.logger.info(f"Calibration metric: {calibration_metric}")
                
                # Collect data
                collector = StandardDataCollector(self.parameters)
                collector.collect_time_step(self.model, 0)
                
                # Add calibration data to metadata
                simulation_data = collector.get_simulation_data()
                simulation_data.metadata['calibration_flows'] = calibration_flows
                simulation_data.metadata['calibration_metric'] = calibration_metric
                
                return simulation_data
        
        return FlowCalibrationRunner(self.model, self.parameters)
        
    def create_collector(self) -> DataCollector:
        """Create collector for flow calibration data."""
        return StandardDataCollector(self.parameters)