"""
Sensitivity Executor for the new clean architecture.

This executor handles sensitivity analysis across parameter combinations using the new 
batch processing architecture with distinct components for execution, data collection, 
analysis, and export.
"""

from typing import TYPE_CHECKING, List, Dict, Any
import logging
import gc
import itertools
from copy import deepcopy
from datetime import datetime

from .base_executor import BaseExecutor
from ..core.interfaces import BatchProcessor
from ..core.data_structures import BatchResult, AnalysisResults
from .disruption_executor import DisruptionExecutor
from ..collection.base_collector import StandardDataCollector

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


def _reset_model_state(model, parameters, full_reset: bool = False):
    """Reset model state for each sensitivity iteration using configured caching."""
    if full_reset:
        model.setup_transport_network(False, parameters.with_transport)
        model.setup_agents(False)
        model.setup_sc_network(False)
        model.set_initial_conditions()
        model.setup_logistic_routes(False)
    else:
        # Use standard caching configuration for sensitivity analysis
        model.setup_transport_network(True, parameters.with_transport)
        model.setup_agents(False)  # Usually rebuild agents for sensitivity
        model.setup_sc_network(False)  # Usually rebuild SC network
        model.set_initial_conditions()
        model.setup_logistic_routes(False)  # Usually rebuild routes


class SensitivityExecutor(BaseExecutor, BatchProcessor):
    """
    Executes sensitivity analysis across parameter combinations using the new architecture.
    
    This replaces the old SensitivityExecutor with batch processing using:
    - BatchProcessor interface for handling multiple parameter combinations
    - DisruptionExecutor for each parameter combination
    - StandardDataCollector: Collects data for each combination
    - CompositeAnalyzer: Analyzes sensitivity results
    - Results writer: Handles sensitivity output (if provided)
    """
    
    def __init__(self, model: "Model", parameters: "Parameters", results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer
        self.logger = logging.getLogger(__name__)
        
    def execute(self) -> BatchResult:
        """
        Execute sensitivity analysis and return batch results.
        
        This orchestrates tests across parameter combinations and aggregates results.
        """
        start_time = datetime.now()
        
        if not self.parameters.sensitivity:
            raise ValueError("No sensitivity parameters defined")
            
        # Generate parameter combinations
        combinations = self._generate_parameter_combinations()
        self.logger.info(f"Starting sensitivity analysis: {len(combinations)} parameter combinations")
        
        # Initialize batch result
        batch_result = BatchResult()
        batch_result.batch_metadata.update({
            'batch_type': 'sensitivity_analysis',
            'parameter_combinations': len(combinations),
            'start_time': start_time.isoformat(),
            'sensitivity_config': self.parameters.sensitivity
        })
        
        try:
            # Execute batch processing
            scenario_results = self.process_batch()
            batch_result.scenario_results = {
                result.analysis_metadata.get('combination_id', f'combination_{i}'): result 
                for i, result in enumerate(scenario_results)
            }
            batch_result.successful_scenarios = list(batch_result.scenario_results.keys())
            
            # Update batch metadata
            end_time = datetime.now()
            batch_result.batch_metadata.update({
                'end_time': end_time.isoformat(),
                'execution_time': (end_time - start_time).total_seconds(),
                'total_scenarios': len(scenario_results)
            })
            
            self.logger.info(f"✓ Sensitivity analysis complete: {len(scenario_results)} combinations tested")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"✗ Sensitivity analysis failed: {e}")
            batch_result.batch_metadata['error'] = str(e)
            raise
    
    def process_batch(self) -> List[AnalysisResults]:
        """
        Process sensitivity analysis for multiple parameter combinations.
        
        This implements the BatchProcessor interface.
        """
        from disruptsc.model.model import Model
        
        combinations = self._generate_parameter_combinations()
        results = []
        
        for i, combination in enumerate(combinations):
            try:
                combination_id = f"combination_{i}"
                self.logger.info(f"=============== Starting combination #{i}: {combination} ===============")
                
                # Create fresh model with modified parameters
                modified_params = self._apply_parameter_combination(combination)
                model = Model(modified_params)
                
                # Reset model state
                if i == 0:
                    _reset_model_state(model, modified_params, full_reset=True)
                else:
                    _reset_model_state(model, modified_params)
                
                # Execute single sensitivity scenario
                scenario_result = self._execute_single_combination(model, modified_params, combination, combination_id)
                results.append(scenario_result)
                
                # Write results if writer provided
                if self.results_writer:
                    self._write_scenario_results(combination_id, scenario_result, model, combination)
                
                # Clean up memory
                del model
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Combination {i} failed: {e}")
                continue
        
        return results
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for sensitivity analysis."""
        sensitivity_config = self.parameters.sensitivity
        
        # Extract parameter names and values
        param_names = []
        param_values = []
        
        for param_name, values in sensitivity_config.items():
            if isinstance(values, list) and len(values) > 1:
                param_names.append(param_name)
                param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for value_combination in itertools.product(*param_values):
            combination = dict(zip(param_names, value_combination))
            combinations.append(combination)
        
        return combinations
    
    def _apply_parameter_combination(self, combination: Dict[str, Any]):
        """Apply parameter combination to create modified parameters."""
        modified_params = deepcopy(self.parameters)
        
        # Apply each parameter modification
        for param_name, value in combination.items():
            if hasattr(modified_params, param_name):
                setattr(modified_params, param_name, value)
            else:
                self.logger.warning(f"Parameter {param_name} not found in parameters object")
        
        return modified_params
    
    def _execute_single_combination(self, model, parameters, combination: Dict[str, Any], 
                                   combination_id: str) -> AnalysisResults:
        """Execute sensitivity test for a single parameter combination."""
        # Use disruption executor for the sensitivity test
        executor = DisruptionExecutor(model, parameters)
        
        # Execute the scenario
        analysis_results = executor.execute()
        
        # Add combination-specific metadata
        analysis_results.analysis_metadata.update({
            'combination_id': combination_id,
            'parameter_combination': combination,
            'simulation_type': 'sensitivity_combination'
        })
        
        return analysis_results
    
    def _write_scenario_results(self, combination_id: str, results: AnalysisResults, 
                               model, combination: Dict[str, Any]):
        """Write results using the provided results writer."""
        if self.results_writer and hasattr(self.results_writer, 'write_sensitivity_results'):
            # Create a mock simulation object with the data needed by the writer
            class MockSimulation:
                def __init__(self, results: AnalysisResults):
                    self.analysis_results = results
                
                def calculate_household_loss(self, household_table=None, **kwargs):
                    return results.get_metric('household', 'losses', {}).get('absolute_cumulated', 0)
                    
                def calculate_country_loss(self, **kwargs):
                    return results.get_metric('country', 'losses', {}).get('absolute_cumulated', 0)
            
            mock_simulation = MockSimulation(results)
            self.results_writer.write_sensitivity_results(
                combination_id, mock_simulation, model, combination
            )
    
    def create_runner(self):
        """Not used in batch processing - scenarios create their own executors."""
        pass
        
    def create_collector(self):
        """Not used in batch processing - scenarios create their own collectors."""
        return StandardDataCollector(self.parameters)