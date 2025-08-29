"""
Criticality Executor for the new clean architecture.

This executor handles criticality analysis simulations using the new batch processing
architecture with distinct components for execution, data collection, analysis, and export.
"""

from typing import TYPE_CHECKING, List, Dict, Any
import logging
import gc
from datetime import datetime

from .base_executor import BaseExecutor
from ..core.interfaces import BatchProcessor
from ..core.data_structures import BatchResult, AnalysisResults
from ..collection.base_collector import StandardDataCollector

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


class CriticalityExecutor(BaseExecutor, BatchProcessor):
    """
    Executes criticality analysis simulations using the new architecture.
    
    This replaces the old CriticalityExecutor with batch processing using:
    - BatchProcessor interface for handling multiple edge criticality tests
    - Custom runner for each edge disruption
    - StandardDataCollector: Collects data for each edge test
    - CompositeAnalyzer: Analyzes criticality impact
    - Results writer: Handles criticality output (if provided)
    """
    
    def __init__(self, model: "Model", parameters: "Parameters", results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer
        self.logger = logging.getLogger(__name__)
        
    def execute(self) -> BatchResult:
        """
        Execute criticality analysis and return batch results.
        
        This orchestrates criticality tests across transport edges and aggregates results.
        """
        start_time = datetime.now()
        self.logger.info("Starting criticality analysis")
        
        # Initialize batch result
        batch_result = BatchResult()
        batch_result.batch_metadata.update({
            'batch_type': 'criticality_analysis',
            'start_time': start_time.isoformat()
        })
        
        try:
            # Execute batch processing
            scenario_results = self.process_batch()
            batch_result.scenario_results = {
                result.analysis_metadata.get('edge_id', f'edge_{i}'): result 
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
            
            self.logger.info(f"✓ Criticality analysis complete: {len(scenario_results)} edges tested")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"✗ Criticality analysis failed: {e}")
            batch_result.batch_metadata['error'] = str(e)
            raise
    
    def process_batch(self) -> List[AnalysisResults]:
        """
        Process criticality analysis for multiple edges.
        
        This implements the BatchProcessor interface.
        """
        from disruptsc.model.utils.caching import load_cached_model
        
        # Save model state for reloading
        suffix = round(datetime.now().timestamp() * 1000)
        self.model.save_pickle(suffix)
        
        # Determine edges to test
        edges_to_test = self._get_edges_to_test()
        disruption_duration = self.parameters.criticality['duration']
        
        self.logger.info(f"========== Criticality simulation of {len(edges_to_test)} edges ==========")
        
        results = []
        
        for edge, attribute in edges_to_test.items():
            try:
                self.logger.info(f"=== Edge {edge} ====")
                
                # Load fresh model state
                model = load_cached_model(suffix)
                
                # Execute single criticality scenario
                scenario_result = self._execute_single_edge_test(model, edge, disruption_duration, attribute)
                results.append(scenario_result)
                
                # Write results if writer provided
                if self.results_writer:
                    self._write_scenario_results(edge, scenario_result, model, attribute)
                
                # Clean up memory
                del model
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Edge test failed for edge {edge}: {e}")
                continue
        
        return results
    
    def _get_edges_to_test(self) -> Dict[Any, Dict[str, Any]]:
        """Get edges to test for criticality analysis."""
        # This implements the logic from the original criticality executor
        import geopandas as gpd
        from disruptsc.model.utils.functions import filter_transport_network
        
        # Get criticality configuration
        criticality_config = self.parameters.criticality
        
        # Filter transport network based on configuration
        if 'filter' in criticality_config and criticality_config['filter']:
            filtered_edges = filter_transport_network(
                self.model.transport_network, 
                criticality_config['filter']
            )
        else:
            filtered_edges = self.model.transport_network
        
        # Convert to dictionary format
        edges_to_test = {}
        for idx, row in filtered_edges.iterrows():
            edge_id = row.get('id', idx)
            edges_to_test[edge_id] = row.to_dict()
        
        return edges_to_test
    
    def _execute_single_edge_test(self, model, edge_id: str, duration: int, 
                                 edge_attributes: Dict[str, Any]) -> AnalysisResults:
        """Execute criticality test for a single edge."""
        from ...runners.base_runner import BaseSimulationRunner
        from ...core.data_structures import SimulationData
        
        class CriticalityRunner(BaseSimulationRunner):
            def __init__(self, model, parameters, edge_id, duration):
                super().__init__(model, parameters)
                self.edge_id = edge_id
                self.duration = duration
            
            def _execute_simulation_logic(self) -> SimulationData:
                # Execute criticality disruption simulation
                simulation_result = self.model.run_criticality_disruption(self.edge_id, self.duration)
                
                # Collect data from the simulation result
                collector = StandardDataCollector(self.parameters)
                
                # Simulate data collection over time steps
                for t in range(self.duration + 1):
                    collector.collect_time_step(self.model, t)
                
                simulation_data = collector.get_simulation_data()
                simulation_data.metadata['edge_id'] = self.edge_id
                simulation_data.metadata['disruption_duration'] = self.duration
                simulation_data.metadata['simulation_type'] = 'criticality'
                
                return simulation_data
        
        # Create custom executor for this edge test
        executor = BaseExecutor(model, self.parameters)
        executor.runner = CriticalityRunner(model, self.parameters, edge_id, duration)
        executor.collector = StandardDataCollector(self.parameters)
        
        # Execute the scenario
        analysis_results = executor.execute()
        
        # Add edge-specific metadata
        analysis_results.analysis_metadata.update({
            'edge_id': edge_id,
            'disruption_duration': duration,
            'edge_attributes': edge_attributes,
            'simulation_type': 'criticality_edge_test'
        })
        
        return analysis_results
    
    def _write_scenario_results(self, edge_id: str, results: AnalysisResults, 
                               model, edge_attributes: Dict[str, Any]):
        """Write results using the provided results writer."""
        if self.results_writer and hasattr(self.results_writer, 'write_criticality_results'):
            # Create a mock simulation object with the data needed by the writer
            class MockSimulation:
                def __init__(self, results: AnalysisResults):
                    self.analysis_results = results
                
                def calculate_household_loss(self, household_table=None, per_region=False, **kwargs):
                    household_losses = results.get_metric('household', 'losses', {})
                    if per_region:
                        return household_losses.get('per_region', {})
                    else:
                        return household_losses.get('absolute_cumulated', 0)
                    
                def calculate_country_loss(self, per_country=False, **kwargs):
                    country_losses = results.get_metric('country', 'losses', {})
                    if per_country:
                        return country_losses.get('per_country', {})
                    else:
                        return country_losses.get('absolute_cumulated', 0)
            
            mock_simulation = MockSimulation(results)
            self.results_writer.write_criticality_results(
                edge_id, mock_simulation, model, edge_attributes
            )
    
    def create_runner(self):
        """Not used in batch processing - scenarios create their own runners."""
        pass
        
    def create_collector(self):
        """Not used in batch processing - scenarios create their own collectors."""
        return StandardDataCollector(self.parameters)