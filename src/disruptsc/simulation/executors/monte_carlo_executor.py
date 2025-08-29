"""
Monte Carlo Executor for the new clean architecture.

This executor handles Monte Carlo simulations using the new batch processing
architecture with distinct components for execution, data collection, analysis, and export.
"""

from typing import TYPE_CHECKING, Type, List
import pandas as pd
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


class MonteCarloExecutor(BaseExecutor, BatchProcessor):
    """
    Executes Monte Carlo simulations using the new architecture.
    
    This replaces the old MonteCarloExecutor with batch processing using:
    - BatchProcessor interface for handling multiple iterations
    - Base executor class for each iteration  
    - StandardDataCollector: Collects data across iterations
    - CompositeAnalyzer: Analyzes aggregated results
    - Results writer: Handles iteration output (if provided)
    """
    
    def __init__(self, model: "Model", parameters: "Parameters", 
                 base_executor_class: Type[BaseExecutor], results_writer=None):
        super().__init__(model, parameters)
        self.base_executor_class = base_executor_class
        self.results_writer = results_writer
        self.logger = logging.getLogger(__name__)
        
    def execute(self) -> BatchResult:
        """
        Execute Monte Carlo iterations and return batch results.
        
        This orchestrates multiple simulation iterations and aggregates results.
        """
        start_time = datetime.now()
        self.logger.info(f"Starting Monte Carlo with {self.parameters.mc_repetitions} iterations")
        
        # Initialize batch result
        batch_result = BatchResult()
        batch_result.batch_metadata.update({
            'batch_type': 'monte_carlo',
            'mc_repetitions': self.parameters.mc_repetitions,
            'base_executor': self.base_executor_class.__name__,
            'start_time': start_time.isoformat()
        })
        
        try:
            # Execute batch processing
            scenario_results = self.process_batch()
            batch_result.scenario_results = {
                f'iteration_{i}': result for i, result in enumerate(scenario_results)
            }
            batch_result.successful_scenarios = list(batch_result.scenario_results.keys())
            
            # Update batch metadata
            end_time = datetime.now()
            batch_result.batch_metadata.update({
                'end_time': end_time.isoformat(),
                'execution_time': (end_time - start_time).total_seconds(),
                'total_scenarios': len(scenario_results)
            })
            
            self.logger.info(f"✓ Monte Carlo complete: {len(scenario_results)} iterations")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"✗ Monte Carlo failed: {e}")
            batch_result.batch_metadata['error'] = str(e)
            raise
    
    def process_batch(self) -> List[AnalysisResults]:
        """
        Process multiple Monte Carlo iterations.
        
        This implements the BatchProcessor interface.
        """
        results = []
        
        for i in range(self.parameters.mc_repetitions):
            try:
                self.logger.info(f"=============== Starting repetition #{i} ===============")
                
                # Reset model state for each iteration
                if i == 0:
                    self._reset_model_state(full_reset=True)
                else:
                    self._reset_model_state()
                
                # Execute base simulation using the base executor class
                executor = self.base_executor_class(self.model, self.parameters)
                iteration_result = executor.execute()
                
                # Add iteration-specific metadata
                iteration_result.analysis_metadata.update({
                    'iteration_number': i,
                    'batch_type': 'monte_carlo',
                    'simulation_type': f'mc_{self.parameters.simulation_type}'
                })
                
                results.append(iteration_result)
                
                # Write iteration results if writer provided
                if self.results_writer:
                    self._write_iteration_results(i, iteration_result)
                
                # Clean up memory
                del executor
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Iteration {i} failed: {e}")
                continue
        
        return results
    
    def _reset_model_state(self, full_reset: bool = False):
        """Reset model state for each Monte Carlo iteration using configured caching."""
        if full_reset:
            self.model.setup_transport_network(False, self.parameters.with_transport)
            self.model.setup_agents(False)
            self.model.setup_sc_network(False)
            self.model.set_initial_conditions()
            self.model.setup_logistic_routes(False)
        else:
            caching_config = getattr(self.parameters, 'mc_caching', {
                'transport_network': True,
                'agents': False,
                'sc_network': False,
                'logistic_routes': False
            })
            self.model.setup_transport_network(caching_config['transport_network'], self.parameters.with_transport)
            self.model.setup_agents(caching_config['agents'])
            self.model.setup_sc_network(caching_config['sc_network'])
            self.model.set_initial_conditions()
            self.model.setup_logistic_routes(caching_config['logistic_routes'])
    
    def _write_iteration_results(self, iteration: int, results: AnalysisResults):
        """Write results using the provided results writer."""
        if hasattr(self.results_writer, 'write_iteration_results'):
            # Create a mock simulation object with the data needed by the writer
            class MockSimulation:
                def __init__(self, results: AnalysisResults):
                    self.analysis_results = results
            
            mock_simulation = MockSimulation(results)
            self.results_writer.write_iteration_results(iteration, mock_simulation, self.model)
    
    def create_runner(self):
        """Not used in batch processing - iterations create their own executors."""
        pass
        
    def create_collector(self):
        """Not used in batch processing - iterations create their own collectors."""
        return StandardDataCollector(self.parameters)


class InitialStateMCExecutor(BaseExecutor, BatchProcessor):
    """
    Specialized Monte Carlo executor for initial state simulations with flow aggregation.
    
    This replaces the old InitialStateMCExecutor with the new architecture.
    """
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        super().__init__(model, parameters)
        self.logger = logging.getLogger(__name__)
        
    def execute(self) -> BatchResult:
        """Execute initial state Monte Carlo with flow aggregation."""
        start_time = datetime.now()
        self.logger.info(f"Starting Initial State Monte Carlo with {self.parameters.mc_repetitions} iterations")
        
        # Initialize batch result
        batch_result = BatchResult()
        batch_result.batch_metadata.update({
            'batch_type': 'initial_state_monte_carlo',
            'mc_repetitions': self.parameters.mc_repetitions,
            'start_time': start_time.isoformat()
        })
        
        try:
            # Execute batch processing with flow aggregation
            scenario_results = self.process_batch()
            batch_result.scenario_results = {
                f'iteration_{i}': result for i, result in enumerate(scenario_results)
            }
            batch_result.successful_scenarios = list(batch_result.scenario_results.keys())
            
            # Update batch metadata
            end_time = datetime.now()
            batch_result.batch_metadata.update({
                'end_time': end_time.isoformat(),
                'execution_time': (end_time - start_time).total_seconds(),
                'total_scenarios': len(scenario_results)
            })
            
            self.logger.info(f"✓ Initial State Monte Carlo complete: {len(scenario_results)} iterations")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"✗ Initial State Monte Carlo failed: {e}")
            batch_result.batch_metadata['error'] = str(e)
            raise
    
    def process_batch(self) -> List[AnalysisResults]:
        """Process multiple initial state iterations with flow aggregation."""
        import pandas as pd
        from disruptsc.model.utils.caching import load_cached_model
        
        flow_dfs = {}
        results = []
        
        # Setup and save initial model state
        self.model.setup_transport_network(cached=False)
        self.model.setup_agents(cached=False)
        self.model.save_pickle('initial_state_mc')
        
        for i in range(self.parameters.mc_repetitions):
            try:
                self.logger.info(f"=============== Starting repetition #{i} ===============")
                
                # Load cached model and reset stochastic components
                model = load_cached_model("initial_state_mc")
                model.shuffle_logistic_costs()
                model.setup_sc_network(cached=False)
                model.set_initial_conditions()
                model.setup_logistic_routes(cached=False)
                
                # Execute initial state simulation
                from .initial_state_executor import InitialStateExecutor
                executor = InitialStateExecutor(model, self.parameters)
                iteration_result = executor.execute()
                
                # Extract and save flow data
                flow_df = self._extract_flow_data(iteration_result)
                flow_df.to_csv(self.parameters.export_folder / f"flow_df_{i}.csv")
                
                # Store minimal data for aggregation
                flow_dfs[i] = flow_df[['id', 'flow_total']].copy() if 'flow_total' in flow_df.columns else flow_df
                
                # Add iteration metadata
                iteration_result.analysis_metadata.update({
                    'iteration_number': i,
                    'batch_type': 'initial_state_monte_carlo'
                })
                
                results.append(iteration_result)
                
                # Clean up memory
                del model
                del executor
                del flow_df
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Iteration {i} failed: {e}")
                continue
        
        # Aggregate flow results
        self._aggregate_flow_results(flow_dfs)
        
        # Clean up
        del flow_dfs
        gc.collect()
        
        return results
    
    def _extract_flow_data(self, results: AnalysisResults) -> pd.DataFrame:
        """Extract flow data from analysis results."""
        import pandas as pd
        
        # Try to get flow data from transport network data in results
        if 'transport_flows' in results.export_tables:
            flow_df = results.export_tables['transport_flows']
        elif results.flow_analysis and 'transport_flows' in results.flow_analysis:
            flow_df = pd.DataFrame(results.flow_analysis['transport_flows'])
        else:
            # Create empty DataFrame if no flow data found
            flow_df = pd.DataFrame({'id': [], 'flow_total': []})
        
        # Filter for non-zero flows at time step 0
        if 'flow_total' in flow_df.columns and 'time_step' in flow_df.columns:
            flow_df = flow_df[(flow_df['flow_total'] > 0) & (flow_df['time_step'] == 0)]
        
        return flow_df
    
    def _aggregate_flow_results(self, flow_dfs):
        """Aggregate flow results across Monte Carlo iterations."""
        import pandas as pd
        
        if not flow_dfs:
            self.logger.warning("No flow data to aggregate")
            return
        
        try:
            mean_flows = pd.concat(flow_dfs.values())
            mean_flows = mean_flows.groupby(mean_flows.index).mean()
            
            # Merge with transport edges
            transport_edges_with_flows = pd.merge(
                self.model.transport_edges.drop(columns=["node_tuple"], errors='ignore'),
                mean_flows, how="left", on="id"
            )
            
            transport_edges_with_flows.to_file(
                self.parameters.export_folder / f"transport_edges_with_flows.geojson",
                driver="GeoJSON", index=False
            )
            
            self.logger.info("✓ Flow aggregation completed")
            
        except Exception as e:
            self.logger.error(f"Flow aggregation failed: {e}")
    
    def create_runner(self):
        """Not used in batch processing."""
        pass
        
    def create_collector(self):
        """Not used in batch processing."""
        return StandardDataCollector(self.parameters)