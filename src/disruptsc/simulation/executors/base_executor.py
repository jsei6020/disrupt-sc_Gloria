"""
Base executor for the new clean architecture.

This module provides the base class that coordinates between:
- SimulationRunner (executes the simulation)
- DataCollector (gathers data during execution)  
- Analyzer (calculates metrics from data)
- Exporter (outputs formatted results)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any
import logging
from datetime import datetime

from ..core.interfaces import SimulationRunner, DataCollector, Analyzer, Exporter
from ..core.data_structures import SimulationData, AnalysisResults, ExportConfiguration
from ..analysis.composite_analyzer import CompositeAnalyzer

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


class BaseExecutor(ABC):
    """
    Base class for the new simulation executor architecture.
    
    This class coordinates the four main components:
    1. SimulationRunner - Executes the actual simulation
    2. DataCollector - Collects data during simulation  
    3. Analyzer - Processes data into metrics
    4. Exporter - Outputs results to files
    
    Unlike the old executors that directly called model methods,
    this architecture provides clean separation of concerns.
    """
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        self.model = model
        self.parameters = parameters
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.runner: Optional[SimulationRunner] = None
        self.collector: Optional[DataCollector] = None
        self.analyzer: Optional[Analyzer] = None
        self.exporter: Optional[Exporter] = None
        
        # Initialize with default analyzer
        self.analyzer = CompositeAnalyzer(parameters)
        
    @abstractmethod
    def create_runner(self) -> SimulationRunner:
        """Create the appropriate simulation runner for this executor type."""
        pass
        
    @abstractmethod
    def create_collector(self) -> DataCollector:
        """Create the appropriate data collector for this executor type."""
        pass
    
    def create_analyzer(self) -> Analyzer:
        """Create the analyzer. Default is CompositeAnalyzer."""
        return CompositeAnalyzer(self.parameters)
        
    def create_exporter(self) -> Optional[Exporter]:
        """Create the exporter. Override if custom export logic needed."""
        return None  # Default to no export, let caller handle it
        
    def execute(self) -> AnalysisResults:
        """
        Execute the complete simulation pipeline.
        
        This is the main entry point that coordinates all components:
        1. Create components if not already created
        2. Run simulation and collect data
        3. Analyze collected data  
        4. Export results (if exporter configured)
        
        Returns:
            AnalysisResults: Analyzed simulation results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting {self.__class__.__name__} execution")
        
        try:
            # Initialize components if needed
            if self.runner is None:
                self.runner = self.create_runner()
            if self.collector is None:
                self.collector = self.create_collector()  
            if self.analyzer is None:
                self.analyzer = self.create_analyzer()
                
            # Run simulation and collect data
            self.logger.info("Running simulation...")
            simulation_data = self.runner.run()
            
            # Analyze collected data
            self.logger.info("Analyzing results...")
            analysis_results = self.analyzer.analyze(simulation_data)
            
            # Add execution metadata
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            analysis_results.analysis_metadata.update({
                'executor_class': self.__class__.__name__,
                'execution_start_time': start_time.isoformat(),
                'execution_end_time': end_time.isoformat(), 
                'execution_time_seconds': execution_time,
                'simulation_type': self.parameters.simulation_type
            })
            
            # Export if exporter configured
            if self.exporter is not None:
                self.logger.info("Exporting results...")
                export_config = self._create_export_config()
                self.exporter.export(analysis_results, export_config.output_path)
            
            self.logger.info(f"✓ Execution complete in {execution_time:.2f}s")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"✗ Execution failed: {e}")
            raise
            
    def execute_with_custom_components(
        self, 
        runner: Optional[SimulationRunner] = None,
        collector: Optional[DataCollector] = None,
        analyzer: Optional[Analyzer] = None,
        exporter: Optional[Exporter] = None
    ) -> AnalysisResults:
        """
        Execute with custom components.
        
        Allows overriding default components for specialized use cases.
        """
        # Temporarily set custom components
        original_runner = self.runner
        original_collector = self.collector
        original_analyzer = self.analyzer
        original_exporter = self.exporter
        
        try:
            if runner is not None:
                self.runner = runner
            if collector is not None:
                self.collector = collector
            if analyzer is not None:
                self.analyzer = analyzer
            if exporter is not None:
                self.exporter = exporter
                
            return self.execute()
            
        finally:
            # Restore original components
            self.runner = original_runner
            self.collector = original_collector 
            self.analyzer = original_analyzer
            self.exporter = original_exporter
            
    def _create_export_config(self) -> ExportConfiguration:
        """Create export configuration from parameters."""
        # This would be expanded to read from parameters
        # For now, provide basic configuration
        from disruptsc.paths import OUTPUT_FOLDER
        
        output_path = OUTPUT_FOLDER / self.parameters.scope / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ExportConfiguration(
            output_path=output_path,
            export_formats=['csv', 'json'],
            use_timestamp_folders=True
        )
        
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation configuration."""
        return {
            'executor_type': self.__class__.__name__,
            'simulation_type': getattr(self.parameters, 'simulation_type', 'unknown'),
            'scope': getattr(self.parameters, 'scope', 'unknown'),
            'model_agent_counts': {
                'firms': len(self.model.firms) if hasattr(self.model, 'firms') else 0,
                'households': len(self.model.households) if hasattr(self.model, 'households') else 0,
                'countries': len(self.model.countries) if hasattr(self.model, 'countries') else 0
            },
            'runner_class': self.runner.__class__.__name__ if self.runner else 'Not initialized',
            'collector_class': self.collector.__class__.__name__ if self.collector else 'Not initialized',
            'analyzer_class': self.analyzer.__class__.__name__ if self.analyzer else 'Not initialized'
        }