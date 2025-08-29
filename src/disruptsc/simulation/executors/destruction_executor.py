"""
Destruction Executor for the new clean architecture.

This executor handles destruction analysis across multiple targets (sectors, provinces, cantons)
using the new batch processing architecture with distinct components for execution,
data collection, analysis, and export.
"""

from typing import TYPE_CHECKING, List, Dict, Any
import logging
import gc
from datetime import datetime

from .base_executor import BaseExecutor
from ..core.interfaces import BatchProcessor
from ..core.data_structures import BatchResult, AnalysisResults
from ..runners.destruction_runner import DestructionRunner
from ..collection.base_collector import StandardDataCollector

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


def _get_disrupted_sector_list() -> List:
    """Get the list of sector combinations to test."""
    return ['all', ['ADM'], ['ADP'], ['ALD'], ['ASO'], ['AYG'], ['BAL'], ['CAR'], ['CIN'], ['COM'], ['CON'], ['DEM'], ['EDU'], ['ELE'], ['FIN'], ['FRT'], ['FRV'], ['GAN'], ['INM'], ['LAC'], ['MAQ'], ['MIP'], ['MOL'], ['MUE'], ['PAN'], ['PES'], ['PPR'], ['QU2'], ['REF'], ['RES'], ['SAL'], ['SEG'], ['TEL'], ['TRA'], ['AGU', 'BNA'], ['AGU', 'CHO'], ['AGU', 'HIL'], ['AGU', 'MET'], ['AGU', 'SIL'], ['AGU', 'VES'], ['AZU', 'BNA'], ['AZU', 'CHO'], ['AZU', 'DOM'], ['AZU', 'HIL'], ['AZU', 'MAD'], ['AZU', 'MET'], ['AZU', 'SIL'], ['AZU', 'VES'], ['BNA', 'CAN'], ['BNA', 'CAU'], ['BNA', 'CER'], ['BNA', 'CHO'], ['BNA', 'CUE'], ['BNA', 'DOM'], ['BNA', 'FID'], ['BNA', 'HIL'], ['BNA', 'HOT'], ['BNA', 'MAD'], ['BNA', 'MAN'], ['BNA', 'MET'], ['BNA', 'PAP'], ['BNA', 'PLS'], ['BNA', 'POS'], ['BNA', 'REP'], ['BNA', 'SIL'], ['BNA', 'TAB'], ['BNA', 'VES'], ['CAN', 'CHO'], ['CAN', 'MET'], ['CAN', 'SIL'], ['CAN', 'VES'], ['CAU', 'CHO'], ['CAU', 'VES'], ['CER', 'CHO'], ['CER', 'DOM'], ['CER', 'HIL'], ['CER', 'MAD'], ['CER', 'MET'], ['CER', 'REP'], ['CER', 'SIL'], ['CER', 'VES'], ['CHO', 'CUE'], ['CHO', 'DOM'], ['CHO', 'FID'], ['CHO', 'HIL'], ['CHO', 'HOT'], ['CHO', 'MAD'], ['CHO', 'MAN'], ['CHO', 'MET'], ['CHO', 'PAP'], ['CHO', 'PLS'], ['CHO', 'POS'], ['CHO', 'REP'], ['CHO', 'SIL'], ['CHO', 'VES'], ['CUE', 'HIL'], ['CUE', 'MAD'], ['CUE', 'MET'], ['CUE', 'SIL'], ['CUE', 'VES'], ['DOM', 'HIL'], ['DOM', 'HOT'], ['DOM', 'MAD'], ['DOM', 'MET'], ['DOM', 'PAP'], ['DOM', 'REP'], ['DOM', 'SIL'], ['DOM', 'VES'], ['FID', 'VES'], ['HIL', 'HOT'], ['HIL', 'MAD'], ['HIL', 'MET'], ['HIL', 'PAP'], ['HIL', 'PLS'], ['HIL', 'REP'], ['HIL', 'SIL'], ['HIL', 'VES'], ['HOT', 'MAD'], ['HOT', 'MET'], ['HOT', 'SIL'], ['HOT', 'VES'], ['MAD', 'MET'], ['MAD', 'PAP'], ['MAD', 'REP'], ['MAD', 'SIL'], ['MAD', 'VES'], ['MAN', 'VES'], ['MET', 'PAP'], ['MET', 'PLS'], ['MET', 'REP'], ['MET', 'SIL'], ['MET', 'VES'], ['PAP', 'REP'], ['PAP', 'SIL'], ['PAP', 'VES'], ['PLS', 'SIL'], ['PLS', 'VES'], ['POS', 'SIL'], ['POS', 'VES'], ['REP', 'SIL'], ['REP', 'VES'], ['SIL', 'VES'], ['TAB', 'VES']]


def _get_disrupted_subregion_list(which_subregion: str) -> List:
    """Get the list of subregion combinations to test."""
    if which_subregion == "province":
        return [['AZUAY'], ['CAÑAR'], ['CHIMBORAZO'], ['COTOPAXI'], ['EL ORO'], ['ESMERALDAS'], ['GUAYAS'], ['LOJA'], ['LOS RIOS'], ['MANABI'], ['PICHINCHA'], ['TUNGURAHUA'], ['CARCHI', 'IMBABURA'], ['IMBABURA', 'NAPO'], ['IMBABURA', 'SUCUMBIOS'], ['ORELLANA', 'SUCUMBIOS']]
    if which_subregion == "canton":
        return [['AZUAY - CUENCA'], ['CAÑAR - AZOGUES'], ['COTOPAXI - LATACUNGA'], ['EL ORO - MACHALA'], ['ESMERALDAS - RIO VERDE'], ['GUAYAS - DURAN'], ['GUAYAS - GUAYAQUIL'], ['GUAYAS - SAMBORONDON'], ['LOJA - LOJA'], ['LOS RIOS - QUEVEDO'], ['MANABI - JARAMIJO'], ['MANABI - MANTA'], ['MANABI - PORTOVIEJO'], ['PICHINCHA - QUITO'], ['PICHINCHA - SANTO DOMINGO'], ['TUNGURAHUA - AMBATO'], ['BOLIVAR - GUARANDA', 'CHIMBORAZO - RIOBAMBA'], ['CHIMBORAZO - GUANO', 'CHIMBORAZO - PENIPE'], ['CHIMBORAZO - GUANO', 'CHIMBORAZO - RIOBAMBA'], ['CHIMBORAZO - GUANO', 'TUNGURAHUA - QUERO'], ['CHIMBORAZO - GUANO', 'TUNGURAHUA - SAN PEDRO DE PELILEO'], ['EL ORO - PASAJE', 'EL ORO - SANTA ROSA'], ['ESMERALDAS - ESMERALDAS', 'ESMERALDAS - QUININDE'], ['GUAYAS - EL TRIUNFO', 'GUAYAS - SAN JACINTO DE YAGUACHI'], ['GUAYAS - MILAGRO', 'GUAYAS - SAN JACINTO DE YAGUACHI'], ['IMBABURA - IBARRA', 'PICHINCHA - CAYAMBE'], ['PICHINCHA - MEJIA', 'PICHINCHA - RUMIÑAHUI'], ['GUAYAS - DAULE', 'GUAYAS - PEDRO CARBO', 'GUAYAS - SANTA LUCIA'], ['GUAYAS - DAULE', 'GUAYAS - SALITRE', 'LOS RIOS - BABAHOYO'], ['ESMERALDAS - LA CONCORDIA', 'MANABI - CHONE', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - EL CARMEN', 'MANABI - ROCAFUERTE', 'MANABI - SUCRE'], ['MANABI - CHONE', 'MANABI - EL CARMEN', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - JUNIN', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - PICHINCHA', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - JIPIJAPA', 'MANABI - MONTECRISTI', 'MANABI - PICHINCHA', 'MANABI - SANTA ANA']]


def _get_disrupted_subregion_sector_list(which_subregion: str) -> List:
    """Get strategic list of (subregion, sector) combinations to test."""
    if which_subregion == "province":
        return [[('AZUAY', 'ELE')], [('AZUAY', 'ADP'), ('AZUAY', 'CAR')], [('AZUAY', 'AYG'), ('AZUAY', 'COM')], [('AZUAY', 'CON'), ('AZUAY', 'EDU')], [('AZUAY', 'FIN'), ('AZUAY', 'INM'), ('AZUAY', 'SAL')], [('CAÑAR', 'PPR')], [('CHIMBORAZO', 'CON'), ('CHIMBORAZO', 'MIP')], [('COTOPAXI', 'CON'), ('COTOPAXI', 'EDU'), ('COTOPAXI', 'TRA')], [('EL ORO', 'CON'), ('EL ORO', 'FRV')], [('EL ORO', 'ADP'), ('EL ORO', 'ALD'), ('EL ORO', 'COM'), ('EL ORO', 'FRT')], [('ESMERALDAS', 'MIP')], [('GUAYAS', 'ADP')], [('GUAYAS', 'COM')], [('GUAYAS', 'CON')], [('GUAYAS', 'EDU')], [('GUAYAS', 'INM')], [('GUAYAS', 'PES')], [('GUAYAS', 'REF')], [('GUAYAS', 'RES')], [('GUAYAS', 'SAL')], [('GUAYAS', 'TEL')], [('GUAYAS', 'TRA')], [('GUAYAS', 'ADM'), ('GUAYAS', 'ASO')], [('GUAYAS', 'AGU'), ('GUAYAS', 'BAL')], [('GUAYAS', 'ALD'), ('GUAYAS', 'ELE')], [('GUAYAS', 'BNA'), ('GUAYAS', 'FIN')], [('GUAYAS', 'CAN'), ('GUAYAS', 'MOL')], [('GUAYAS', 'CAR'), ('GUAYAS', 'FRT')], [('GUAYAS', 'CHO'), ('GUAYAS', 'LAC')], [('GUAYAS', 'FRV'), ('GUAYAS', 'MAQ')], [('GUAYAS', 'MET'), ('GUAYAS', 'PPR')], [('GUAYAS', 'PAN'), ('GUAYAS', 'SEG')], [('GUAYAS', 'CIN'), ('GUAYAS', 'DEM'), ('GUAYAS', 'MIP')], [('GUAYAS', 'GAN'), ('GUAYAS', 'PAP'), ('GUAYAS', 'QU2')], [('LOJA', 'CON'), ('LOJA', 'MIP')], [('LOS RIOS', 'FRT')], [('LOS RIOS', 'ASO'), ('LOS RIOS', 'CIN'), ('LOS RIOS', 'CON')], [('MANABI', 'CON')], [('MANABI', 'PPR')], [('MANABI', 'AYG'), ('MANABI', 'FRV')], [('MANABI', 'EDU'), ('MANABI', 'TRA')], [('PICHINCHA', 'ADP')], [('PICHINCHA', 'CAR')], [('PICHINCHA', 'COM')], [('PICHINCHA', 'CON')], [('PICHINCHA', 'EDU')], [('PICHINCHA', 'GAN')], [('PICHINCHA', 'INM')], [('PICHINCHA', 'MIP')], [('PICHINCHA', 'RES')], [('PICHINCHA', 'SAL')], [('PICHINCHA', 'TEL')], [('PICHINCHA', 'TRA')], [('PICHINCHA', 'ADM'), ('PICHINCHA', 'ALD')], [('PICHINCHA', 'ASO'), ('PICHINCHA', 'BNA')], [('PICHINCHA', 'AYG'), ('PICHINCHA', 'FIN')], [('PICHINCHA', 'CHO'), ('PICHINCHA', 'MAQ')], [('PICHINCHA', 'BAL'), ('PICHINCHA', 'CIN'), ('PICHINCHA', 'PAN')], [('PICHINCHA', 'DEM'), ('PICHINCHA', 'ELE'), ('PICHINCHA', 'QU2')], [('PICHINCHA', 'HIL'), ('PICHINCHA', 'LAC'), ('PICHINCHA', 'SEG')], [('PICHINCHA', 'DOM'), ('PICHINCHA', 'MAD'), ('PICHINCHA', 'MUE'), ('PICHINCHA', 'REF')], [('TUNGURAHUA', 'ADP'), ('TUNGURAHUA', 'CON'), ('TUNGURAHUA', 'FRT')]]
    
    elif which_subregion == "canton":
        return [[('AZUAY - CUENCA', 'ELE')], [('AZUAY - CUENCA', 'ADP'), ('AZUAY - CUENCA', 'CAR')], [('AZUAY - CUENCA', 'AYG'), ('AZUAY - CUENCA', 'COM')], [('AZUAY - CUENCA', 'CON'), ('AZUAY - CUENCA', 'EDU')], [('AZUAY - CUENCA', 'FIN'), ('AZUAY - CUENCA', 'INM'), ('AZUAY - CUENCA', 'SAL')], [('CAÑAR - AZOGUES', 'PPR')], [('GUAYAS - GUAYAQUIL', 'ADP')], [('GUAYAS - GUAYAQUIL', 'COM')], [('GUAYAS - GUAYAQUIL', 'CON')], [('GUAYAS - GUAYAQUIL', 'EDU')], [('GUAYAS - GUAYAQUIL', 'INM')], [('GUAYAS - GUAYAQUIL', 'PES')], [('GUAYAS - GUAYAQUIL', 'REF')], [('GUAYAS - GUAYAQUIL', 'SAL')], [('GUAYAS - GUAYAQUIL', 'TEL')], [('GUAYAS - GUAYAQUIL', 'TRA')], [('GUAYAS - GUAYAQUIL', 'ADM'), ('GUAYAS - GUAYAQUIL', 'ASO')], [('GUAYAS - GUAYAQUIL', 'AGU'), ('GUAYAS - GUAYAQUIL', 'ELE')], [('GUAYAS - GUAYAQUIL', 'BAL'), ('GUAYAS - GUAYAQUIL', 'BNA')], [('GUAYAS - GUAYAQUIL', 'CAR'), ('GUAYAS - GUAYAQUIL', 'FIN')], [('GUAYAS - GUAYAQUIL', 'CHO'), ('GUAYAS - GUAYAQUIL', 'LAC')], [('GUAYAS - GUAYAQUIL', 'FRT'), ('GUAYAS - GUAYAQUIL', 'MAQ')], [('GUAYAS - GUAYAQUIL', 'HIL'), ('GUAYAS - GUAYAQUIL', 'RES')], [('GUAYAS - GUAYAQUIL', 'MOL'), ('GUAYAS - GUAYAQUIL', 'PAN')], [('GUAYAS - GUAYAQUIL', 'ALD'), ('GUAYAS - GUAYAQUIL', 'MIP'), ('GUAYAS - GUAYAQUIL', 'SEG')], [('LOJA - LOJA', 'CON'), ('LOJA - LOJA', 'MIP')], [('AZUAY - CUENCA', 'MUE'), ('GUAYAS - NARANJAL', 'FRV'), ('GUAYAS - SAN JACINTO DE YAGUACHI', 'PPR'), ('GUAYAS - MILAGRO', 'MET')], [('GUAYAS - SAN JACINTO DE YAGUACHI', 'CON'), ('GUAYAS - SAMBORONDON', 'EDU'), ('GUAYAS - SAMBORONDON', 'INM')], [('GUAYAS - GUAYAQUIL', 'AYG'), ('GUAYAS - GUAYAQUIL', 'DEM'), ('GUAYAS - GUAYAQUIL', 'QU2'), ('GUAYAS - DAULE', 'CON')], [('LOS RIOS - QUEVEDO', 'FRT')], [('BOLIVAR - CALUMA', 'MAQ'), ('BOLIVAR - GUARANDA', 'ADM'), ('TUNGURAHUA - AMBATO', 'FRT')], [('CHIMBORAZO - GUANO', 'CON'), ('CHIMBORAZO - GUANO', 'MIP'), ('TUNGURAHUA - SAN PEDRO DE PELILEO', 'CON')], [('TUNGURAHUA - AMBATO', 'COM'), ('TUNGURAHUA - AMBATO', 'EDU'), ('COTOPAXI - SALCEDO', 'CON'), ('COTOPAXI - LATACUNGA', 'TRA')], [('PICHINCHA - QUITO', 'ADP')], [('PICHINCHA - QUITO', 'CAR')], [('PICHINCHA - QUITO', 'COM')], [('PICHINCHA - QUITO', 'CON')], [('PICHINCHA - QUITO', 'EDU')], [('PICHINCHA - QUITO', 'GAN')], [('PICHINCHA - QUITO', 'INM')], [('PICHINCHA - QUITO', 'MIP')], [('PICHINCHA - QUITO', 'SAL')], [('PICHINCHA - QUITO', 'TEL')], [('PICHINCHA - QUITO', 'TRA')], [('PICHINCHA - QUITO', 'ADM'), ('PICHINCHA - QUITO', 'ALD')], [('PICHINCHA - QUITO', 'ASO'), ('PICHINCHA - QUITO', 'BNA')], [('PICHINCHA - QUITO', 'AYG'), ('PICHINCHA - QUITO', 'FIN')], [('PICHINCHA - QUITO', 'BAL'), ('PICHINCHA - QUITO', 'RES')], [('PICHINCHA - QUITO', 'ELE'), ('PICHINCHA - QUITO', 'MAQ')], [('PICHINCHA - QUITO', 'CHO'), ('PICHINCHA - QUITO', 'LAC'), ('PICHINCHA - QUITO', 'PAN')], [('PICHINCHA - QUITO', 'DEM'), ('PICHINCHA - QUITO', 'QU2'), ('PICHINCHA - QUITO', 'SEG')], [('PICHINCHA - QUITO', 'HIL'), ('PICHINCHA - QUITO', 'MAD'), ('PICHINCHA - QUITO', 'SIL'), ('PICHINCHA - PEDRO VICENTE MALDONADO', 'GAN')], [('PICHINCHA - QUITO', 'MET'), ('PICHINCHA - QUITO', 'MUE'), ('PICHINCHA - QUITO', 'REF'), ('PICHINCHA - CAYAMBE', 'COM')], [('EL ORO - MACHALA', 'ADP'), ('EL ORO - MACHALA', 'COM'), ('EL ORO - MACHALA', 'CON'), ('EL ORO - MACHALA', 'INM')], [('MANABI - PORTOVIEJO', 'CON'), ('MANABI - PORTOVIEJO', 'EDU'), ('MANABI - PORTOVIEJO', 'PPR'), ('MANABI - PORTOVIEJO', 'SAL')], [('ESMERALDAS - RIO VERDE', 'MIP')], [('MANABI - JARAMIJO', 'PPR')], [('MANABI - JIPIJAPA', 'CON'), ('MANABI - JIPIJAPA', 'EDU'), ('MANABI - MONTECRISTI', 'AYG'), ('MANABI - MANTA', 'CON')]]
    
    return []


def flatten_to_str(x):
    """Flatten complex targets to string identifier."""
    if isinstance(x, str):
        return x
    elif isinstance(x, (list, tuple)):
        return '_'.join(flatten_to_str(i) for i in x)
    else:
        return str(x)


class DestructionExecutor(BaseExecutor, BatchProcessor):
    """
    Executes destruction analysis across multiple targets using the new architecture.
    
    This replaces the old DestructionExecutor with a clean separation using:
    - BatchProcessor interface for handling multiple scenarios
    - DestructionRunner: Executes individual destruction simulations
    - StandardDataCollector: Collects data across scenarios
    - CompositeAnalyzer: Analyzes results for each scenario
    - Results writer: Handles output (if provided)
    
    The batch processing handles different target types:
    - sectors: Single sectors or combinations
    - provinces/cantons: Geographic regions
    - province_sectors/canton_sectors: Combined geographic-sectoral targets
    """
    
    def __init__(self, model: "Model", parameters: "Parameters", 
                 target_types: str = "sectors", subregion: str = None, 
                 results_writer=None):
        super().__init__(model, parameters)
        self.target_types = target_types
        self.subregion = subregion
        self.results_writer = results_writer
        self.logger = logging.getLogger(__name__)
        
    def execute(self) -> BatchResult:
        """
        Execute destruction analysis and return batch results.
        
        This orchestrates multiple destruction scenarios and aggregates results.
        """
        start_time = datetime.now()
        self.logger.info(f"Starting destruction analysis: {self.target_types}")
        
        # Initialize batch result
        batch_result = BatchResult()
        batch_result.batch_metadata.update({
            'batch_type': 'destruction_analysis',
            'target_types': self.target_types,
            'subregion': self.subregion,
            'start_time': start_time.isoformat()
        })
        
        try:
            # Execute batch processing
            scenario_results = self.process_batch()
            batch_result.scenario_results = {
                result.analysis_metadata.get('scenario_id', f'scenario_{i}'): result 
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
            
            self.logger.info(f"✓ Destruction analysis complete: {len(scenario_results)} scenarios")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"✗ Destruction analysis failed: {e}")
            batch_result.batch_metadata['error'] = str(e)
            raise
    
    def process_batch(self) -> List[AnalysisResults]:
        """
        Process multiple destruction scenarios.
        
        This implements the BatchProcessor interface.
        """
        from disruptsc.model.utils.caching import load_cached_model
        
        # Save model state for reloading
        suffix = round(datetime.now().timestamp() * 1000)
        self.model.save_pickle(suffix)
        
        # Get target combinations based on type
        disrupted_targets_list, targets_in_model = self._get_target_combinations()
        
        # Get periods from parameters
        periods = getattr(self.parameters, 'destruction_periods', [30, 90, 180])
        results = []
        
        for disrupted_targets in disrupted_targets_list:
            try:
                # Prepare targets
                scenario_targets = self._prepare_scenario_targets(disrupted_targets, targets_in_model)
                if not scenario_targets:
                    continue
                
                scenario_id = flatten_to_str(scenario_targets)
                self.logger.info(f"=============== Disrupting {self.target_types} #{scenario_targets} ===============")
                
                # Load fresh model state
                model = load_cached_model(suffix)
                
                # Configure disruption filter
                model.parameters.disruptions[0]['filter'] = {}
                model.parameters.disruptions[0]['filter'][self.target_types] = scenario_targets
                
                # Execute single destruction scenario
                scenario_result = self._execute_single_scenario(model, scenario_id, periods)
                results.append(scenario_result)
                
                # Write results if writer provided
                if self.results_writer:
                    self._write_scenario_results(scenario_id, scenario_result, model)
                
                # Clean up memory
                del model
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Scenario failed for targets {disrupted_targets}: {e}")
                continue
        
        return results
    
    def _get_target_combinations(self):
        """Get target combinations and model targets based on destruction type."""
        if self.target_types == "sectors":
            disrupted_targets_list = _get_disrupted_sector_list()
            targets_in_model = self.model.firms.get_properties('sector', 'set')
            self.logger.info(f"{len(disrupted_targets_list)} sector combinations to test")
            
        elif self.target_types in ["canton", "province"]:
            disrupted_targets_list = _get_disrupted_subregion_list(self.subregion)
            targets_in_model = self.model.firms.get_subregions(self.subregion, 'set')
            self.logger.info(f"{len(disrupted_targets_list)} {self.subregion} combinations to test")
            
        elif self.target_types in ["canton_sector", "province_sector"]:
            disrupted_targets_list = _get_disrupted_subregion_sector_list(self.subregion)
            targets_in_model = self.model.firms.get_subregion_sectors(self.subregion, 'list')
            self.logger.info(f"{len(disrupted_targets_list)} {self.subregion}-sector combinations to test")
            
        else:
            raise ValueError(f"Unknown target_types: {self.target_types}")
            
        return disrupted_targets_list, targets_in_model
    
    def _prepare_scenario_targets(self, disrupted_targets, targets_in_model):
        """Prepare and validate targets for a scenario."""
        # Handle 'all' targets
        if disrupted_targets == 'all':
            disrupted_targets = targets_in_model
        
        # Skip if no targets present
        if all([target not in targets_in_model for target in disrupted_targets]):
            self.logger.info(f"No targets present in model: {disrupted_targets}")
            return None
        
        # Filter out targets not in model
        disrupted_targets_not_in_model = set(disrupted_targets) - set(targets_in_model)
        if len(disrupted_targets_not_in_model) > 0:
            self.logger.info(f"Skipping {disrupted_targets_not_in_model} - not present in model")
            disrupted_targets = list(set(disrupted_targets) & set(targets_in_model))
        
        return disrupted_targets
    
    def _execute_single_scenario(self, model, scenario_id: str, periods: List[int]) -> AnalysisResults:
        """Execute a single destruction scenario."""
        # Create custom executor for this scenario
        executor = BaseExecutor(model, self.parameters)
        executor.runner = DestructionRunner(model, self.parameters)
        executor.collector = StandardDataCollector(self.parameters)
        
        # Execute the scenario
        analysis_results = executor.execute()
        
        # Add scenario-specific metadata
        analysis_results.analysis_metadata.update({
            'scenario_id': scenario_id,
            'target_types': self.target_types,
            'destruction_periods': periods,
            'simulation_type': 'destruction_scenario'
        })
        
        return analysis_results
    
    def _write_scenario_results(self, scenario_id: str, results: AnalysisResults, model):
        """Write results using the provided results writer."""
        if self.results_writer and hasattr(self.results_writer, 'write_destruction_results'):
            # Create a mock simulation object with the data needed by the writer
            class MockSimulation:
                def __init__(self, results: AnalysisResults):
                    self.analysis_results = results
                
                def calculate_household_loss(self, household_table=None, **kwargs):
                    return results.get_metric('household', 'losses', {}).get('absolute_cumulated', 0)
                    
                def calculate_country_loss(self, **kwargs):
                    return results.get_metric('country', 'losses', {}).get('absolute_cumulated', 0)
            
            mock_simulation = MockSimulation(results)
            self.results_writer.write_destruction_results(
                scenario_id, mock_simulation, model.household_table, 
                self.parameters.monetary_units_in_model
            )
    
    def create_runner(self):
        """Not used in batch processing - scenarios create their own runners."""
        return DestructionRunner(self.model, self.parameters)
        
    def create_collector(self):
        """Not used in batch processing - scenarios create their own collectors."""
        return StandardDataCollector(self.parameters)