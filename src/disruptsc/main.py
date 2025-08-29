# Import modules
import cProfile
import logging
import pstats
import time
import argparse
import sys
from pathlib import Path

# Add src to Python path for direct script execution
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from disruptsc import paths
from disruptsc.model.utils.caching import generate_cache_parameters_from_command_line_argument
from disruptsc.parameters import Parameters
from disruptsc.model.model import Model
from disruptsc.simulation.orchestration.simulation_orchestrator import SimulationOrchestrator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DisruptSC: Spatial agent-based supply chain disruption model")
    parser.add_argument("scope", type=str, help="Region/scope to simulate")
    parser.add_argument("--cache", type=str, help="Caching behavior")
    parser.add_argument("--duration", type=int, help="Disruption duration")
    parser.add_argument("--io_cutoff", type=float, help="IO cutoff")
    parser.add_argument("--simulation_type", type=str, help="Simulation type")
    parser.add_argument("--cache_isolation", action="store_true", help="Isolate cache directory per process (for server runs)")
    parser.add_argument("--version", action="version", version=f"DisruptSC {__import__('disruptsc').__version__}")
    return parser.parse_args()


def setup_model(parameters, cache_parameters):
    """Setup and initialize the model."""
    model = Model(parameters)

    # Setup model components
    model.setup_transport_network(cache_parameters['transport_network'], parameters.with_transport)
    if parameters.with_output_folder and parameters.with_transport:
        model.export_transport_nodes_edges()
    
    model.setup_agents(cache_parameters['agents'])
    if parameters.with_output_folder:
        model.export_agent_tables()
    
    model.setup_sc_network(cache_parameters['sc_network'])
    model.set_initial_conditions()
    model.setup_logistic_routes(cache_parameters['logistic_routes'], parameters.with_transport)
    
    return model


def export_results(analysis_results, model, parameters):
    """Export simulation results if export_files is enabled.
    
    Note: In the new architecture, exports are typically handled by executors.
    This function is kept for backward compatibility and special cases.
    """
    if not parameters.export_files:
        return
    
    # Skip exports for certain simulation types (CSV-only output)
    if not parameters.with_output_folder:
        return
    
    # The new architecture handles exports through the executor
    # So this function primarily serves as a compatibility layer
    logging.info(f"Analysis completed with {len(analysis_results.export_tables)} export tables")
    
    # Log summary of results for user feedback
    if hasattr(analysis_results, 'metrics') and analysis_results.metrics:
        logging.info(f"Metrics calculated: {list(analysis_results.metrics.keys())}")
    
    # If export tables are available, log what was generated
    if analysis_results.export_tables:
        logging.info(f"Export tables available: {list(analysis_results.export_tables.keys())}")


def main():
    """Main entry point for DisruptSC simulations."""
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.time()
    
    # Parse arguments and load parameters
    args = parse_arguments()
    logging.info(f'Simulation starting for {args.scope}')
    
    # Setup cache isolation if requested
    if args.cache_isolation:
        from disruptsc.paths import setup_cache_isolation
        setup_cache_isolation(args.scope)
        logging.info(f'Cache isolation enabled for scope: {args.scope}')
    
    try:
        # Generate cache parameters
        cache_parameters = generate_cache_parameters_from_command_line_argument(args.cache)
        
        # Load and configure parameters
        parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, args.scope)
        if args.io_cutoff:
            parameters.io_cutoff = args.io_cutoff
        if args.duration:
            parameters.criticality['duration'] = args.duration
        
        # Setup output folder and logging
        parameters.initialize_exports()
        parameters.adjust_logging_behavior()
        
        # Setup model
        if parameters.is_monte_carlo or (parameters.simulation_type == "disruption-sensitivity"):
            model = Model(parameters)
        else:
            model = setup_model(parameters, cache_parameters)
        
        # Execute simulation using orchestrator (new architecture)
        if args.simulation_type:
            parameters.simulation_type = args.simulation_type
        orchestrator = SimulationOrchestrator(model, parameters)
        results = orchestrator.execute()
        
        # Log results (exports handled by orchestrator in new architecture)
        if results is not None:
            export_results(results, model, parameters)
        
        # Finish
        logging.info(f"End of simulation, running time {time.time() - t0}")
        
    finally:
        # Clean up isolated cache if enabled
        if args.cache_isolation:
            from disruptsc.paths import cleanup_isolated_cache
            cleanup_isolated_cache()
            logging.info('Isolated cache cleaned up')
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()


if __name__ == "__main__":
    main()
