import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from disruptsc import paths

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


def get_simplified_timestamp(last_n_digits: int = 10):
    return str(round(datetime.now().timestamp()*1e5))[-last_n_digits:]


class CSVResultsWriter:
    """Handles writing simulation results to CSV files."""

    def __init__(self, output_file: Path, headers: list):
        self.output_file = output_file
        self.headers = headers
        self._write_headers()

    def _write_headers(self):
        """Write CSV headers."""
        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)

    def write_row(self, data: list):
        """Write a single row of data."""
        with open(self.output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    @classmethod
    def create_disruption_mc_writer(cls, parameters: "Parameters") -> "DisruptionMCWriter":
        """Create writer for disruption Monte Carlo results."""
        process_pid = os.getpid()
        suffix = get_simplified_timestamp()
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"disruption_{suffix}_{process_pid}.csv"
        return DisruptionMCWriter(output_file, parameters)

    @classmethod
    def create_criticality_writer(cls, parameters: "Parameters") -> "CriticalityWriter":
        """Create writer for criticality analysis results."""
        suffix = get_simplified_timestamp()
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"criticality_{suffix}.csv"
        return CriticalityWriter(output_file, parameters)

    @classmethod
    def create_destruction_writer(cls, parameters: "Parameters") -> "DestructionWriter":
        """Create writer for ad-hoc analysis results."""
        suffix = get_simplified_timestamp()
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"destruction_{suffix}.csv"
        return DestructionWriter(output_file, parameters)

    @classmethod
    def create_sensitivity_writer(cls, parameters: "Parameters") -> "SensitivityWriter":
        """Create writer for sensitivity analysis results."""
        suffix = get_simplified_timestamp()
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"sensitivity_{suffix}.csv"
        return SensitivityWriter(output_file, parameters)


class DisruptionMCWriter(CSVResultsWriter):
    """Writer for disruption Monte Carlo simulation results."""

    def __init__(self, output_file: Path, parameters: "Parameters"):
        # Need to create a dummy model to get region info - this is a limitation of current design
        from disruptsc.model.model import Model
        temp_model = Model(parameters)
        temp_model._prepare_mrio_and_sectors()

        region_household_loss_labels = ['household_loss_' + region for region in temp_model.mrio.regions]
        country_loss_labels = ['country_loss_' + country for country in temp_model.mrio.external_buying_countries]

        headers = ["mc_repetition", "household_loss", "country_loss"] + \
                  region_household_loss_labels + country_loss_labels

        super().__init__(output_file, headers)
        self.parameters = parameters


    def write_iteration_results(self, iteration: int, simulation: "Simulation", model: "Model"):
        """Write results for a single Monte Carlo iteration."""
        # Check if this is a baseline simulation (no disruptions occurred)
        if not hasattr(model, 'disruption_list') or len(model.disruption_list) == 0:
            # Write zeros for all loss values
            household_loss = 0.0
            country_loss = 0.0
            household_loss_per_region_values = [0.0 for _ in model.mrio.regions]
            country_loss_per_region_values = [0.0 for _ in model.mrio.external_buying_countries]

            logging.info(f"No disruptions occurred - writing zero losses for iteration {iteration}")
        else:
            # Normal disruption simulation - calculate actual losses
            household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
            household_loss = sum(household_loss_per_region.values())
            country_loss_per_country = simulation.calculate_country_loss(per_country=True)
            country_loss = sum(country_loss_per_country.values())

            logging.info(f"Simulation terminated. "
                         f"Household loss: {int(household_loss)} {self.parameters.monetary_units_in_model}. "
                         f"Country loss: {int(country_loss)} {self.parameters.monetary_units_in_model}.")

            household_loss_per_region_values = [
                household_loss_per_region.get(region, 0.0) for region in model.mrio.regions
            ]
            country_loss_per_region_values = [
                country_loss_per_country.get(country, 0.0) for country in model.mrio.external_buying_countries
            ]

        self.write_row([iteration, household_loss, country_loss] +
                       household_loss_per_region_values + country_loss_per_region_values)


class CriticalityWriter(CSVResultsWriter):
    """Writer for criticality analysis results."""

    def __init__(self, output_file: Path, parameters: "Parameters"):
        # Need to create a dummy model to get region info
        from disruptsc.model.model import Model
        temp_model = Model(parameters)
        temp_model._prepare_mrio_and_sectors()

        region_household_loss_labels = ['household_loss_' + region for region in temp_model.mrio.regions]
        country_loss_labels = ['country_loss_' + country for country in temp_model.mrio.external_buying_countries]

        headers = ["edge", "name", "type", parameters.criticality['attribute'], "duration",
                   "household_loss", "country_loss"] + \
                  region_household_loss_labels + country_loss_labels + ['geometry']

        super().__init__(output_file, headers)
        self.parameters = parameters

    def write_criticality_results(self, edge: int, attribute, simulation: "Simulation", model: "Model",
                                  household_loss: float, country_loss: float):
        """Write results for a single edge criticality analysis."""
        household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
        country_loss_per_country = simulation.calculate_country_loss(per_country=True)

        household_loss_per_region_values = [
            household_loss_per_region.get(region, 0.0) for region in model.mrio.regions
        ]
        country_loss_per_region_values = [
            country_loss_per_country.get(country, 0.0) for country in model.mrio.external_buying_countries
        ]

        geometry = model.transport_edges.loc[edge, 'geometry']
        name = model.transport_edges.loc[edge, 'name']
        transport_type = model.transport_edges.loc[edge, 'type']

        self.write_row([edge, name, transport_type, attribute, self.parameters.criticality['duration'],
                        household_loss, country_loss] +
                       household_loss_per_region_values + country_loss_per_region_values + [geometry])


class DestructionWriter(CSVResultsWriter):
    """Writer for ad-hoc analysis results."""

    def __init__(self, output_file: Path, parameters: "Parameters"):
        # Get periods from parameters, with default fallback
        periods = getattr(parameters, 'destruction_periods', [30, 90, 180])
        
        # Define comprehensive header set for all loss metrics
        headers = [
            "sectors",
            # Household loss metrics
            "household_loss_absolute_cumulated",
            "household_loss_relative_cumulated", 
            # Household loss per time target (cumulated)
            *[f"household_loss_cumulated_abs_t{period}" for period in periods],
            *[f"household_loss_cumulated_rel_t{period}" for period in periods],
            # Household loss instant (flow)
            *[f"household_loss_instant_abs_t{period}" for period in periods],
            *[f"household_loss_instant_rel_t{period}" for period in periods],
            # Country loss metrics
            "country_loss_absolute_cumulated",
            "country_loss_relative_cumulated",
            # Country loss per time target (cumulated)
            *[f"country_loss_cumulated_abs_t{period}" for period in periods],
            *[f"country_loss_cumulated_rel_t{period}" for period in periods],
            # Country loss instant (flow)
            *[f"country_loss_instant_abs_t{period}" for period in periods],
            *[f"country_loss_instant_rel_t{period}" for period in periods]
        ]

        super().__init__(output_file, headers)
        self.parameters = parameters
        self.periods = periods

    def write_destruction_results(self, disruption_identifier: list | str, simulation: "Simulation", 
                                  household_table, monetary_unit_in_model: str):
        """Write comprehensive destruction results for a single analysis."""
        if isinstance(disruption_identifier, list):
            disruption_identifier = "_".join(disruption_identifier)
        
        # Calculate all required household loss metrics
        household_losses = {
            # Absolute cumulated loss (default behavior)
            'absolute_cumulated': simulation.calculate_household_loss(
                household_table, calculation_type="stock", value_type="absolute", time_steps=None
            ),
            # Relative cumulated loss 
            'relative_cumulated': simulation.calculate_household_loss(
                household_table, calculation_type="stock", value_type="relative", time_steps=None
            ),
            # Per time target cumulated losses (absolute and relative)
            'per_time_cumulated_absolute': simulation.calculate_household_loss(
                household_table, calculation_type="stock", value_type="absolute", time_steps=self.periods
            ),
            'per_time_cumulated_relative': simulation.calculate_household_loss(
                household_table, calculation_type="stock", value_type="relative", time_steps=self.periods
            ),
            # Instant losses (flow) at target time steps - absolute and relative
            'instant_absolute': simulation.calculate_household_loss(
                household_table, calculation_type="flow", value_type="absolute", time_steps=self.periods
            ),
            'instant_relative': simulation.calculate_household_loss(
                household_table, calculation_type="flow", value_type="relative", time_steps=self.periods
            )
        }
        
        # Calculate all required country loss metrics
        country_losses = {
            # Absolute cumulated loss (default behavior)
            'absolute_cumulated': simulation.calculate_country_loss(
                calculation_type="stock", value_type="absolute", time_steps=None
            ),
            # Relative cumulated loss
            'relative_cumulated': simulation.calculate_country_loss(
                calculation_type="stock", value_type="relative", time_steps=None
            ),
            # Per time target cumulated losses (absolute and relative)
            'per_time_cumulated_absolute': simulation.calculate_country_loss(
                calculation_type="stock", value_type="absolute", time_steps=self.periods
            ),
            'per_time_cumulated_relative': simulation.calculate_country_loss(
                calculation_type="stock", value_type="relative", time_steps=self.periods
            ),
            # Instant losses (flow) at target time steps - absolute and relative
            'instant_absolute': simulation.calculate_country_loss(
                calculation_type="flow", value_type="absolute", time_steps=self.periods
            ),
            'instant_relative': simulation.calculate_country_loss(
                calculation_type="flow", value_type="relative", time_steps=self.periods
            )
        }
        
        # Build row data with all metrics
        row_data = [disruption_identifier]
        
        # Add household loss metrics
        row_data.extend([
            household_losses['absolute_cumulated'],
            household_losses['relative_cumulated'],
            # Per time cumulated absolute
            *[household_losses['per_time_cumulated_absolute'].get(period, 0.0) for period in self.periods],
            # Per time cumulated relative  
            *[household_losses['per_time_cumulated_relative'].get(period, 0.0) for period in self.periods],
            # Instant absolute
            *[household_losses['instant_absolute'].get(period, 0.0) for period in self.periods],
            # Instant relative
            *[household_losses['instant_relative'].get(period, 0.0) for period in self.periods]
        ])
        
        # Add country loss metrics
        row_data.extend([
            country_losses['absolute_cumulated'],
            country_losses['relative_cumulated'],
            # Per time cumulated absolute
            *[country_losses['per_time_cumulated_absolute'].get(period, 0.0) for period in self.periods],
            # Per time cumulated relative
            *[country_losses['per_time_cumulated_relative'].get(period, 0.0) for period in self.periods],
            # Instant absolute
            *[country_losses['instant_absolute'].get(period, 0.0) for period in self.periods],
            # Instant relative
            *[country_losses['instant_relative'].get(period, 0.0) for period in self.periods]
        ])
        
        # Write the row
        self.write_row(row_data)
        
        # Log summary
        logging.info(f"Destruction results for {disruption_identifier}:")
        logging.info(f"  Household Loss (Absolute Cumulated): {household_losses['absolute_cumulated']:,.2f} {monetary_unit_in_model}")
        logging.info(f"  Household Loss (Relative Cumulated): {household_losses['relative_cumulated']:.4%}")
        logging.info(f"  Country Loss (Absolute Cumulated): {country_losses['absolute_cumulated']:,.2f} {monetary_unit_in_model}")
        logging.info(f"  Country Loss (Relative Cumulated): {country_losses['relative_cumulated']:.4%}")
        
        return household_losses, country_losses


class SensitivityWriter(CSVResultsWriter):
    """Writer for sensitivity analysis results."""

    def __init__(self, output_file: Path, parameters: "Parameters"):
        # Build headers dynamically based on sensitivity parameters
        if not parameters.sensitivity:
            raise ValueError("No sensitivity parameters defined")
        
        param_headers = list(parameters.sensitivity.keys())
        headers = ["combination_id"] + param_headers + ["household_loss", "country_loss"]

        super().__init__(output_file, headers)
        self.parameters = parameters

    def write_sensitivity_results(self, combination_id: int, combination: dict, 
                                  household_loss: float, country_loss: float):
        """Write results for a single parameter combination."""
        param_values = [combination[param] for param in combination.keys()]
        self.write_row([combination_id] + param_values + [household_loss, country_loss])
