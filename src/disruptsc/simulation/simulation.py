import json
import logging
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd

from disruptsc.network.sc_network import ScNetwork


class Simulation(object):
    def __init__(self, simulation_type: str):
        admissible_types = ["initial_state", "event", "disruption", "stationary_test", "criticality",
                           "destruction_sectors", "destruction_provinces", "destruction_cantons", 
                           "destruction_province_sectors", "destruction_canton_sectors"]
        if simulation_type not in admissible_types:
            raise ValueError(f"Simulation type should be {admissible_types}")
        self.type = simulation_type
        self.firm_data = []
        self.country_data = []
        self.household_data = []
        self.sc_network_data = []
        self.transport_network_data = []

    def export_agent_data(self, export_folder):
        logging.info(f'Exporting agent data to {export_folder}')
        with open(os.path.join(export_folder, 'firm_data.json'), 'w') as jsonfile:
            json.dump(self.firm_data, jsonfile)
        with open(os.path.join(export_folder, 'country_data.json'), 'w') as jsonfile:
            json.dump(self.country_data, jsonfile)
        with open(os.path.join(export_folder, 'household_data.json'), 'w') as jsonfile:
            json.dump(self.household_data, jsonfile)

    def export_transport_network_data(self, transport_edges: gpd.GeoDataFrame, export_folder: Path):
        if export_folder:
            logging.info(f'Exporting transport network data to {export_folder}')
            flow_df = pd.DataFrame(self.transport_network_data)
            flow_df = flow_df[flow_df['flow_total'] > 0]
            for time_step in flow_df['time_step'].unique():
                transport_edges_with_flows = pd.merge(
                    transport_edges.drop(columns=["node_tuple"]), flow_df[flow_df['time_step'] == time_step],
                    how="left", on="id")
                transport_edges_with_flows.to_file(export_folder / f"transport_edges_with_flows_{time_step}.geojson",
                                                   driver="GeoJSON", index=False)

    def export_sc_network_matrices(self, sc_network: ScNetwork, export_folder: Path):
        """Export supply chain network matrices for initial_state simulation."""
        logging.info(f'Exporting resulting IO matrix to {export_folder}')
        sc_network.calculate_io_matrix().to_csv(export_folder / "io_table.csv")
        logging.info(f'Exporting edgelist to {export_folder}')
        sc_network.generate_edge_list().to_csv(export_folder / "sc_network_edgelist.csv")

    def export_times_series(self, household_table: pd.DataFrame, export_folder: Path):
        """Export detailed time series for disruption simulation."""
        self._export_household_time_series(household_table, export_folder)
        self._export_country_time_series(export_folder)

    def log_and_export_summary_results(self, household_table: pd.DataFrame, 
                                     monetary_unit_in_model: str, export_folder: Path):
        """Log and export summary results for disruption simulation and stationary_test."""
        # Calculate summary totals (for logging and summary CSV)
        household_loss = self.calculate_household_loss(household_table)
        country_loss = self.calculate_country_loss()
        
        # Log summary totals
        logging.info(f"Cumulated household loss: {household_loss:,.2f} {monetary_unit_in_model}")
        logging.info(f"Cumulated country loss: {country_loss:,.2f} {monetary_unit_in_model}")
        
        # Export loss summary CSV
        if export_folder:
            total_loss = pd.DataFrame({"households": household_loss, "countries": country_loss}, index=[0])
            total_loss.to_csv(export_folder / "loss_summary.csv", index=False)

    def calculate_and_export_summary_result(self, sc_network: ScNetwork, household_table: pd.DataFrame,
                                            monetary_unit_in_model: str, export_folder: Path):
        """Legacy method that calls the appropriate separated functions based on simulation type."""
        if self.type in ["initial_state"]:
            self.export_sc_network_matrices(sc_network, export_folder)
        elif self.type in ["disruption"]:
            # Default behavior for disruption simulations without Monte Carlo
            self.export_times_series(household_table, export_folder)
            self.log_and_export_summary_results(household_table, monetary_unit_in_model, export_folder)
        else:  # self.type == "event" or other types:
            self.export_times_series(household_table, export_folder)
            self.log_and_export_summary_results(household_table, monetary_unit_in_model, export_folder)

        # elif self.type == "criticality":
        #     household_loss = self.calculate_household_loss()
        #     country_loss = self.calculate_country_loss()

    def get_flow_specific_edges(self, edge_names: list, transport_edges: gpd.GeoDataFrame, usd_or_ton: str = 'usd'):
        flow_df = pd.DataFrame(self.transport_network_data)
        specific_edges_id_to_name = transport_edges.loc[transport_edges['name'].isin(edge_names), ['name', 'id']]
        specific_edges_id_to_name = specific_edges_id_to_name.set_index('id')['name'].to_dict()
        flow_df = flow_df[flow_df['id'].isin(list(specific_edges_id_to_name.keys()))].copy()
        flow_df['name'] = flow_df['id'].map(specific_edges_id_to_name)
        col_to_report = 'flow_total' if usd_or_ton == 'usd' else 'flow_total_tons'
        return flow_df.set_index('name')[col_to_report].to_dict()

    def report_annual_flow_specific_edges(self, edge_names: list, transport_edges: gpd.GeoDataFrame,
                                          time_resolution: str, usd_or_ton: str = 'usd'):
        flows = self.get_flow_specific_edges(edge_names, transport_edges, usd_or_ton)
        periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
        flows = pd.Series(flows) * periods[time_resolution]
        return flows.to_dict()

    @staticmethod
    def summarize_results_one_household(household_result_table_one_household):
        extra_spending_per_sector_table = pd.DataFrame(
            household_result_table_one_household.set_index('time_step')['extra_spending_per_sector'].to_dict()
        ).transpose()
        consumption_loss_per_sector_table = pd.DataFrame(
            household_result_table_one_household.set_index('time_step')['consumption_loss_per_sector'].to_dict()
        ).transpose()
        loss_per_sector = extra_spending_per_sector_table + consumption_loss_per_sector_table
        result = loss_per_sector.stack().reset_index()
        result.columns = ['time_step', 'sector', 'loss']
        return result

    @staticmethod
    def _calculate_stock_loss(loss_data, target_time_step, agent_col):
        """Calculate cumulative loss from start to target time step."""
        mask = loss_data['time_step'] <= target_time_step
        return loss_data[mask].groupby(agent_col)['total_loss'].sum()

    @staticmethod
    def _calculate_flow_loss(loss_data, target_time_step, agent_col):
        """Calculate loss at specific time step only."""
        mask = loss_data['time_step'] == target_time_step
        return loss_data[mask].groupby(agent_col)['total_loss'].sum()

    @staticmethod
    def _get_baseline_consumption(loss_data, baseline_time_step, agent_col, consumption_col):
        """Get baseline consumption for relative calculations."""
        baseline_mask = loss_data['time_step'] == baseline_time_step
        if consumption_col in loss_data.columns:
            return loss_data[baseline_mask].groupby(agent_col)[consumption_col].sum()
        else:
            # If no consumption column, return empty series
            return pd.Series(dtype=float, name=consumption_col)

    @staticmethod
    def _calculate_relative_loss(loss_total, baseline_total, duration=1):
        """Calculate relative loss with optional duration adjustment."""
        if baseline_total == 0:
            return 0.0
        return loss_total / (baseline_total * duration)

    @staticmethod
    def _aggregate_by_grouping_column(agent_data, grouping_table, agent_col, group_col):
        """Generic function to aggregate agent-level data by any grouping column."""
        agent_to_group = grouping_table.set_index(agent_col)[group_col].to_dict()
        group_data = {}
        for agent, value in agent_data.items():
            group = agent_to_group.get(agent, 'Unknown')
            group_data[group] = group_data.get(group, 0) + value
        return group_data

    def _export_household_time_series(self, household_table: pd.DataFrame, export_folder: Path):
        """Export detailed household loss time series by region and sector."""
        if not export_folder:
            return None
            
        household_result_table = pd.DataFrame(self.household_data)
        if household_result_table.empty:
            return None
            
        loss_per_region_sector_time = household_result_table.groupby('household').apply(
            self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
        household_table_copy = household_table.copy()
        household_table_copy['id'] = 'hh_' + household_table_copy['id'].astype(str)
        loss_per_region_sector_time['region'] = loss_per_region_sector_time['household'].map(
            household_table_copy.set_index('id')['region'])
        loss_per_region_sector_time = \
            loss_per_region_sector_time.groupby(['region', 'sector', 'time_step'], as_index=False)['loss'].sum()
        
        logging.info(f'Exporting loss time series of households per region sector to {export_folder}')
        loss_per_region_sector_time.to_csv(export_folder / "loss_per_region_sector_time.csv", index=False)
        return loss_per_region_sector_time

    def _export_country_time_series(self, export_folder: Path):
        """Export detailed country loss time series."""
        if not export_folder:
            return None
            
        country_result_table = pd.DataFrame(self.country_data)
        if country_result_table.empty:
            return None
            
        country_result_table['loss'] = country_result_table['extra_spending'] \
                                       + country_result_table['consumption_loss']
        country_result_table = country_result_table[['time_step', 'country', 'loss']]
        
        logging.info(f'Exporting loss time series of countries to {export_folder}')
        country_result_table.to_csv(export_folder / "loss_per_country.csv", index=False)
        return country_result_table

    def _calculate_generic_loss(self, agent_data, agent_table, agent_col, group_col,
                               consumption_col, calculation_type="stock", value_type="absolute",
                               time_steps=None, per_group=False, baseline_time_step=0):
        """
        Generic loss calculation function for any agent type (households, countries, etc.)
        
        Args:
            agent_data: List of agent data dictionaries
            agent_table: DataFrame with agent information and grouping column
            agent_col: Column name for agent identifier (e.g., 'household', 'country')
            group_col: Column name for grouping (e.g., 'region', 'country')
            consumption_col: Column name for baseline consumption data
            calculation_type: "stock" (cumulative) or "flow" (single time step)
            value_type: "absolute" or "relative" (to baseline consumption)
            time_steps: None (last step), int (specific step), or list (multiple steps)
            per_group: bool, whether to disaggregate by grouping column
            baseline_time_step: time step for baseline in relative calculations
        """
        # Prepare loss data
        loss_data = pd.DataFrame(agent_data)
        if loss_data.empty:
            return {} if per_group else (0.0 if time_steps is None or isinstance(time_steps, int) else {})

        loss_data['total_loss'] = loss_data['extra_spending'] + loss_data['consumption_loss']
        
        # Determine target time steps
        available_steps = sorted(loss_data['time_step'].unique())
        max_step = max(available_steps) if available_steps else 0
        
        if time_steps is None:
            target_steps = [max_step]
        elif isinstance(time_steps, int):
            target_steps = [time_steps]
        elif isinstance(time_steps, list):
            target_steps = time_steps
        else:
            raise ValueError("time_steps must be None, int, or list")

        results = {}
        for target_t in target_steps:
            # Calculate loss for this time step
            if calculation_type == "stock":
                agent_losses = self._calculate_stock_loss(loss_data, target_t, agent_col)
            elif calculation_type == "flow":
                agent_losses = self._calculate_flow_loss(loss_data, target_t, agent_col)
            else:
                raise ValueError("calculation_type must be 'stock' or 'flow'")

            # Convert to absolute or relative
            if value_type == "absolute":
                if per_group:
                    result = self._aggregate_by_grouping_column(agent_losses, agent_table, agent_col, group_col)
                else:
                    result = agent_losses.sum()
            elif value_type == "relative":
                baseline_data = self._get_baseline_consumption(loss_data, baseline_time_step, agent_col, consumption_col)
                if baseline_data.empty or baseline_data.sum() == 0:
                    result = {} if per_group else 0.0
                else:
                    duration = max(1, target_t - baseline_time_step) if calculation_type == "stock" else 1
                    if per_group:
                        loss_by_group = self._aggregate_by_grouping_column(agent_losses, agent_table, agent_col, group_col)
                        baseline_by_group = self._aggregate_by_grouping_column(baseline_data, agent_table, agent_col, group_col)
                        result = {group: self._calculate_relative_loss(loss_by_group.get(group, 0), 
                                                                     baseline_by_group.get(group, 0), duration)
                                for group in set(loss_by_group.keys()) | set(baseline_by_group.keys())}
                    else:
                        result = self._calculate_relative_loss(agent_losses.sum(), baseline_data.sum(), duration)
            else:
                raise ValueError("value_type must be 'absolute' or 'relative'")
            
            results[target_t] = result

        # Return single value or dict based on input
        return results[target_steps[0]] if len(target_steps) == 1 else results

    def calculate_household_loss(self, household_table: pd.DataFrame, 
                               calculation_type="stock", value_type="absolute", 
                               time_steps=None, per_region=False, baseline_time_step=0):
        """
        Calculate household losses with flexible options.
        
        Args:
            household_table: DataFrame with household information
            calculation_type: "stock" (cumulative) or "flow" (single time step)
            value_type: "absolute" or "relative" (to baseline consumption)
            time_steps: None (last step), int (specific step), or list (multiple steps)
            per_region: bool, whether to disaggregate by region
            baseline_time_step: time step for baseline in relative calculations
        """
        return self._calculate_generic_loss(
            agent_data=self.household_data,
            agent_table=household_table,
            agent_col='household',
            group_col='region',
            consumption_col='tot_consumption',
            calculation_type=calculation_type,
            value_type=value_type,
            time_steps=time_steps,
            per_group=per_region,
            baseline_time_step=baseline_time_step
        )

    def calculate_household_loss_legacy(self, household_table: pd.DataFrame, per_region=False, periods=None):
        """Backward compatibility wrapper for the old calculate_household_loss interface."""
        if isinstance(periods, list):
            return self.calculate_household_loss(
                household_table, 
                calculation_type="stock",
                value_type="relative", 
                time_steps=periods,
                per_region=False
            )
        else:
            return self.calculate_household_loss(
                household_table,
                calculation_type="stock",
                value_type="absolute",
                time_steps=None,
                per_region=per_region
            )

    def calculate_country_loss(self, country_table=None, calculation_type="stock", value_type="absolute", 
                             time_steps=None, per_country=False, baseline_time_step=0):
        """
        Calculate country losses with flexible options.
        
        Args:
            country_table: DataFrame with country information (optional, defaults to country names from data)
            calculation_type: "stock" (cumulative) or "flow" (single time step)
            value_type: "absolute" or "relative" (to baseline consumption)  
            time_steps: None (last step), int (specific step), or list (multiple steps)
            per_country: bool, whether to disaggregate by country
            baseline_time_step: time step for baseline in relative calculations
        """
        # Create default country table if not provided
        if country_table is None:
            country_data_df = pd.DataFrame(self.country_data)
            if not country_data_df.empty:
                unique_countries = country_data_df['country'].unique()
                country_table = pd.DataFrame({
                    'country': unique_countries,
                    'country_group': unique_countries  # Create separate column for grouping
                })
            else:
                country_table = pd.DataFrame(columns=['country', 'country_group'])
        
        return self._calculate_generic_loss(
            agent_data=self.country_data,
            agent_table=country_table,
            agent_col='country',
            group_col='country_group',
            consumption_col='tot_consumption',  # Countries may not have this, but keeping consistent
            calculation_type=calculation_type,
            value_type=value_type,
            time_steps=time_steps,
            per_group=per_country,
            baseline_time_step=baseline_time_step
        )

    def calculate_country_loss_legacy(self, per_country=False):
        """Backward compatibility wrapper for the old calculate_country_loss interface."""
        return self.calculate_country_loss(
            calculation_type="stock",
            value_type="absolute",
            time_steps=None,
            per_country=per_country
        )
