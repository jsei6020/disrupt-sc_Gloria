import json
import logging
import os
import gc
from pathlib import Path
from typing import Literal
import pandas as pd
import geopandas as gpd
import csv
import networkx as nx

from disruptsc.network.sc_network import ScNetwork
from disruptsc.parameters import Parameters


class Simulation(object):
    def __init__(
        self,
        simulation_type: str,
        parameters: Parameters,
    ):
        """
        Initialize Simulation with configurable streaming mode.
        
        Args:
            simulation_type: "initial_state", "event", "disruption", "stationary_test", "criticality"
            parameters: Parameters object with export_folder
            stream_mode: 
                - "full": Stream individual agent data + loss data (original behavior)
                - "loss_only": Stream only aggregated loss data (more efficient)
                - "none": No streaming, keep everything in RAM
        """
        admissible_types = ["initial_state", "event", "disruption", "stationary_test", "criticality"]
        if simulation_type not in admissible_types:
            raise ValueError(f"Simulation type should be {admissible_types}")
        
        admissible_stream_modes = ["full", "loss_only", "none"]
        if parameters.stream_mode not in admissible_stream_modes:
            raise ValueError(f"stream_mode should be one of {admissible_stream_modes}")
        
        self.type = simulation_type
        self.export_folder = parameters.export_folder
        self.stream_mode = parameters.stream_mode
        self.disruption_steps = parameters.disruptions[0]["start_time"] # adapt to take several disruptions

        # Determine if we stream at all
        self.streaming_mode = self.type in ["event", "disruption"] and self.stream_mode != "none"
        self.stream_individual_agents = self.stream_mode == "full"
        self.stream_losses = self.stream_mode in ["full", "loss_only"]

        if self.streaming_mode and self.export_folder:
            self._init_streaming_files(self.export_folder)
        else:
            # RAM-only mode
            self.firm_data = []
            self.country_data = []
            self.household_data = []
            self.sc_network_data = []
            self.transport_network_data = []
            self.loss_data = []  # For RAM mode loss accumulation

    def _init_streaming_files(self, export_folder):
        """Initialize streaming files based on stream_mode."""
        
        # Individual agent data files (only if stream_mode == "full")
        if self.stream_individual_agents:
            self.firm_data_file = open(export_folder / 'firm_data.jsonl', 'w')
            self.country_data_file = open(export_folder / 'country_data.jsonl', 'w')
            self.household_data_file = open(export_folder / 'household_data.jsonl', 'w')
            self.sc_network_data_file = open(export_folder / 'sc_network_data.jsonl', 'w')
        
        # Loss data files (always created if streaming losses)
        if self.stream_losses:
            # CSV for sorted/aggregated loss per region-sector-time
            self.loss_stream_file = open(export_folder / 'loss_stream_unsorted.csv', 'w', newline='')
            self.loss_stream_writer = csv.writer(self.loss_stream_file)
            self.loss_stream_writer.writerow(['region', 'sector', 'time_step', 'loss'])
            
            # CSV for country losses
            self.country_loss_file = open(export_folder / 'loss_per_country_streamed.csv', 'w', newline='')
            self.country_loss_writer = csv.writer(self.country_loss_file)
            self.country_loss_writer.writerow(['time_step', 'country', 'loss'])
            
                
        # Transport network data (always created if streaming)
        self.transport_network_data_file = open(export_folder / 'transport_network_data.jsonl', 'w')
        
        # Running totals for summary
        self.total_household_loss = 0.0
        self.total_country_loss = 0.0
        
        # In-memory accumulator for loss records (to batch sort before final export)
        self.loss_records_buffer = []
        self.country_loss_records_buffer = []

    def store_agent_data(
        self,
        time_step: int,
        household_table: pd.DataFrame,
        firms,
        households,
        countries
    ):
        """Store agent data with configurable streaming."""
        
        if not self.streaming_mode or not self.export_folder:
            # RAM mode: store everything
            self._store_agent_data_ram(time_step, household_table, firms, households, countries)
        else:
            if self.stream_individual_agents:
                # Stream individual agent records
                self._stream_individual_agents(time_step, household_table, firms, households, countries)
            
            if self.stream_losses:
                # Always compute and stream loss data
                self._stream_loss_data(time_step, household_table, households, countries)

    def _stream_individual_agents(self, time_step: int, household_table: pd.DataFrame, firms, households, countries):
        """Stream individual firm, household, country records to JSONL."""
        
        # Store firm data
        for firm in firms.values():
            record = {
                'time_step': time_step,
                'firm': firm.pid,
                'production': firm.production,
                'profit': firm.profit,
                'transport_cost': firm.finance['costs']['transport'],
                'input_cost': firm.finance['costs']['input'],
                'other_cost': firm.finance['costs']['other'],
                'inventory_duration': firm.current_inventory_duration,
                'generalized_transport_cost': firm.generalized_transport_cost,
                'usd_transported': firm.usd_transported,
                'tons_transported': firm.tons_transported,
                'tonkm_transported': firm.tonkm_transported
            }
            json.dump(record, self.firm_data_file)
            self.firm_data_file.write('\n')

        # Store country data
        for country in countries.values():
            record = {
                'time_step': time_step,
                'country': country.pid,
                'generalized_transport_cost': country.generalized_transport_cost,
                'usd_transported': country.usd_transported,
                'tons_transported': country.tons_transported,
                'tonkm_transported': country.tonkm_transported,
                'extra_spending': country.extra_spending,
                'consumption_loss': country.consumption_loss,
                'spending': sum(list(country.qty_purchased.values()))
            }
            json.dump(record, self.country_data_file)
            self.country_data_file.write('\n')

        # Store household data
        for household in households.values():
            record = {
                'time_step': time_step,
                'household': household.pid,
                'tot_consumption': household.tot_consumption,
                'spending_per_retailer': household.spending_per_retailer,
                'consumption_per_retailer': household.consumption_per_retailer,
                'extra_spending_per_sector': household.extra_spending_per_sector,
                'consumption_loss_per_sector': household.consumption_loss_per_sector,
                'extra_spending': household.extra_spending,
                'consumption_loss': household.consumption_loss
            }
            json.dump(record, self.household_data_file)
            self.household_data_file.write('\n')

        self.firm_data_file.flush()
        self.household_data_file.flush()
        self.country_data_file.flush()
        gc.collect()

    def _stream_loss_data(self, time_step: int, household_table: pd.DataFrame, households, countries):
        """
        At each time step:
        - aggregate household losses by (region, sector, time_step)
        - drop zero-loss cells
        - append to a temporary CSV (unsorted)
        """
        # --- country losses (unchanged, still streamed) ---
        for country in countries.values():
            loss = country.extra_spending + country.consumption_loss
            if loss > 0:
                record = {
                    'time_step': time_step,
                    'country': country.pid,
                    'extra_spending': country.extra_spending,
                    'consumption_loss': country.consumption_loss,
                    'loss': loss
                }

                self.country_loss_writer.writerow([time_step, country.pid, loss])
                self.total_country_loss += loss

        # --- household losses, aggregated per (region, sector, time_step) ---
        hh_rows = []
        for hh in households.values():
            # per-sector dicts (as in the old batch logic)
            extra_dict = getattr(hh, "extra_spending_per_sector", {}) or {}
            loss_dict = getattr(hh, "consumption_loss_per_sector", {}) or {}

            # union of all sector keys
            all_sectors = set(extra_dict.keys()) | set(loss_dict.keys())
            if not all_sectors:
                continue

            pid = int(hh.pid.replace("hh_", ""))
            region = household_table.loc[household_table['id'] == pid, 'region'].iloc[0]

            for sector in all_sectors:
                loss = extra_dict.get(sector, 0.0) + loss_dict.get(sector, 0.0)
                if loss <= 0:
                    continue  # remove zero-loss immediately

                hh_rows.append({
                    'region': region,
                    'sector': sector,
                    'time_step': time_step,
                    'loss': loss
                })
                self.total_household_loss += loss

        if hh_rows:
            df = pd.DataFrame(hh_rows)
            # group within this timestep (many households per region/sector)
            grouped = (
                df.groupby(['region', 'sector', 'time_step'], as_index=False)['loss']
                .sum()
            )
            # still explicit, but everything is already non-zero
            grouped = grouped[grouped['loss'] != 0]

            # append to temporary, unsorted CSV (opened in _init_streaming_files)
            for _, row in grouped.iterrows():
                self.loss_stream_writer.writerow(
                    [row['region'], row['sector'], int(row['time_step']), float(row['loss'])]
                )

            self.loss_stream_file.flush()

        self.country_loss_file.flush()
        gc.collect()
        
    def _store_agent_data_ram(self, time_step: int, household_table: pd.DataFrame, firms, households, countries):
        """Fallback for RAM-only mode."""
        
        self.firm_data += [
            {
                'time_step': time_step,
                'firm': firm.pid,
                'production': firm.production,
                'profit': firm.profit,
                'transport_cost': firm.finance['costs']['transport'],
                'input_cost': firm.finance['costs']['input'],
                'other_cost': firm.finance['costs']['other'],
                'inventory_duration': firm.current_inventory_duration,
                'generalized_transport_cost': firm.generalized_transport_cost,
                'usd_transported': firm.usd_transported,
                'tons_transported': firm.tons_transported,
                'tonkm_transported': firm.tonkm_transported
            }
            for firm in firms.values()
        ]
        
        self.country_data += [
            {
                'time_step': time_step,
                'country': country.pid,
                'generalized_transport_cost': country.generalized_transport_cost,
                'usd_transported': country.usd_transported,
                'tons_transported': country.tons_transported,
                'tonkm_transported': country.tonkm_transported,
                'extra_spending': country.extra_spending,
                'consumption_loss': country.consumption_loss,
                'spending': sum(list(country.qty_purchased.values()))
            }
            for country in countries.values()
        ]
        
        self.household_data += [
            {
                'time_step': time_step,
                'household': household.pid,
                'tot_consumption': household.tot_consumption,
                'spending_per_retailer': household.spending_per_retailer,
                'consumption_per_retailer': household.consumption_per_retailer,
                'extra_spending_per_sector': household.extra_spending_per_sector,
                'consumption_loss_per_sector': household.consumption_loss_per_sector,
                'extra_spending': household.extra_spending,
                'consumption_loss': household.consumption_loss
            }
            for household in households.values()
        ]
        
        # Accumulate loss data for RAM mode
        for household in households.values():
            loss = household.extra_spending + household.consumption_loss
            if loss > 0:
                pid = int(household.pid.replace("hh_", ""))
                region = household_table.loc[household_table['id'] == pid, 'region'].iloc[0]
                sector = getattr(household, 'sector', 'unknown')
                
                self.loss_data.append({
                    'time_step': time_step,
                    'household': household.pid,
                    'region': region,
                    'sector': sector,
                    'loss': loss
                })
                self.total_household_loss += loss

        for country in countries.values():
            loss = country.extra_spending + country.consumption_loss
            if loss > 0:
                self.total_country_loss += loss

    def store_sc_network_data(self, time_step: int, model):
        """Store supply chain network data (only if streaming individual agents)."""
        if not self.stream_individual_agents or not self.streaming_mode:
            return
        
        rows = [
            {
                'time_step': time_step,
                'pid': link.pid,
                'status': link.status,
                'price': link.price,
                'order': link.order,
                'delivery': link.delivery,
                "fulfilment_rate": link.fulfilment_rate
            }
            for link in list(nx.get_edge_attributes(model.sc_network, "object").values())
            if link.status != "ok"
        ]
        
        if self.streaming_mode and self.export_folder:
            for record in rows:
                json.dump(record, self.sc_network_data_file)
                self.sc_network_data_file.write('\n')
            self.sc_network_data_file.flush()
        else:
            self.sc_network_data += rows

    def store_transport_network_data(self, time_step, transport_network, transport_edges):
        """Store transport network data only at t=0 and disruption step."""
        if self.streaming_mode and self.export_folder:
            # only compute and export flows at initial state and disruption time
            timesteps_to_export = {0, self.disruption_steps}
            if time_step not in timesteps_to_export:
                return

            flow_data = transport_network.compute_flow_per_segment(time_step)
            if flow_data:
                for flow in flow_data:
                    json.dump(flow, self.transport_network_data_file)
                    self.transport_network_data_file.write('\n')
                flow_df = pd.DataFrame(flow_data)
                flow_df = flow_df[flow_df['flow_total'] > 0]
                transport_edges_with_flows = pd.merge(
                    transport_edges.drop(columns=["node_tuple"]), flow_df,
                    how="left", on="id"
                )
                transport_edges_with_flows.to_file(
                    self.export_folder / f"transport_edges_with_flows_{time_step}.geojson",
                    driver="GeoJSON", index=False
                )
            self.transport_network_data_file.flush()
            gc.collect()
        else:
            if time_step in (0, getattr(self, "disruption_start_step", 0)):
                self.transport_network_data.extend(
                    transport_network.compute_flow_per_segment(time_step)
                )


    def finalize_streaming_exports(self, monetary_unit_in_model: str):
        """Finalize streaming exports after simulation completes."""
        if not self.streaming_mode or not self.export_folder:
            return

        # Close files
        if self.stream_individual_agents:
            if hasattr(self, 'firm_data_file'):
                self.firm_data_file.close()
            if hasattr(self, 'country_data_file'):
                self.country_data_file.close()
            if hasattr(self, 'household_data_file'):
                self.household_data_file.close()
            if hasattr(self, 'sc_network_data_file'):
                self.sc_network_data_file.close()

        if self.stream_losses:
            if hasattr(self, 'loss_stream_file'):
                self.loss_stream_file.close()
            if hasattr(self, 'country_loss_file'):
                self.country_loss_file.close()
            # build main ordered loss file from streamed CSV
            self._finalize_loss_per_region_sector_time()



        if hasattr(self, 'transport_network_data_file'):
            self.transport_network_data_file.close()


        # Write summary stats
        summary = pd.DataFrame({
            "households": [self.total_household_loss],
            "countries": [self.total_country_loss]
        })
        summary.to_csv(self.export_folder / "loss_summary.csv", index=False)
 
    def calculate_and_export_summary_result(
        self,
        sc_network: ScNetwork,
        household_table: pd.DataFrame,
        monetary_unit_in_model: str,
        export_folder: Path
    ):
        """Calculate and export summary results."""
        
        if self.type in ["initial_state"]:
            logging.info(f'Exporting resulting IO matrix to {export_folder}')
            sc_network.calculate_io_matrix().to_csv(export_folder / "io_table.csv")
            logging.info(f'Exporting edgelist to {export_folder}')
            sc_network.generate_edge_list().to_csv(export_folder / "sc_network_edgelist.csv")

        else:  # event, disruption, etc.
            if self.stream_mode == "loss_only" or self.stream_mode == "full":
                # Loss data already streamed and sorted
                logging.debug("Loss data already exported during simulation (streaming mode)")
                
                # Just read the summary
                loss_summary = pd.read_csv(export_folder / "loss_summary.csv")
                household_loss = loss_summary.loc[0, 'households']
                country_loss = loss_summary.loc[0, 'countries']
                
            else:  # RAM mode
                # Process loss data from RAM
                household_result_table = pd.DataFrame(self.household_data)
                loss_per_region_sector_time = household_result_table.groupby('household').apply(
                    self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
                
                if export_folder:
                    logging.info(f'Exporting loss time series of households per region sector to {export_folder}')
                    loss_per_region_sector_time.to_csv(
                        export_folder / "loss_per_region_sector_time.csv", index=False)
                
                household_loss = loss_per_region_sector_time['loss'].sum()

            # Country loss
            if self.stream_mode in ["loss_only", "full"]:
                country_loss_df = pd.read_csv(export_folder / "loss_per_country_streamed.csv")
                country_loss = country_loss_df['loss'].sum()
            else:
                country_result_table = pd.DataFrame(self.country_data)
                if not country_result_table.empty and country_result_table["consumption_loss"].any():
                    country_result_table['loss'] = country_result_table['extra_spending'] \
                                                    + country_result_table['consumption_loss']
                    country_result_table = country_result_table[['time_step', 'country', 'loss']]
                    country_loss = country_result_table['loss'].sum()
                    if export_folder:
                        country_result_table.to_csv(export_folder / "loss_per_country.csv", index=False)
                else:
                    country_loss = 0
            
            logging.info(f"Cumulated household loss: {household_loss:,.2f} {monetary_unit_in_model}")
            logging.info(f"Cumulated country loss: {country_loss:,.2f} {monetary_unit_in_model}")

    def get_flow_specific_edges(
        self,
        edge_names: list,
        transport_edges: gpd.GeoDataFrame,
        usd_or_ton: str = 'usd'
    ):
        """Get flow data for specific edges."""
        flow_df = pd.DataFrame(self.transport_network_data)
        specific_edges_id_to_name = transport_edges.loc[
            transport_edges['name'].isin(edge_names), ['name', 'id']
        ]
        specific_edges_id_to_name = specific_edges_id_to_name.set_index('id')['name'].to_dict()
        flow_df = flow_df[flow_df['id'].isin(list(specific_edges_id_to_name.keys()))].copy()
        flow_df['name'] = flow_df['id'].map(specific_edges_id_to_name)
        col_to_report = 'flow_total' if usd_or_ton == 'usd' else 'flow_total_tons'
        return flow_df.set_index('name')[col_to_report].to_dict()

    def report_annual_flow_specific_edges(
        self,
        edge_names: list,
        transport_edges: gpd.GeoDataFrame,
        time_resolution: str,
        usd_or_ton: str = 'usd'
    ):
        """Report annualized flows for specific edges."""
        flows = self.get_flow_specific_edges(edge_names, transport_edges, usd_or_ton)
        periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
        flows = pd.Series(flows) * periods[time_resolution]
        return flows.to_dict()

    @staticmethod
    def summarize_results_one_household(household_result_table_one_household):
        """Summarize results for a single household."""
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

    def calculate_household_loss(
        self,
        household_table: pd.DataFrame,
        per_region: bool = False,
        periods: list = None
    ):
        """Calculate household losses with various aggregation options."""
        household_result_table = pd.DataFrame(self.household_data)
        loss_per_region_sector_time = household_result_table.groupby('household').apply(
            self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
        loss_per_region_sector_time['origin_region'] = loss_per_region_sector_time['sector'].str.extract(r'([A-Z]*)_')
        loss_per_region_sector_time['household_region'] = loss_per_region_sector_time['household'].map(
            household_table.set_index('household')['region'])
        
        if per_region:
            return loss_per_region_sector_time.groupby('household_region')['loss'].sum().to_dict()
        elif isinstance(periods, list):
            household_result_table['total_loss'] = household_result_table['extra_spending'] \
                                                    + household_result_table['consumption_loss']
            ts = household_result_table.groupby('time_step')['total_loss'].sum()
            baseline = household_result_table.loc[household_result_table['time_step'] == 0, 'tot_consumption'].sum()
            return {
                period: ts[:period].sum() / (baseline * period)
                for period in periods
            }
        else:
            return loss_per_region_sector_time['loss'].sum()

    def calculate_country_loss(self, per_country: bool = False):
        """Calculate country losses with optional per-country breakdown."""
        country_result_table = pd.DataFrame(self.country_data)
        country_result_table['loss'] = country_result_table['extra_spending'] \
                                       + country_result_table['consumption_loss']
        country_result_table = country_result_table[['time_step', 'country', 'loss']]
        
        if per_country:
            return country_result_table.groupby('country')['loss'].sum().to_dict()
        else:
            return country_result_table['loss'].sum()

    def _finalize_loss_per_region_sector_time(self):
        """
        Build the final, ordered loss_per_region_sector_time.csv from the
        streamed, unsorted per-step aggregation file.
        """
        path = self.export_folder / 'loss_stream_unsorted.csv'
        if not path.exists():
            return

        df = pd.read_csv(path)

        if df.empty:
            # nothing to do; optional: still write an empty file with header
            df.to_csv(self.export_folder / 'loss_per_region_sector_time.csv', index=False)
            return

        # optional extra safety: re-aggregate in case some (region, sector, time_step)
        # combos appeared in multiple chunks (shouldn't if you only ever add once per step)
        grouped = (
            df.groupby(['region', 'sector', 'time_step'], as_index=False)['loss']
            .sum()
        )

        # sort as requested: first by region, then sector, then time_step
        grouped = grouped.sort_values(['region', 'sector', 'time_step'])

        # write main output
        grouped.to_csv(self.export_folder / 'loss_per_region_sector_time.csv', index=False)

        #delete unsorted file for storage efficiency
        path.unlink()
