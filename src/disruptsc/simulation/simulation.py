import json
import logging
import os
import gc
from pathlib import Path
import pandas as pd
import geopandas as gpd
import csv

from disruptsc.network.sc_network import ScNetwork
from disruptsc.parameters import Parameters


class Simulation(object):
    def __init__(self, simulation_type: str, parameters: Parameters):
        admissible_types = ["initial_state", "event", "disruption", "stationary_test", "criticality"]
        if simulation_type not in admissible_types:
            raise ValueError(f"Simulation type should be {admissible_types}")
        self.type = simulation_type
        self.export_folder = parameters.export_folder

        # Use streaming files for disruption/event type
        self.streaming_mode = self.type in ["event", "disruption"]
        if self.streaming_mode and self.export_folder:
            self._init_streaming_files(self.export_folder)
        else:
            self.firm_data = []
            self.country_data = []
            self.household_data = []
            self.sc_network_data = []
            self.transport_network_data = []
    
    def _init_streaming_files(self, export_folder):
        # JSONL for individual agent data
        self.firm_data_file = open(export_folder / 'firm_data.jsonl', 'w')
        self.country_data_file = open(export_folder / 'country_data.jsonl', 'w')
        self.household_data_file = open(export_folder / 'household_data.jsonl', 'w')
        self.sc_network_data_file = open(export_folder / 'sc_network_data.jsonl', 'w')
        self.transport_network_data_file = open(export_folder / 'transport_network_data.jsonl', 'w')

        # CSV writers for aggregated loss
        self.household_loss_file = open(export_folder / 'loss_per_region_time_streamed.csv', 'w', newline='')
        self.household_loss_writer = csv.writer(self.household_loss_file)
        self.household_loss_writer.writerow(['region', 'sector', 'time_step', 'loss'])

        self.country_loss_file = open(export_folder / 'loss_per_country_streamed.csv', 'w', newline='')
        self.country_loss_writer = csv.writer(self.country_loss_file)
        self.country_loss_writer.writerow(['time_step', 'country', 'loss'])

        # Running totals for summary
        self.total_household_loss = 0.0
        self.total_country_loss = 0.0

    def store_agent_data(self, time_step: int, household_table: pd.DataFrame, firms, households, countries):
        # Streaming mode
        if self.streaming_mode and self.export_folder:
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

            # Store country data and loss
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
                loss = record.get('extra_spending', 0) + record.get('consumption_loss', 0)
                if loss > 0:
                    self.country_loss_writer.writerow([time_step, country.pid, loss])
                    self.total_country_loss += loss

            # Store household data and aggregate losses per region-sector-timestep
            household_losses = []
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
                loss = record.get('extra_spending', 0) + record.get('consumption_loss', 0)
                if loss > 0:
                    pid = int(household.pid.replace("hh_", ""))
                    region = household_table.loc[household_table['id'] == pid, 'region'].iloc[0]
                    household_losses.append({
                        'region': region,
                        'time_step': time_step,
                        'loss': loss
                    })
            if household_losses:
                loss_df = pd.DataFrame(household_losses)
                for _, row in loss_df.groupby(['region', 'time_step']).sum().reset_index().iterrows():
                    self.household_loss_writer.writerow([row['region'], row['time_step'], row['loss']])
                    self.total_household_loss += row['loss']
            # Flush to disk and clean memory
            self.firm_data_file.flush()
            self.household_data_file.flush()
            self.country_data_file.flush()
            self.household_loss_file.flush()
            self.country_loss_file.flush()

            gc.collect()
        else:
            # Fallback: store in RAM for non-streaming type (initial_state, criticality)
            # Fallback for batch mode: retain in RAM
            simulation.firm_data += [
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
                for firm in self.firms.values()
            ]
            simulation.country_data += [
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
                for country in self.countries.values()
            ]
            simulation.household_data += [
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
                for household in self.households.values()
            ]

    def store_transport_network_data(self, time_step, transport_network, transport_edges):
        if self.streaming_mode and self.export_folder:
            # Compute and immediately export flows at just required timesteps
            flow_data = transport_network.compute_flow_per_segment(time_step)
            if flow_data:
                for flow in flow_data:
                    json.dump(flow, self.transport_network_data_file)
                    self.transport_network_data_file.write('\n')
                flow_df = pd.DataFrame(flow_data)
                flow_df = flow_df[flow_df['flow_total'] > 0]
                transport_edges_with_flows = pd.merge(
                    transport_edges.drop(columns=["node_tuple"]), flow_df,
                    how="left", on="id")
                transport_edges_with_flows.to_file(self.export_folder / f"transport_edges_with_flows_{time_step}.geojson",
                                                   driver="GeoJSON", index=False)
            self.transport_network_data_file.flush()
            gc.collect()
        else:
            # RAM mode
            self.transport_network_data.extend(transport_network.compute_flow_per_segment(time_step))

    def finalize_streaming_exports(self, monetary_unit_in_model):
        """Call after simulation ends to close and write summary"""
        if not self.streaming_mode or not self.export_folder:
            return
        self.country_data_file.close()
        self.household_data_file.close()
        self.transport_network_data_file.close()
        self.household_loss_file.close()
        self.country_loss_file.close()


        # Write summary stats
        summary = pd.DataFrame({"households": [self.total_household_loss], "countries": [self.total_country_loss]})
        summary.to_csv(self.export_folder / "loss_summary.csv", index=False)
        logging.info(f"Cumulated household loss: {self.total_household_loss:,.2f} {monetary_unit_in_model}")
        logging.info(f"Cumulated country loss: {self.total_country_loss:,.2f} {monetary_unit_in_model}")

    # The rest of your methods are left as in your code, except:
    # - `export_agent_data`, `export_transport_network_data`, `calculate_and_export_summary_result`
    #   now only operate on RAM mode (initial_state/etc), since streaming mode writes incrementally

    def calculate_and_export_summary_result(self, sc_network: ScNetwork, household_table: pd.DataFrame,
                                            monetary_unit_in_model: str, export_folder: Path):
        if self.type in ["initial_state"]:
            # export io matrix for equilibrium simulations
            logging.info(f'Exporting resulting IO matrix to {export_folder}')
            sc_network.calculate_io_matrix().to_csv(export_folder / "io_table.csv")
            logging.info(f'Exporting edgelist to {export_folder}')
            sc_network.generate_edge_list().to_csv(export_folder / "sc_network_edgelist.csv")

        else:# self.type == "event" or other disruption types:
            # export loss time series for households
            household_result_table = pd.read_json(
                self.export_folder / 'household_data.jsonl', 
                lines=True
            )
            #household_result_table = pd.DataFrame(self.household_data)
            loss_per_region_sector_time = household_result_table.groupby('household').apply(
                self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
            household_table['id'] = 'hh_' + household_table['id'].astype(str)
            loss_per_region_sector_time['region'] = loss_per_region_sector_time['household'].map(
                household_table.set_index('id')['region'])
            loss_per_region_sector_time = \
                loss_per_region_sector_time.groupby(['region', 'sector', 'time_step'], as_index=False)['loss'].sum()
            if export_folder:
                logging.info(f'Exporting loss time series of households per region sector to {export_folder}')
                loss_per_region_sector_time.to_csv(export_folder / "loss_per_region_sector_time.csv", index=False)
            household_loss = loss_per_region_sector_time['loss'].sum()
            logging.info(f"Cumulated household loss: {household_loss:,.2f} {monetary_unit_in_model}")

            # export loss time series for countries
            country_result_table = pd.read_json(
                self.export_folder / 'country_data.jsonl', 
                lines=True
            )
            if (country_result_table["consumption_loss"].any()):
                country_result_table['loss'] = country_result_table['extra_spending'] \
                                            + country_result_table['consumption_loss']
                country_result_table = country_result_table[['time_step', 'country', 'loss']]
                country_loss = country_result_table['loss'].sum()
                print(country_loss)
                if export_folder:
                    logging.info(f'Exporting loss time series of countries to {export_folder}')
                    country_result_table.to_csv(export_folder / "loss_per_country.csv", index=False)
            else:
                country_loss = 0
            logging.info(f"Cumulated country loss: {country_loss:,.2f} {monetary_unit_in_model}")
            # Export summary
            total_loss = pd.DataFrame({"households": household_loss, "countries": country_loss}, index=[0])
            if export_folder:
                total_loss.to_csv(export_folder / "loss_summary.csv", index=False)

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

    def calculate_household_loss(self, household_table: pd.DataFrame, per_region=False, periods=None):
        household_result_table = pd.DataFrame(self.household_data)
        loss_per_region_sector_time = household_result_table.groupby('household').apply(
            self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
        loss_per_region_sector_time['origin_region'] = loss_per_region_sector_time['sector'].str.extract(r'([A-Z]*)_')
        loss_per_region_sector_time['household_region'] = loss_per_region_sector_time['household'].map(
            household_table.set_index('household')['region'])
        if per_region:
            return loss_per_region_sector_time.groupby('household_region')['loss'].sum().to_dict()
        elif isinstance(periods, list):
            household_result_table['total_loss'] = household_result_table['extra_spending'] + household_result_table['consumption_loss']
            ts = household_result_table.groupby('time_step')['total_loss'].sum()
            baseline = household_result_table.loc[household_result_table['time_step'] == 0, 'tot_consumption'].sum()
            return {
                period: ts[:period].sum() / (baseline * period)
                for period in periods
            }
        else:
            return loss_per_region_sector_time['loss'].sum()

    def calculate_country_loss(self, per_country=False):
        country_result_table = pd.DataFrame(self.country_data)
        country_result_table['loss'] = country_result_table['extra_spending'] \
                                       + country_result_table['consumption_loss']
        country_result_table = country_result_table[['time_step', 'country', 'loss']]
        if per_country:
            return country_result_table.groupby('country')['loss'].sum().to_dict()
        else:
            return country_result_table['loss'].sum()
