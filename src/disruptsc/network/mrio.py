from typing import TYPE_CHECKING

import logging
import os

import pandas as pd

from disruptsc.model.utils.functions import rescale_monetary_values

if TYPE_CHECKING:
    from pathlib import Path


EPSILON = 1e-6


class Mrio(pd.DataFrame):

    _metadata = [
        'region_sectors',
        'region_sector_names',
        'regions',
        'sectors',
        'external_buying_countries',
        'external_selling_countries',
        'region_households',
        'export_label',
        'final_demand_label',
        'capital_label',
        'import_label',
        'value_added_label',
        'tax_label',
        'monetary_units'
    ]

    def __init__(self, *args, monetary_units, **kwargs):
        super().__init__(*args, **kwargs)
        self.export_label = self.detect_level1_label('export', axis=1)
        self.final_demand_label = self.detect_level1_label('final.?demand|P.3|P.52|P.53', axis=1) #P.51| Capital
        self.capital_label = self.detect_level1_label('capital', axis=1)
        self.import_label = self.detect_level1_label('import', axis=0)
        self.value_added_label = self.detect_level1_label('value.?added|va|D.1|D.39|B.2n|B.3n|K.1', axis=0) #D.29 in tax?
        self.tax_label = self.detect_level1_label('tax', axis=0)
        self.check_square_structure()
        self.region_sectors = [tup for tup in self.columns
                               if tup[1] not in list(self.final_demand_label) + list(self.export_label) + list(self.capital_label)]
        self.region_sector_names = ['_'.join(tup) for tup in self.region_sectors]
        self.regions = list(set([tup[0] for tup in self.region_sectors]))
        self.sectors = list(set([tup[1] for tup in self.region_sectors]))
        self.external_buying_countries = [tup[0] for tup in self.columns if tup[1] in self.export_label]
        self.external_selling_countries = [tup[0] for tup in self.index if tup[1] in self.import_label]
        self.region_households = [tup for tup in self.columns.to_list() if tup[1] in self.final_demand_label]
        self.monetary_units = monetary_units
        self.adjust_output()

    def detect_level1_label(self, pattern: str, axis: int):
        sectors = pd.Series([], dtype=str)
        if axis == 0:
            sectors = self.index.get_level_values(1)
        elif axis == 1:
            sectors = self.columns.get_level_values(1)
        else:
            ValueError("Wrong axis selected")
        labels = sectors[sectors.str.contains(pattern, case=False)]
        if len(labels) == 0:
            logging.warning(f"Failed to detect the label used for {pattern} in the MRIO")
            return ""
        else:
            labels = labels.unique()
            return labels.values

    @classmethod
    def load_mrio_from_filepath(cls, filepath_mrio: "Path", monetary_units: str):
        _, ext = os.path.splitext(filepath_mrio)
        if ext == '.pkl' or ext == '.pickle':
            table = pd.read_pickle(filepath_mrio)
        elif ext == '.csv':
            table = pd.read_csv(filepath_mrio, header=[0, 1],  index_col=[0, 1])
        else:
            ValueError("Wrong MRIO filetype")
        # remove region_sectors with no flows
        zero_output = table.index[table.sum(axis=1) == 0].to_list()
        zero_input = table.columns[table.sum(axis=0) == 0].to_list()
        no_flow_region_sectors = list(set(zero_output) & set(zero_input))
        table.drop(index=no_flow_region_sectors, inplace=True)
        table.drop(columns=no_flow_region_sectors, inplace=True)
        return cls(table, monetary_units=monetary_units)

    def get_total_output_per_region_sectors(self, selected_industries=None):
        if not isinstance(selected_industries, list):
            selected_industries = self.region_sectors
        return self.loc[selected_industries].sum(axis=1)

    def get_total_input_per_region_sectors(self, selected_industries=None):
        if not isinstance(selected_industries, list):
            selected_industries = self.region_sectors
            #selected_industries = [tup for tup in self.index]
        return self[selected_industries].sum()

    def get_margin_per_industry(self, selected_industries=None):
        if not isinstance(selected_industries, list):
            selected_industries = self.region_sectors
        if not self.value_added_label.any():
            logging.warning("No value added in MRIO, defaulting to uniform 20% value added per industry")
            return {industry: 0.2 for industry in selected_industries}
        else:
            va = self.loc[(slice(None), list(self.value_added_label) + list(self.tax_label)), selected_industries]
            va = va.sum()
            output = self.loc[selected_industries].sum(axis=1)
            va_to_output_ratios = va / output
            #va_to_output_ratios.to_csv("/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/margins.csv")
            return va_to_output_ratios.to_dict()

    def get_transport_input_share_per_industry(self, sector_types: pd.Series | dict, selected_industries=None):
        if not isinstance(selected_industries, list):
            selected_industries = self.region_sectors
        transport_industries = [industry for industry in selected_industries
                                if sector_types[industry[1]].casefold() == "transport"]
        if len(transport_industries) == 0:
            logging.warning("No transport industry detected in the MRIO, defaulting to 0.2 transport input share")
            return {industry: 0.2 for industry in selected_industries}
        else:
            transport_input = self.loc[transport_industries, selected_industries].sum(axis=0)
            total_input = self[selected_industries].sum(axis=0)
            transport_to_input_ratios = transport_input / total_input
            return transport_to_input_ratios.to_dict()

    def get_region_sectors_with_internal_flows(self, threshold: float = 0):
        tot_outputs = self.get_total_output_per_region_sectors()
        matrix_output = pd.concat([tot_outputs] * len(self.index), axis=1).transpose()
        matrix_output.index = self.index
        tech_coef_matrix = self[self.region_sectors] / matrix_output
        return [tup for tup in self.region_sectors if tech_coef_matrix.loc[tup, tup] > threshold]

    def get_final_demand(self, selected_region_sectors=None):
        print(selected_region_sectors)
        if selected_region_sectors:
            if isinstance(selected_region_sectors[0], str):
                selected_region_sectors = [tuple(region_sector.split('_', 1))
                                           for region_sector in selected_region_sectors]
            elif isinstance(selected_region_sectors[0], tuple):
                pass
            else:
                raise ValueError('selected_region_sectors should be a list of tuples or strings')
            mask = self.columns.get_level_values(1).isin(self.final_demand_label)
            return self.loc[selected_region_sectors, mask]
        else:    
            mask = self.columns.get_level_values(1).isin(self.final_demand_label)
            return self.loc[:, mask]
    def check_square_structure(self):
        intermediary_matrix = self.get_intermediary_part()
        if intermediary_matrix.shape[0] != intermediary_matrix.shape[1]:
            logging.info(f"Column label not in row: {set(intermediary_matrix.columns) - set(intermediary_matrix.index)}")
            logging.info(f"Row label not in column: {set(intermediary_matrix.index) - set(intermediary_matrix.columns)}")
            raise ValueError('Intermediary matrix not square')

    def get_intermediary_part(self):
        columns_cut = list(self.final_demand_label) + list(self.export_label) + list(self.capital_label)
        rows_cut = labels_array = list(self.value_added_label) + list(self.import_label) + list(self.tax_label)
        region_sectors_in_columns = [tup for tup in self.columns
                                     if tup[1] not in columns_cut]
        region_sectors_in_rows = [tup for tup in self.index
                                  if tup[1] not in rows_cut]
        return self.loc[region_sectors_in_rows, region_sectors_in_columns]

    def adjust_output(self):
        total_output = self.get_total_output_per_region_sectors()
        total_input = self.get_total_input_per_region_sectors()
        unbalanced_region_sectors = total_input[total_input > total_output].index.to_list()
        if len(unbalanced_region_sectors) > 0:
            logging.warning(f"There are {len(unbalanced_region_sectors)} region_sectors with more inputs "
                            f"than outputs. Currently not correcting it.") # {unbalanced_region_sectors}
            # TODO very ad-hoc and dirty to accommodate with bad input file
            #this adds the difference between input and output as an average to external countries
            #for region_sector in unbalanced_region_sectors:
                # print(self.get_total_output_per_region_sectors()[region_sector])
                #print(total_input[region_sector] - total_output[region_sector])
                #where_to_add = pd.MultiIndex.from_product([self.external_buying_countries,
                #                                           [self.export_label]], names=['region', 'sector'])
                #how_much_to_add = (total_input[region_sector] - total_output[region_sector] + EPSILON) / len(where_to_add)
                #self.loc[region_sector, where_to_add] += how_much_to_add
                #print(self.get_total_output_per_region_sectors()[region_sector])
    
    def filtered_tech_coefficients(self, tech_coef_matrix, selected_region_sectors, threshold):
        import gc
        import numpy as np

        # Assume tech_coef_matrix.columns is a MultiIndex of tuples (country, sector)
        supplier_tuples = [tuple(col) for col in tech_coef_matrix.columns]
        is_domestic = np.array([tup in selected_region_sectors for tup in supplier_tuples])  # 1D array, shape (num_columns,)
        is_external = np.array([tup[0] in self.external_selling_countries for tup in supplier_tuples])  # 1D array
        supplier_mask = is_domestic | is_external  # 1D boolean mask for columns
        # Same fix for buyers (rows):
        buyer_tuples = [tuple(row) for row in tech_coef_matrix.index]
        buyer_mask = np.array([tup in selected_region_sectors for tup in buyer_tuples])  # 1D boolean mask for rows
        filtered_dict = {}
        chunk_size = 4000
        for start in range(0, len(buyer_tuples), chunk_size):
            print(start)
            end = start + chunk_size
            buyers_chunk = buyer_tuples[start:end]
            # Create a buyer mask for just this chunk
            chunk_mask = np.array([tup in selected_region_sectors for tup in buyers_chunk])
            chunk_df = tech_coef_matrix.loc[buyers_chunk, supplier_mask]
            chunk_df = chunk_df.where(chunk_df > threshold)
            chunk_dict = chunk_df.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()
            filtered_dict.update(chunk_dict)
            del buyers_chunk, chunk_mask, chunk_df, chunk_dict
            gc.collect()
        return filtered_dict

    def get_tech_coef_dict(self, threshold, selected_region_sectors):
        """
        returns a dict region_sector = {input_region_sector_1: tech_coef, ...}
        """
        tot_outputs = self.get_total_output_per_region_sectors()
        matrix_output = pd.concat([tot_outputs] * len(self.index), axis=1).transpose()
        matrix_output.index = self.index
        tot_outputs = tot_outputs.astype('float32')
        matrix_output = matrix_output.astype('float32')
        #tech_coef_matrix = self[self.region_sectors] / matrix_output
        import gc
        chunk_size = 1312  # set based on memory ceiling
        results = []
        for start in range(0, len(self.index), chunk_size):
            end = min(start + chunk_size, len(self.index))
            chunk = self[self.region_sectors].iloc[start:end] / matrix_output.iloc[start:end]
            results.append(chunk)
            del chunk
            gc.collect()
        tech_coef_matrix = pd.concat(results)
        del tot_outputs, matrix_output, results
        gc.collect()
        if selected_region_sectors:
            if isinstance(selected_region_sectors[0], str):
                selected_region_sectors = [tuple(region_sector.split('_', 1))
                                           for region_sector in selected_region_sectors]
            elif isinstance(selected_region_sectors[0], tuple):
                pass
            else:
                raise ValueError('selected_region_sectors should be a list of tuples or strings')
            result = dict(self.filtered_tech_coefficients(tech_coef_matrix, selected_region_sectors, threshold))
            return result
        else:
            return {
                '_'.join(region_sector_tuple): {
                    '_'.join(key): val
                    for key, val in sublist.items()
                    if val > threshold
                }
                for region_sector_tuple, sublist in tech_coef_matrix.to_dict().items()
            }

    @staticmethod
    def _filter_industries(series, cut_off_value, cut_off_type, cut_off_monetary_units, units_in_data):
        if cut_off_type == "percentage":
            rel_series = series / series.sum()
            return rel_series.index[rel_series > cut_off_value].to_list()
        elif cut_off_type == "absolute":
            unit_adjusted_cutoff = rescale_monetary_values(cut_off_value,
                                                           input_time_resolution="year",
                                                           target_time_resolution="year",
                                                           target_units=units_in_data,
                                                           input_units=cut_off_monetary_units)
            return series.index[series > unit_adjusted_cutoff].to_list()
        elif cut_off_type == "relative_to_average":
            cutoff = cut_off_value * series.sum() / series.shape[0]
            return series.index[series > cutoff].to_list()
        else:
            raise ValueError("cutoff type should be 'percentage', 'absolute', or 'relative_to_average'")

    def filter_industries_by_output(self, cut_off_value, cut_off_type, cut_off_monetary_units, units_in_data):
        tot_outputs = self.get_total_output_per_region_sectors().loc[self.region_sectors]
        return self._filter_industries(tot_outputs, cut_off_value, cut_off_type, cut_off_monetary_units, units_in_data)

    def filter_industries_by_final_demand(self, cut_off_value, cut_off_type, cut_off_monetary_units, units_in_data):
        final_demand = self.get_final_demand().sum(axis=1).loc[self.region_sectors]
        return self._filter_industries(final_demand, cut_off_value, cut_off_type, cut_off_monetary_units, units_in_data)
