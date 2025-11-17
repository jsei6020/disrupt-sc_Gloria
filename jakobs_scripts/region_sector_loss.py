import pandas as pd

household_result_table = pd.read_json(
                '/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/20251116_162216/household_data.jsonl', 
                lines=True
            )
            #household_result_table = pd.DataFrame(self.household_data)
loss_per_region_sector_time = household_result_table.groupby('household').apply(
    summarize_results_one_household).reset_index().drop(columns=['level_1'])
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