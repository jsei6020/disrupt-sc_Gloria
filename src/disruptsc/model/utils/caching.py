import logging
import pickle
import threading
import networkx as nx
import pickle
from pathlib import Path
import glob


from disruptsc.network.mrio import Mrio
from disruptsc.paths import TMP_FOLDER, get_cache_dir


def generate_cache_parameters_from_command_line_argument(argument: str):
    # Generate cache parameters
    cache_parameters: dict[str, bool] = {
        "transport_network": False,
        "agents": False,
        "sc_network": False,
        "logistic_routes": False
    }
    if isinstance(argument, str):
        accepted_script_arguments: list[str] = [
            'same_transport_network_new_agents',
            'same_agents_new_sc_network',
            'same_sc_network_new_logistic_routes',
            'same_logistic_routes',
            'same_agents_new_transport_network',
            'same_sc_network_new_transport_network',
            'new_agents_same_all'
        ]
        if argument not in accepted_script_arguments:
            raise ValueError(f"Argument {argument} is not valid.\
                Possible values are: " + ','.join(accepted_script_arguments))
        cache_parameters: dict[str, bool] = {
            "transport_network": False,
            "agents": False,
            "sc_network": False,
            "logistic_routes": False
        }
        if argument == accepted_script_arguments[0]:
            cache_parameters['transport_network'] = True
        if argument == accepted_script_arguments[1]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = True
        if argument == accepted_script_arguments[2]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = True
            cache_parameters['sc_network'] = True
        if argument == accepted_script_arguments[3]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = True
            cache_parameters['sc_network'] = True
            cache_parameters['logistic_routes'] = True
        if argument == accepted_script_arguments[4]:
            cache_parameters['transport_network'] = False
            cache_parameters['agents'] = True
            cache_parameters['sc_network'] = False
            cache_parameters['logistic_routes'] = False
        if argument == accepted_script_arguments[5]:
            cache_parameters['transport_network'] = False
            cache_parameters['agents'] = True
            cache_parameters['sc_network'] = True
            cache_parameters['logistic_routes'] = False
        if argument == accepted_script_arguments[6]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = False
            cache_parameters['sc_network'] = True
            cache_parameters['logistic_routes'] = True
    return cache_parameters


def cache_agent_data(data_dic):
    pickle_filename = get_cache_dir() / 'firms_households_countries_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Firms, households, and countries saved in cache folder: {pickle_filename}')


def cache_model(model, suffix):
    pickle_filename = get_cache_dir() / f'model_{suffix}.pickle'
    pickle.dump(model, open(pickle_filename, 'wb'))
    logging.info(f'Model saved in cache folder: {pickle_filename}')


def cache_transport_network(data_dic):
    pickle_filename = get_cache_dir() / 'transport_network_pickle'
    # data_dic['transport_network'].cost_heuristic = None
    # data_dic['transport_network'].lock = None
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Transport network saved in cache folder: {pickle_filename}')


import pickle
from pathlib import Path

def cache_sc_network(data_dic, chunk_size=100000):
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data_dic.items():
    
        filename = cache_dir / f"{key}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(value, f)
    logging.info(f'Supply chain components saved (network chunked) in cache folder: {cache_dir}')


def cache_logistic_routes(data_dic):
    pickle_filename = get_cache_dir() / 'logistic_routes_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Logistics routes saved in cache folder: {pickle_filename}')


def load_cached_agent_data():
    pickle_filename = get_cache_dir() / 'firms_households_countries_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_mrio = tmp_data['mrio']
    loaded_sector_table = tmp_data['sector_table']
    # loaded_present_sectors = tmp_data['present_sectors']
    # loaded_flow_types_to_export = tmp_data['flow_types_to_export']
    loaded_firm_table = tmp_data['firm_table']
    loaded_household_table = tmp_data['household_table']
    loaded_firms = tmp_data['firms']
    loaded_households = tmp_data['households']
    loaded_countries = tmp_data['countries']
    logging.info('Firms, households, and countries generated from temp file.')
    logging.info("Nb firms: " + str(len(loaded_firms)))
    logging.info("Nb households: " + str(len(loaded_households)))
    logging.info("Nb countries: " + str(len(loaded_countries)))
    return loaded_mrio, loaded_sector_table, loaded_firms, loaded_firm_table, \
        loaded_households, loaded_household_table, loaded_countries


def load_cached_transaction_table():
    pickle_filename = get_cache_dir() / 'firms_households_countries_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_transaction_table = tmp_data['transaction_table']
    return loaded_transaction_table


def load_cached_model(suffix):
    pickle_filename = get_cache_dir() / f'model_{suffix}.pickle'
    model = pickle.load(open(pickle_filename, 'rb'))
    model.mrio = Mrio(model.mrio, monetary_units=model.parameters.monetary_units_in_data)
    return model


def load_cached_transport_network():
    pickle_filename = get_cache_dir() / 'transport_network_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_transport_network = tmp_data['transport_network']
    loaded_transport_nodes = tmp_data['transport_nodes']
    loaded_transport_edges = tmp_data['transport_edges']
    # loaded_transport_network.build_cost_heuristic()
    # loaded_transport_network.lock = threading.Lock()
    logging.info('Transport network generated from temp file.')
    return loaded_transport_network, loaded_transport_edges, loaded_transport_nodes


import gc
import pickle
import networkx as nx
from pathlib import Path
import glob

def load_cached_sc_network(checkpoint_frequency=5, max_edges_per_batch=50000):
    """
    Load supply chain network with periodic checkpointing and garbage collection.
    
    Args:
        checkpoint_frequency: Save a checkpoint every N chunks (helps recovery and RAM management)
        max_edges_per_batch: If a chunk is very large, break it into smaller batches before adding
    """
    cache_dir = get_cache_dir()
    loaded = {}
    
    for key in ["supply_chain_network", "firms", "households", "countries"]:
        filename = cache_dir / f"{key}.pkl"
        print(f"Loading {key}...")
        with open(filename, 'rb') as f:
            loaded[key] = pickle.load(f)
        print(f"Loaded {key}")

    gc.collect()  # Final cleanup
    logging.info('Supply chain components loaded from separate files (network chunked).')
    return loaded["supply_chain_network"], loaded["firms"], loaded["households"], loaded["countries"]



#def load_cached_sc_network():
#    print("yay")
    #pickle_filename = get_cache_dir() / 'supply_chain_pickle'
#    cache_dir = get_cache_dir()
#    loaded = {}
#    for key in ["supply_chain_network", "firms", "households", "countries"]:
#        filename = cache_dir / f"{key}.pkl"
#        print(filename)
#        with open(filename, 'rb') as f:
#            loaded[key] = pickle.load(f)
#        print(f"Loaded {key}")
#    logging.info('Supply chain components loaded from separate files.')
#    return loaded["supply_chain_network"], loaded["firms"], loaded["households"], loaded["countries"]    
    #print("sc_done")
    #loaded_sc_network = tmp_data['supply_chain_network']
    #print("sc")
    #loaded_firms = tmp_data['firms']
    #print("firms")
    #loaded_households = tmp_data['households']
    #print("households")
    #loaded_countries = tmp_data['countries']
    #print("countries")
    #logging.info('Supply chain generated from temp file.')
    #return loaded_sc_network, loaded_firms, loaded_households, loaded_countries


def load_cached_logistic_routes():
    pickle_filename = get_cache_dir() / 'logistic_routes_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sc_network = tmp_data['supply_chain_network']
    print("sc")
    loaded_transport_network = tmp_data['transport_network']
    print("transport")
    loaded_commercial_link_table = tmp_data['commercial_link_table']
    print("commercial")
    loaded_firms = tmp_data['firms']
    loaded_households = tmp_data['households']
    loaded_countries = tmp_data['countries']
    logging.info('Logistic routes generated from temp file.')
    return loaded_sc_network, loaded_transport_network, loaded_commercial_link_table, \
        loaded_firms, loaded_households, loaded_countries
