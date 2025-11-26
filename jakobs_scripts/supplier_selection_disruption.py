def overcome_export_stop(self, model, adapting_firms):
    #call select_suppliers again with new available suppliers
    #pass select_suppliers the appropriate firms list (for those in another region remove)
    #identify suppliers automatically adapts
    # current suppliers
    old_suppliers = self.suppliers
    #select suppliers is more focused on selecting one of several available suppliers within a region
    self.select_suppliers(model.sc_network, adapting_firms, model.countries,
                                        model.parameters.nb_suppliers_per_input,
                                        model.parameters.weight_localization_firm,
                                        model.parameters.logistics['sector_types_to_shipment_method'],
                                        import_label=model.mrio.import_label,
                                        transport_network=model.transport_network)

    for sector_id, sector_weight in self.input_mix.items():
        old_pid = old_suppliers.get(sector_id)
        new_pid = new_suppliers.get(inp)

        # only print when supplier changed
        if old_pid != new_pid:
            print(f"input={inp}, sector={self.sector}, region={self.region} old={old_pid}, new={new_pid}") 

def implement(self, model: "Model"):
    """Implement export stop."""
    #Can only do one country deciding to stop exporting at a time -> TODO make a loop for affected regions
    #Identify suppliers with new rules.
    print("Remove export controlled firms from available suppliers.")
    controlled_region = model.firms[self[0]].region
    controlled_pids = [self[i] for i in range(len(self))]
    adapting_firms = model.firms
    for pid in controlled_pids:
        if pid in adapting_firms:
            del adapting_firms[pid]
    #this is an instantiation of all firms excluding those controlled by the export stop
    adapting_firms._build_region_sector_index()
    for firm_id in model.firms:
        #if the firm is not in the controlled region and therefore loses access to a supplier
        #and a controlled firm is a supplier
        #print(firm_id, controlled_pids, model.firms[firm_id].suppliers)
        #if any(pid in model.firms[firm_id].suppliers for pid in controlled_pids):
            #print(model.firms[firm_id].region,  model.firms[firm_id].sector, controlled_region)
            #print(firm_id, controlled_pids, model.firms[firm_id].suppliers)

        if model.firms[firm_id].region != controlled_region \
        and any(pid in model.firms[firm_id].suppliers for pid in controlled_pids):  
            print(firm_id)
            print("Firm overcomes export stop.")
            model.firms[firm_id].disrupted_suppliers = controlled_pids
            model.firms[firm_id].overcome_export_stop(model, adapting_firms)
    if self.reconstruction_market:
        model.reconstruction_market = ReconstructionMarket(
            reconstruction_target_time=self.reconstruction_target_time,
            capital_input_mix=self.capital_input_mix
        )

    ###do the same for countries and households