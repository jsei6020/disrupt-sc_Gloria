if key == "supply_chain_network":
            print(f"Loading {key}...")
            G = nx.DiGraph()
            
            # Load nodes once
            with open(cache_dir / "supply_chain_network_nodes.pkl", 'rb') as f:
                nodes = pickle.load(f)
                G.add_nodes_from(nodes)
            
            del nodes
            gc.collect()  # Force garbage collection after deleting nodes
            print("Nodes loaded")
            
            # Stream edges in chunks, with periodic checkpointing
            edge_files = sorted(glob.glob(str(cache_dir / "supply_chain_network_edges_chunk_*.pkl")))
            total_edges = 0
            
            for chunk_idx, edge_file in enumerate(edge_files):
                with open(edge_file, 'rb') as f:
                    edges = pickle.load(f)
                    total_edges += len(edges)
                
                # If chunk is very large, add edges in smaller batches
                if len(edges) > max_edges_per_batch:
                    for batch_start in range(0, len(edges), max_edges_per_batch):
                        batch = edges[batch_start:batch_start + max_edges_per_batch]
                        G.add_edges_from(batch)
                else:
                    G.add_edges_from(edges)
                
                del edges
                gc.collect()  # Force garbage collection after each chunk
                
                print(f"Loaded chunk {chunk_idx + 1}/{len(edge_files)} ({total_edges} edges so far)")
                
                # Periodic checkpoint: save partial graph and reset if needed
                if (chunk_idx + 1) % checkpoint_frequency == 0:
                    checkpoint_file = cache_dir / f"supply_chain_network_checkpoint_{chunk_idx}.pkl"
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(G, f)
                    print(f"Checkpoint saved at {checkpoint_file}")
            
            loaded[key] = G
            print(f"Loaded {key} (chunked graph): {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        else:
        
        
        
        if key == "supply_chain_network":
            # Save nodes separately
            nodes = list(value.nodes(data=True))
            with open(cache_dir / "supply_chain_network_nodes.pkl", 'wb') as f:
                pickle.dump(nodes, f)
            # Save edges in chunks
            edges = list(value.edges(data=True))
            for i in range(0, len(edges), chunk_size):
                chunk = edges[i:i+chunk_size]
                filename = cache_dir / f"supply_chain_network_edges_chunk_{i//chunk_size:04d}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(chunk, f)
        else: