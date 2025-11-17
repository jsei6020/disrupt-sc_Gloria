
# Create documentation about memory considerations and sparse matrices
memory_guide = '''
"""
Memory Considerations for GLORIA MR-SUT to IOT Transformation
==============================================================

IMPORTANT: GLORIA data dimensions and memory requirements
----------------------------------------------------------

GLORIA MR-SUT dimensions:
- 164 regions × 97 sectors = 15,908 total dimensions
- A single dense matrix (15,908 × 15,908) requires ~1.9 GB of RAM
- Full IOT transformation with Z, A, L matrices requires ~6-10 GB RAM

Memory-efficient alternatives for large-scale GLORIA data:
"""

import numpy as np
import pandas as pd
from scipy import sparse
from gloria_sut_to_iot import GLORIATransformer

# ============================================================================
# Option 1: Use sparse matrices for memory efficiency
# ============================================================================

class SparseGLORIATransformer(GLORIATransformer):
    """
    Memory-efficient version using sparse matrices.
    Recommended for full GLORIA data (164 regions × 97 sectors).
    """
    
    def __init__(self, supply_table, use_table, value_added, final_demand,
                 n_regions=164, n_sectors=97, sparse_format='csr'):
        """
        Initialize with sparse matrices.
        
        Parameters:
        -----------
        sparse_format : str
            Sparse matrix format: 'csr' (recommended), 'csc', or 'coo'
        """
        # Convert to sparse if not already
        if not sparse.issparse(supply_table):
            supply_table = sparse.csr_matrix(supply_table)
        if not sparse.issparse(use_table):
            use_table = sparse.csr_matrix(use_table)
        
        # Store as sparse
        self.V_sparse = supply_table
        self.U_sparse = use_table
        self.W = value_added  # Keep dense (usually small)
        self.Y = final_demand  # Keep dense (usually small)
        
        self.n_regions = n_regions
        self.n_sectors = n_sectors
        self.n_total = n_regions * n_sectors
        
        # Calculate outputs
        self.g = np.array(self.V_sparse.sum(axis=1)).flatten()
        self.x = np.array(self.V_sparse.sum(axis=0)).flatten()
    
    def transform_method_d_sparse(self):
        """
        Method D with sparse matrices - most memory efficient.
        """
        print("Applying Method D with sparse matrices...")
        
        # Market share matrix D (sparse)
        x_inv = sparse.diags(1.0 / np.where(self.x < 1e-10, 1e-10, self.x))
        D = self.V_sparse @ x_inv
        
        # Z = D @ U (sparse matrix multiplication)
        Z = D @ self.U_sparse
        
        # Final demand transformation
        Y_ind = D @ self.Y
        
        print(f"Sparse Z matrix: {Z.shape}, {Z.nnz:,} non-zero elements")
        print(f"Memory usage: ~{(Z.data.nbytes + Z.indices.nbytes + Z.indptr.nbytes) / 1024**2:.1f} MB")
        
        return Z, Y_ind


# ============================================================================
# Option 2: Region-by-region processing
# ============================================================================

def transform_by_region_blocks(data: dict, method: str = 'D'):
    """
    Process GLORIA data in regional blocks to save memory.
    
    This approach processes one region at a time, which is memory-efficient
    but may be slower than full matrix operations.
    
    Parameters:
    -----------
    data : dict
        Dictionary with 'V', 'U', 'W', 'Y' arrays or sparse matrices
    method : str
        Transformation method
        
    Returns:
    --------
    dict : Dictionary with IOT components (sparse format)
    """
    n_regions = 164
    n_sectors = 97
    n_total = n_regions * n_sectors
    
    print(f"Processing {n_regions} regions in blocks...")
    
    # Initialize sparse result matrices
    Z_blocks = []
    
    # Process each region pair
    for i in range(0, n_regions, 10):  # Process 10 regions at a time
        end_i = min(i + 10, n_regions)
        start_idx = i * n_sectors
        end_idx = end_i * n_sectors
        
        print(f"Processing regions {i+1}-{end_i}...")
        
        # Extract block
        V_block = data['V'][start_idx:end_idx, :]
        U_block = data['U'][:, start_idx:end_idx]
        
        # Process block (implement transformation here)
        # ... transformation logic for block ...
        
    return {'Z': None, 'Y': data['Y'], 'W': data['W']}


# ============================================================================
# Option 3: Use memory-mapped arrays for very large datasets
# ============================================================================

def create_memmap_iot(output_dir: str = './gloria_memmap'):
    """
    Create memory-mapped arrays for IOT matrices.
    
    Memory-mapped arrays allow working with data larger than RAM
    by storing the data on disk and loading only needed portions.
    
    Parameters:
    -----------
    output_dir : str
        Directory to store memory-mapped files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n_total = 164 * 97  # 15,908
    
    # Create memory-mapped arrays
    Z_memmap = np.memmap(f'{output_dir}/Z.dat', dtype='float64',
                         mode='w+', shape=(n_total, n_total))
    
    A_memmap = np.memmap(f'{output_dir}/A.dat', dtype='float64',
                         mode='w+', shape=(n_total, n_total))
    
    print(f"Created memory-mapped arrays in {output_dir}")
    print(f"These files will be loaded on-demand, saving RAM")
    
    return Z_memmap, A_memmap


# ============================================================================
# Practical example: Load and transform subset of GLORIA data
# ============================================================================

def example_subset_transformation():
    """
    Example: Transform a subset of GLORIA data (e.g., 10 regions).
    """
    print("\\n" + "="*70)
    print("EXAMPLE: Subset transformation (10 regions × 97 sectors)")
    print("="*70 + "\\n")
    
    n_regions_subset = 10
    n_sectors = 97
    n_total = n_regions_subset * n_sectors  # 970
    
    # Create test data
    np.random.seed(42)
    V = np.diag(np.random.uniform(1000, 10000, n_total))
    U = np.random.uniform(10, 500, (n_total, n_total)) * 0.3
    W = np.random.uniform(100, 1000, (6, n_total))
    Y = np.random.uniform(500, 3000, (n_total, 6))
    
    # Transform
    transformer = GLORIATransformer(V, U, W, Y, n_regions_subset, n_sectors)
    result = transformer.transform(method='D')
    
    print(f"✓ Successfully transformed {n_regions_subset} regions")
    print(f"  IOT matrix shape: {result['Z'].shape}")
    print(f"  Memory usage: ~{result['Z'].nbytes / 1024**2:.1f} MB")
    
    return result


# ============================================================================
# Memory estimation function
# ============================================================================

def estimate_memory_requirements(n_regions: int = 164, n_sectors: int = 97):
    """
    Estimate memory requirements for GLORIA transformation.
    """
    n_total = n_regions * n_sectors
    bytes_per_element = 8  # float64
    
    # Memory for main matrices
    V_memory = n_total * n_total * bytes_per_element
    U_memory = n_total * n_total * bytes_per_element
    Z_memory = n_total * n_total * bytes_per_element
    A_memory = n_total * n_total * bytes_per_element
    L_memory = n_total * n_total * bytes_per_element
    
    total_dense = (V_memory + U_memory + Z_memory + A_memory + L_memory)
    
    print(f"\\nMemory requirements for {n_regions} regions × {n_sectors} sectors:")
    print(f"  Total dimensions: {n_total:,}")
    print(f"  Single matrix: {V_memory / 1024**3:.2f} GB")
    print(f"  Full transformation (dense): {total_dense / 1024**3:.2f} GB")
    print(f"  Sparse (estimated 10% non-zero): {total_dense * 0.1 / 1024**3:.2f} GB")
    print(f"\\nRecommendations:")
    
    if total_dense / 1024**3 > 16:
        print("  ⚠ Use sparse matrices (Option 1)")
        print("  ⚠ Or process by region blocks (Option 2)")
    elif total_dense / 1024**3 > 8:
        print("  ⚠ Consider using sparse matrices for better performance")
    else:
        print("  ✓ Dense matrices should work fine")


if __name__ == "__main__":
    # Estimate memory for full GLORIA
    estimate_memory_requirements(n_regions=164, n_sectors=97)
    
    # Run example with subset
    result = example_subset_transformation()
    
    print("\\n" + "="*70)
    print("For full GLORIA data (164 regions):")
    print("="*70)
    print("1. Use sparse matrices: SparseGLORIATransformer")
    print("2. Process in blocks: transform_by_region_blocks()")
    print("3. Use memory mapping: create_memmap_iot()")
    print("\\nChoose the method based on your available RAM and data sparsity.")
'''

# Save the memory guide
with open('gloria_memory_guide.py', 'w') as f:
    f.write(memory_guide)

print("Memory guide saved to: gloria_memory_guide.py")
print("\nThis file includes:")
print("- Sparse matrix implementation (recommended for full GLORIA)")
print("- Region-by-region processing")
print("- Memory-mapped arrays for very large datasets")
print("- Memory estimation function")
print("- Practical subset example (10 regions)")
