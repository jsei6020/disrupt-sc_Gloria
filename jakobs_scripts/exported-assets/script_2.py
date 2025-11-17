
# Create a practical usage example with pandas
usage_example = '''
"""
Practical Usage Example: GLORIA MR-SUT to IOT Transformation
=============================================================

This script shows how to use the GLORIATransformer class with actual data
loaded from files and how to save results as CSV files.
"""

import numpy as np
import pandas as pd
from gloria_sut_to_iot import GLORIATransformer

# ============================================================================
# EXAMPLE 1: Loading GLORIA data from CSV files
# ============================================================================

def load_gloria_from_csv(base_path: str, year: int = 2015):
    """
    Load GLORIA MR-SUT data from CSV files.
    
    Assumes GLORIA data is stored in separate CSV files for each component.
    Adjust file names and structure based on your actual GLORIA data format.
    
    Parameters:
    -----------
    base_path : str
        Directory containing GLORIA CSV files
    year : int
        Year of GLORIA data to load
        
    Returns:
    --------
    dict : Dictionary with numpy arrays for V, U, W, Y
    """
    print(f"Loading GLORIA data for year {year}...")
    
    # Load supply table (164*97 x 164*97)
    # File might be named like: 'GLORIA_Supply_2015.csv'
    V = pd.read_csv(f"{base_path}/GLORIA_Supply_{year}.csv", 
                    index_col=0).values
    
    # Load use table (164*97 x 164*97)
    U = pd.read_csv(f"{base_path}/GLORIA_Use_{year}.csv", 
                    index_col=0).values
    
    # Load value added (rows: VA categories, cols: industries)
    W = pd.read_csv(f"{base_path}/GLORIA_ValueAdded_{year}.csv", 
                    index_col=0).values
    
    # Load final demand (rows: products, cols: FD categories)
    Y = pd.read_csv(f"{base_path}/GLORIA_FinalDemand_{year}.csv", 
                    index_col=0).values
    
    print(f"✓ Data loaded successfully")
    print(f"  Supply table: {V.shape}")
    print(f"  Use table: {U.shape}")
    print(f"  Value added: {W.shape}")
    print(f"  Final demand: {Y.shape}")
    
    return {'V': V, 'U': U, 'W': W, 'Y': Y}


# ============================================================================
# EXAMPLE 2: Transform GLORIA MR-SUT to IOT
# ============================================================================

def transform_gloria_to_iot(data: dict, method: str = 'D'):
    """
    Transform GLORIA MR-SUT to IOT using specified method.
    
    Parameters:
    -----------
    data : dict
        Dictionary with 'V', 'U', 'W', 'Y' numpy arrays
    method : str
        Transformation method: 'A', 'B', 'C', or 'D'
        
    Returns:
    --------
    dict : Dictionary with IOT components
    """
    print(f"\\nTransforming MR-SUT to IOT using Method {method}...")
    
    # Initialize transformer
    transformer = GLORIATransformer(
        supply_table=data['V'],
        use_table=data['U'],
        value_added=data['W'],
        final_demand=data['Y'],
        n_regions=164,
        n_sectors=97
    )
    
    # Perform transformation
    result = transformer.transform(method=method)
    
    # Compute technical coefficients
    A = transformer.compute_technical_coefficients(result['Z'], result['x'])
    result['A'] = A
    
    # Compute Leontief inverse
    print("Computing Leontief inverse (I-A)^(-1)...")
    L = transformer.compute_leontief_inverse(A)
    result['L'] = L
    
    print("✓ Transformation complete")
    
    return result, transformer


# ============================================================================
# EXAMPLE 3: Save IOT results to CSV with proper labels
# ============================================================================

def save_iot_to_csv(result: dict, output_path: str, 
                   region_labels: list = None, sector_labels: list = None):
    """
    Save IOT results to CSV files with proper row and column labels.
    
    Parameters:
    -----------
    result : dict
        Dictionary with IOT matrices from transformer
    output_path : str
        Directory to save CSV files
    region_labels : list
        List of 164 region names/codes
    sector_labels : list
        List of 97 sector names/codes
    """
    print(f"\\nSaving IOT results to {output_path}...")
    
    # Generate labels if not provided
    if region_labels is None:
        region_labels = [f"REG{i:03d}" for i in range(1, 165)]
    if sector_labels is None:
        sector_labels = [f"SEC{i:02d}" for i in range(1, 98)]
    
    # Create multi-index for 164 regions x 97 sectors
    index = pd.MultiIndex.from_product(
        [region_labels, sector_labels],
        names=['Region', 'Sector']
    )
    
    # Save intermediate transactions matrix Z
    df_Z = pd.DataFrame(result['Z'], index=index, columns=index)
    df_Z.to_csv(f"{output_path}/IOT_Intermediate_Z.csv")
    print(f"  ✓ Saved Z matrix: {df_Z.shape}")
    
    # Save technical coefficients A
    df_A = pd.DataFrame(result['A'], index=index, columns=index)
    df_A.to_csv(f"{output_path}/IOT_TechnicalCoeff_A.csv")
    print(f"  ✓ Saved A matrix: {df_A.shape}")
    
    # Save Leontief inverse L
    df_L = pd.DataFrame(result['L'], index=index, columns=index)
    df_L.to_csv(f"{output_path}/IOT_LeontiefInverse_L.csv")
    print(f"  ✓ Saved L matrix: {df_L.shape}")
    
    # Save final demand Y
    fd_cols = [f"FD{i}" for i in range(result['Y'].shape[1])]
    df_Y = pd.DataFrame(result['Y'], index=index, columns=fd_cols)
    df_Y.to_csv(f"{output_path}/IOT_FinalDemand_Y.csv")
    print(f"  ✓ Saved Y matrix: {df_Y.shape}")
    
    # Save total output x
    df_x = pd.DataFrame(result['x'], index=index, columns=['TotalOutput'])
    df_x.to_csv(f"{output_path}/IOT_TotalOutput_x.csv")
    print(f"  ✓ Saved x vector: {df_x.shape}")
    
    print("\\n✓ All files saved successfully!")


# ============================================================================
# EXAMPLE 4: Quick analysis of IOT
# ============================================================================

def analyze_iot(result: dict, top_n: int = 10):
    """
    Perform quick analysis of IOT results.
    
    Parameters:
    -----------
    result : dict
        Dictionary with IOT matrices
    top_n : int
        Number of top sectors to display
    """
    print(f"\\n{'='*70}")
    print("IOT ANALYSIS SUMMARY")
    print('='*70)
    
    # Total intermediate consumption
    total_intermediate = result['Z'].sum()
    print(f"\\nTotal intermediate consumption: ${total_intermediate:,.0f}")
    
    # Total final demand
    total_fd = result['Y'].sum()
    print(f"Total final demand: ${total_fd:,.0f}")
    
    # Total output
    total_output = result['x'].sum()
    print(f"Total output: ${total_output:,.0f}")
    
    # Technical coefficients statistics
    A = result['A']
    print(f"\\nTechnical coefficients (A matrix):")
    print(f"  Mean: {A.mean():.4f}")
    print(f"  Median: {np.median(A):.4f}")
    print(f"  Max: {A.max():.4f}")
    print(f"  Min: {A.min():.4f}")
    
    # Leontief inverse statistics
    L = result['L']
    print(f"\\nLeontief inverse (L matrix):")
    print(f"  Mean multiplier: {L.mean():.4f}")
    print(f"  Max multiplier: {L.max():.4f}")
    
    # Top sectors by output
    top_sectors_idx = np.argsort(result['x'])[-top_n:][::-1]
    print(f"\\nTop {top_n} sectors by output:")
    for i, idx in enumerate(top_sectors_idx, 1):
        print(f"  {i}. Sector {idx}: ${result['x'][idx]:,.0f}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for transforming GLORIA MR-SUT to IOT.
    """
    # Step 1: Load GLORIA data
    # Replace with your actual data path
    # data = load_gloria_from_csv(base_path='./gloria_data', year=2015)
    
    # For demonstration, create synthetic GLORIA-sized data
    print("Creating synthetic GLORIA-sized data for demonstration...")
    n_total = 164 * 97  # 15,908
    
    data = {
        'V': np.diag(np.random.uniform(1000, 10000, n_total)),  # Diagonal supply
        'U': np.random.uniform(10, 500, (n_total, n_total)) * 0.3,
        'W': np.random.uniform(100, 1000, (6, n_total)),  # 6 VA categories
        'Y': np.random.uniform(500, 3000, (n_total, 6))   # 6 FD categories
    }
    print(f"✓ Synthetic data created: {n_total} dimensions")
    
    # Step 2: Transform to IOT
    result, transformer = transform_gloria_to_iot(data, method='D')
    
    # Step 3: Analyze results
    analyze_iot(result, top_n=10)
    
    # Step 4: Save results
    # save_iot_to_csv(result, output_path='./gloria_iot_output')
    print("\\n✓ Workflow complete!")
    print("\\nNote: Uncomment save_iot_to_csv() to save results to files")


if __name__ == "__main__":
    main()
'''

# Save the usage example
with open('gloria_usage_example.py', 'w') as f:
    f.write(usage_example)

print("Usage example saved to: gloria_usage_example.py")
print("\nThis file includes:")
print("- Loading GLORIA data from CSV files")
print("- Transforming MR-SUT to IOT")
print("- Saving results with proper labels")
print("- Quick analysis functions")
print("- Complete main workflow")
