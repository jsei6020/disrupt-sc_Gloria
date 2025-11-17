
"""
GLORIA MR-SUT to IOT Transformation
====================================
Transforms a Multi-Regional Supply-Use Table from GLORIA database 
into an Input-Output Table using numpy and pandas.

GLORIA Structure:
- 164 regions
- 97 sectors (industry = commodity)
- Homogeneous MR-SUT with diagonal supply table

Methods Available:
- Method A: Product-by-product IOT (product technology assumption)
- Method B: Product-by-product IOT (industry technology assumption)
- Method C: Industry-by-industry IOT (fixed industry sales structure)
- Method D: Industry-by-industry IOT (fixed product sales structure)

Author: Based on Eurostat Manual and UN Handbook
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

class GLORIATransformer:
    """
    Transform GLORIA MR-SUT into IOT using various transformation methods.
    """

    def __init__(self, supply_table: np.ndarray, use_table: np.ndarray, 
                 value_added: np.ndarray, final_demand: np.ndarray,
                 n_regions: int = 164, n_sectors: int = 97):
        """
        Initialize the transformer with GLORIA MR-SUT components.

        Parameters:
        -----------
        supply_table : np.ndarray
            Supply table V (industries x products), shape: (164*97, 164*97)
            For GLORIA, this should be diagonal as supply is homogeneous
        use_table : np.ndarray  
            Use table U (products x industries), shape: (164*97, 164*97)
            Intermediate consumption of products by industries
        value_added : np.ndarray
            Value added by industry, shape: (n_va_categories, 164*97)
        final_demand : np.ndarray
            Final demand by product, shape: (164*97, n_fd_categories)
        n_regions : int
            Number of regions (default: 164 for GLORIA)
        n_sectors : int
            Number of sectors (default: 97 for GLORIA)
        """
        self.V = supply_table  # Supply table (Make matrix)
        self.U = use_table     # Use table
        self.W = value_added   # Value added
        self.Y = final_demand  # Final demand

        self.n_regions = n_regions
        self.n_sectors = n_sectors
        self.n_total = n_regions * n_sectors  # Total dimensions: 164*97 = 15,908

        # Calculate total outputs
        self.g = self.V.sum(axis=1)  # Industry output (row sums of V)
        self.x = self.V.sum(axis=0)  # Product output (column sums of V)

        # Validate dimensions
        self._validate_dimensions()

    def _validate_dimensions(self):
        """Validate that all matrices have correct dimensions."""
        assert self.V.shape == (self.n_total, self.n_total),             f"Supply table shape mismatch: {self.V.shape} != ({self.n_total}, {self.n_total})"
        assert self.U.shape == (self.n_total, self.n_total),             f"Use table shape mismatch: {self.U.shape} != ({self.n_total}, {self.n_total})"
        assert self.Y.shape[0] == self.n_total,             f"Final demand rows mismatch: {self.Y.shape[0]} != {self.n_total}"
        assert self.W.shape[1] == self.n_total,             f"Value added columns mismatch: {self.W.shape[1]} != {self.n_total}"

    def _safe_diag_inverse(self, vector: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Create diagonal matrix with inverse of vector, handling zeros safely.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector to invert
        epsilon : float
            Small value to avoid division by zero

        Returns:
        --------
        np.ndarray : Diagonal matrix with inverted values
        """
        vector_safe = np.where(np.abs(vector) < epsilon, epsilon, vector)
        return np.diag(1.0 / vector_safe)

    def transform_method_a(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method A: Product-by-product IOT based on PRODUCT TECHNOLOGY assumption.

        Assumption: Each product has a unique input structure, regardless of 
        the industry where it is produced.

        Returns:
        --------
        Z : np.ndarray
            Intermediate transaction matrix (product x product)
        Y_out : np.ndarray
            Final demand matrix
        """
        print("Applying Method A: Product-by-product (Product Technology)")

        # Market share matrix D: share of each industry in total product output
        # D = V * diag(x)^(-1)
        # Dimensions: (n_industries, n_products) * (n_products, n_products)
        x_inv = self._safe_diag_inverse(self.x)
        D = self.V @ x_inv

        # Product-by-product intermediate matrix
        # Z = U * D
        # Dimensions: (n_products, n_industries) * (n_industries, n_products)
        Z = self.U @ D

        return Z, self.Y

    def transform_method_b(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method B: Product-by-product IOT based on INDUSTRY TECHNOLOGY assumption.

        Assumption: Each industry has a unique input structure, regardless of 
        its product mix.

        Returns:
        --------
        Z : np.ndarray
            Intermediate transaction matrix (product x product)
        Y_out : np.ndarray
            Final demand matrix
        """
        print("Applying Method B: Product-by-product (Industry Technology)")

        # Product mix matrix C: share of each product in industry output
        # C = V^T * diag(g)^(-1)
        # Dimensions: (n_products, n_industries) * (n_industries, n_industries)
        g_inv = self._safe_diag_inverse(self.g)
        C = self.V.T @ g_inv

        # Transformation matrix T
        # T = C^T = (V^T * diag(g)^(-1))^T
        T = C.T

        # Product-by-product intermediate matrix
        # Z = U * T
        # Dimensions: (n_products, n_industries) * (n_industries, n_products)
        Z = self.U @ T

        return Z, self.Y

    def transform_method_c(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method C: Industry-by-industry IOT based on FIXED INDUSTRY SALES STRUCTURE.

        Assumption: Each industry has its own specific sales structure, 
        regardless of its product mix.

        Returns:
        --------
        Z : np.ndarray
            Intermediate transaction matrix (industry x industry)
        Y_out : np.ndarray
            Final demand matrix
        """
        print("Applying Method C: Industry-by-industry (Fixed Industry Sales)")

        # Product mix matrix C: share of each product in industry output
        # C = V^T * diag(g)^(-1)
        g_inv = self._safe_diag_inverse(self.g)
        C = self.V.T @ g_inv

        # Industry-by-industry intermediate matrix
        # Z = C^(-1) * U
        # Need to invert C (this may produce negative values)
        # Using pseudo-inverse for numerical stability
        try:
            C_inv = np.linalg.pinv(C)
        except:
            C_inv = np.linalg.inv(C + np.eye(C.shape[0]) * 1e-10)

        Z = C_inv @ self.U

        # Transform final demand from product to industry basis
        Y_ind = C_inv @ self.Y

        return Z, Y_ind

    def transform_method_d(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method D: Industry-by-industry IOT based on FIXED PRODUCT SALES STRUCTURE.

        Assumption: Each product has its own specific sales structure, 
        regardless of the industry where it is produced.

        This is the most commonly used method and does not produce negative values.

        Returns:
        --------
        Z : np.ndarray
            Intermediate transaction matrix (industry x industry)
        Y_out : np.ndarray
            Final demand matrix
        """
        print("Applying Method D: Industry-by-industry (Fixed Product Sales)")

        # Market share matrix D: share of each industry in total product output
        # D = V * diag(x)^(-1)
        x_inv = self._safe_diag_inverse(self.x)
        D = self.V @ x_inv

        # Industry-by-industry intermediate matrix
        # Z = D * U
        # Dimensions: (n_industries, n_products) * (n_products, n_industries)
        Z = D @ self.U

        # Transform final demand from product to industry basis
        Y_ind = D @ self.Y

        return Z, Y_ind

    def transform(self, method: str = 'D') -> Dict[str, np.ndarray]:
        """
        Transform MR-SUT to IOT using specified method.

        Parameters:
        -----------
        method : str
            Transformation method: 'A', 'B', 'C', or 'D'
            Default: 'D' (most stable, no negative values)

        Returns:
        --------
        dict : Dictionary containing:
            - 'Z': Intermediate transactions matrix
            - 'Y': Final demand matrix
            - 'W': Value added matrix
            - 'x_out': Total output vector
        """
        method = method.upper()

        if method == 'A':
            Z, Y_out = self.transform_method_a()
            x_out = self.x  # Product output
        elif method == 'B':
            Z, Y_out = self.transform_method_b()
            x_out = self.x  # Product output
        elif method == 'C':
            Z, Y_out = self.transform_method_c()
            x_out = self.g  # Industry output
        elif method == 'D':
            Z, Y_out = self.transform_method_d()
            x_out = self.g  # Industry output
        else:
            raise ValueError(f"Unknown method: {method}. Use 'A', 'B', 'C', or 'D'")

        print(f"Transformation complete. IOT shape: {Z.shape}")
        print(f"Final demand shape: {Y_out.shape}")

        return {
            'Z': Z,           # Intermediate transactions
            'Y': Y_out,       # Final demand
            'W': self.W,      # Value added (unchanged)
            'x': x_out        # Total output
        }

    def compute_technical_coefficients(self, Z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Compute technical coefficients matrix A from transaction matrix Z.

        A_ij = Z_ij / x_j (input per unit of output)

        Parameters:
        -----------
        Z : np.ndarray
            Intermediate transaction matrix
        x : np.ndarray
            Total output vector

        Returns:
        --------
        A : np.ndarray
            Technical coefficients matrix
        """
        x_inv = self._safe_diag_inverse(x)
        A = Z @ x_inv
        return A

    def compute_leontief_inverse(self, A: np.ndarray) -> np.ndarray:
        """
        Compute Leontief inverse L = (I - A)^(-1).

        Parameters:
        -----------
        A : np.ndarray
            Technical coefficients matrix

        Returns:
        --------
        L : np.ndarray
            Leontief inverse matrix
        """
        I = np.eye(A.shape[0])
        L = np.linalg.inv(I - A)
        return L


def example_usage_small_scale():
    """
    Example with small test data to demonstrate usage.
    For actual GLORIA data, load from files.
    """
    print("\n" + "="*70)
    print("EXAMPLE: Small Scale Test (6 regions x 3 sectors = 18 dimensions)")
    print("="*70 + "\n")

    n_regions = 6
    n_sectors = 3
    n_total = n_regions * n_sectors  # 18

    # Create synthetic test data
    np.random.seed(42)

    # GLORIA has diagonal supply table (homogeneous)
    V = np.diag(np.random.uniform(100, 1000, n_total))

    # Use table (products x industries)
    U = np.random.uniform(10, 100, (n_total, n_total))
    # Make it realistic: inputs < outputs
    U = U * 0.3

    # Value added (2 categories x industries)
    W = np.random.uniform(50, 200, (2, n_total))

    # Final demand (products x 3 categories)
    Y = np.random.uniform(100, 500, (n_total, 3))

    # Initialize transformer
    transformer = GLORIATransformer(V, U, W, Y, n_regions, n_sectors)

    # Test all methods
    for method in ['A', 'B', 'C', 'D']:
        print(f"\n{'='*70}")
        result = transformer.transform(method=method)

        # Compute technical coefficients
        A = transformer.compute_technical_coefficients(result['Z'], result['x'])
        print(f"Technical coefficients A shape: {A.shape}")
        print(f"Max coefficient: {A.max():.4f}")

        # Check for negative values
        neg_count = np.sum(result['Z'] < 0)
        print(f"Negative values in Z: {neg_count}")

    return transformer


def load_gloria_data_template(filepath: str) -> Dict[str, np.ndarray]:
    """
    Template function to load GLORIA data from files.

    Adjust this function based on your actual GLORIA data format.
    GLORIA data might be in CSV, pickle, or other formats.

    Parameters:
    -----------
    filepath : str
        Path to GLORIA MR-SUT data file

    Returns:
    --------
    dict : Dictionary with 'V', 'U', 'W', 'Y' matrices
    """
    # Example for CSV format (adjust based on actual format)
    # df = pd.read_csv(filepath)

    # Example for pickle format
    # import pickle
    # with open(filepath, 'rb') as f:
    #     data = pickle.load(f)

    # Return structure:
    return {
        'V': None,  # Supply table (164*97 x 164*97)
        'U': None,  # Use table (164*97 x 164*97)
        'W': None,  # Value added (n_va x 164*97)
        'Y': None   # Final demand (164*97 x n_fd)
    }


# Run example
if __name__ == "__main__":
    transformer = example_usage_small_scale()

    print("\n" + "="*70)
    print("CODE READY FOR GLORIA DATA")
    print("="*70)
    print("\nTo use with actual GLORIA data (164 regions x 97 sectors):")
    print("1. Load your GLORIA MR-SUT components (V, U, W, Y)")
    print("2. Initialize: transformer = GLORIATransformer(V, U, W, Y)")
    print("3. Transform: result = transformer.transform(method='D')")
    print("4. Access results: Z = result['Z'], Y = result['Y']")
