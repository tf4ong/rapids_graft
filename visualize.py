import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re



def visualize_overlay_connectivity(img_frame, corr_kern, seed_pixel_coords,valid_pixels,
                                    ax,contours, title_prefix=""):
    """
    Visualizes connectivity by overlaying a semi-transparent heatmap
    on top of the raw image on a SINGLE axis.
    """
    height, width = img_frame.shape
    seed_y, seed_x = seed_pixel_coords

    pixel_index = seed_x * height + seed_y
    
    if pixel_index >= corr_kern.shape[0]:
        print(f"Error: Seed pixel ({seed_y}, {seed_x}) with index {pixel_index} is out of bounds for the kernel.")
        ax.axis('off')
        return None

    # Extract and reshape the connectivity vector (same as before)
    connectivity_vector = corr_kern[pixel_index, :]
    connectivity_map = connectivity_vector.toarray().reshape((height, width), order='F')
    connectivity_map = connectivity_map*valid_pixels

    # --- PLOTTING LOGIC CHANGED ---
    # 1. Plot the base grayscale image
    ax.imshow(img_frame*2, cmap='gray', interpolation='none')
    for x_coords, y_coords in contours:
        x_coords= [int(x) for x in x_coords]
        y_coords= [int(y) for y in y_coords]
        ax.plot(x_coords, y_coords, linewidth=0.5,color='white',alpha=0.3)
    # 2. Plot the connectivity heatmap ON TOP with transparency (alpha)
    # The 'alpha' parameter is key. 0.6 means 60% opaque.
    im = ax.imshow(connectivity_map, cmap='hot', interpolation='none', alpha=0.65)
    
    # 3. Plot the seed marker on top of everything
    ax.plot(seed_x, seed_y, 'cx', markersize=5, markeredgewidth=0.75) # Cyan 'c' for better visibility
    
    ax.set_title(f'{title_prefix}')
    ax.axis('off')
    
    return im

def parse_coords(coord_string: str) -> tuple:
    """
    Parses a string like '(50.54, 20.17)' into a tuple of integers (y, x).
    """
    try:
        # Find all numbers in the string
        numbers = [float(num) for num in re.findall(r'-?\d+\.?\d*', coord_string)]
        if len(numbers) != 2:
            raise ValueError("String does not contain two numbers.")
        # Convert to integers for pixel coordinates, assuming (y, x) order
        return (int(numbers[1]), int(numbers[0]))
    except (ValueError, TypeError):
        print(f"Warning: Could not parse coordinate string: {coord_string}. Skipping.")
        return None

# Assumes you have S_final with shape (height, width, n_atoms)
# S_final = your_result_from_graft_gpu

def visualize_spatial_maps(S_final, n_cols=8, cmap='viridis', save_path='spatial_maps.png'):
    """
    Visualize spatial activity maps in an organized grid.
    
    Args:
        S_final (np.ndarray): Spatial maps array of shape (height, width, n_atoms)
        n_cols (int): Number of columns in the grid layout
        cmap (str): Colormap to use ('viridis', 'hot', 'plasma', 'inferno', etc.)
        save_path (str): Path to save the figure
    """
    height, width, n_atoms = S_final.shape
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(n_atoms / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    axes = axes.ravel()  # Flatten to 1D array for easy indexing
    
    fig.suptitle(f'Spatial Activity Maps - {n_atoms} Atoms', 
                 fontsize=16, fontweight='bold')
    
    # Plot each spatial map
    for i in range(n_atoms):
        s_map = S_final[:, :, i]
        
        # Display the map
        im = axes[i].imshow(s_map, cmap=cmap, interpolation='nearest')
        
        # Title with statistics
        axes[i].set_title(f'Atom {i+1}\nmax={s_map.max():.2e}\nnnz={np.count_nonzero(s_map)}', 
                         fontsize=9)
        axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide any unused subplots
    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def visualize_top_atoms(S_final, n_top=16, n_cols=4, cmap='viridis', save_path=None):
    """
    Visualize only the top N most active spatial maps.
    
    Args:
        S_final (np.ndarray): Spatial maps array of shape (height, width, n_atoms)
        n_top (int): Number of top atoms to display
        n_cols (int): Number of columns in the grid layout
        cmap (str): Colormap to use
        save_path (str): Path to save the figure
    """
    height, width, n_atoms = S_final.shape
    
    # Calculate activity (sum of absolute values) for each atom
    activity = np.array([np.sum(np.abs(S_final[:, :, i])) for i in range(n_atoms)])
    
    # Get indices of top N most active atoms
    top_indices = np.argsort(activity)[::-1][:n_top]
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(n_top / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.ravel()
    
    fig.suptitle(f'Top {n_top} Most Active Spatial Maps', 
                 fontsize=16, fontweight='bold')
    
    # Plot top atoms
    for idx, atom_idx in enumerate(top_indices):
        s_map = S_final[:, :, atom_idx]
        
        im = axes[idx].imshow(s_map, cmap=cmap, interpolation='nearest')
        axes[idx].set_title(f'Atom {atom_idx+1} (Rank {idx+1})\n'
                           f'activity={activity[atom_idx]:.2e}', 
                           fontsize=10)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_top, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    
    return fig, top_indices


def create_composite_maps(S_final):
    """
    Create summary composite visualizations.
    
    Args:
        S_final (np.ndarray): Spatial maps array of shape (height, width, n_atoms)
        save_path (str): Path to save the figure
    """
    height, width, n_atoms = S_final.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Sum of all activity
    composite_sum = np.sum(S_final, axis=2)
    im1 = axes[0, 0].imshow(composite_sum, cmap='hot', interpolation='nearest')
    axes[0, 0].set_title('Sum of All Activity', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Max projection
    composite_max = np.max(S_final, axis=2)
    im2 = axes[0, 1].imshow(composite_max, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title('Max Projection Across Atoms', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Number of active atoms per pixel
    active_count = np.sum(S_final > 0.01 * np.max(S_final), axis=2)
    im3 = axes[1, 0].imshow(active_count, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title('Number of Active Atoms per Pixel', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. Dominant atom per pixel
    dominant_atom = np.argmax(S_final, axis=2)
    im4 = axes[1, 1].imshow(dominant_atom, cmap='tab20', interpolation='nearest')
    axes[1, 1].set_title('Dominant Atom ID per Pixel', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()

    
    return fig


def plot_reconstruction_comparison(results, pixel_idx=None, time_range=None):
    """
    Plot comparison of original data vs reconstruction for visual inspection.
    
    Args:
        results (dict): Output from calculate_variance_explained()
        pixel_idx (int): Which pixel to plot (random if None)
        time_range (tuple): (start, end) time range to plot (all if None)
    """
    data_matrix = results['data_matrix']
    data_reconstructed = results['data_reconstructed']
    
    if pixel_idx is None:
        # Pick a pixel with high activity
        pixel_variance = np.var(data_matrix, axis=1)
        pixel_idx = np.argmax(pixel_variance)
    
    if time_range is None:
        time_range = (0, min(500, data_matrix.shape[1]))  # First 500 frames
    
    t_start, t_end = time_range
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    
    time_axis = np.arange(t_start, t_end)
    
    # Original data
    axes[0].plot(time_axis, data_matrix[pixel_idx, t_start:t_end], 'b-', linewidth=1, label='Original')
    axes[0].set_ylabel('Signal', fontweight='bold')
    axes[0].set_title(f'Original Data - Pixel {pixel_idx}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Reconstructed data
    axes[1].plot(time_axis, data_reconstructed[pixel_idx, t_start:t_end], 'r-', linewidth=1, label='Reconstructed')
    axes[1].set_ylabel('Signal', fontweight='bold')
    axes[1].set_title('Reconstructed Data', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Overlay
    axes[2].plot(time_axis, data_matrix[pixel_idx, t_start:t_end], 'b-', linewidth=1.5, alpha=0.7, label='Original')
    axes[2].plot(time_axis, data_reconstructed[pixel_idx, t_start:t_end], 'r--', linewidth=1.5, alpha=0.7, label='Reconstructed')
    axes[2].set_xlabel('Time (frames)', fontweight='bold')
    axes[2].set_ylabel('Signal', fontweight='bold')
    axes[2].set_title('Overlay Comparison', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    return fig


def plot_variance_explained(results, n_top=20):
    """
    Create visualization of variance explained.
    
    Args:
        results (dict): Output from calculate_variance_explained()
        n_top (int): Number of top atoms to show in detail
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall variance explained
    ax = axes[0, 0]
    categories = ['Explained\nby Model', 'Residual\n(Unexplained)']
    values = [results['variance_explained_pct'], 
              100 - results['variance_explained_pct']]
    colors = ['#2ecc71', '#e74c3c']
    
    ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Total Variance Explained\nR² = {results["r_squared"]:.4f} (corr method: {results["r_squared_corr"]:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + 2, f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')
    
    # 2. Variance explained per atom (top N)
    ax = axes[0, 1]
    sorted_idx = results['sorted_atom_indices'][:n_top]
    var_pct = results['atom_variance_pct'][sorted_idx]
    
    bars = ax.barh(range(len(sorted_idx)), var_pct, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([f'Atom {i+1}' for i in sorted_idx], fontsize=9)
    ax.set_xlabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_top} Atoms by Variance Explained', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, var_pct)):
        if val > 1:  # Only label if > 1%
            ax.text(val + 0.1, i, f'{val:.2f}%', va='center', fontsize=8)
    
    # 3. Cumulative variance explained
    ax = axes[1, 0]
    n_atoms = len(results['cumulative_variance_pct'])
    ax.plot(range(1, n_atoms + 1), results['cumulative_variance_pct'], 
            'o-', linewidth=2, markersize=4, color='darkgreen')
    ax.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Number of Atoms', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, n_atoms + 1])
    ax.set_ylim([0, 105])
    
    # Find how many atoms for thresholds
    idx_50 = np.argmax(results['cumulative_variance_pct'] >= 50) + 1 if np.any(results['cumulative_variance_pct'] >= 50) else 0
    idx_90 = np.argmax(results['cumulative_variance_pct'] >= 90) + 1 if np.any(results['cumulative_variance_pct'] >= 90) else 0
    idx_95 = np.argmax(results['cumulative_variance_pct'] >= 95) + 1 if np.any(results['cumulative_variance_pct'] >= 95) else 0
    
    if idx_50 > 0:
        ax.axvline(x=idx_50, color='blue', linestyle=':', alpha=0.5)
        ax.text(idx_50, 5, f'{idx_50} atoms', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if idx_90 > 0:
        ax.axvline(x=idx_90, color='red', linestyle=':', alpha=0.5)
        ax.text(idx_90, 92, f'{idx_90} atoms', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Distribution of atom contributions
    ax = axes[1, 1]
    all_var_pct = results['atom_variance_pct'][results['sorted_atom_indices']]
    ax.semilogy(range(1, len(all_var_pct) + 1), all_var_pct, 'o-', 
                linewidth=2, markersize=5, color='purple')
    ax.set_xlabel('Atom Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Explained (%) [log scale]', fontsize=12, fontweight='bold')
    ax.set_title('Atom Importance Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, len(all_var_pct) + 1])
    
    plt.tight_layout()
    plt.savefig('variance_explained_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def print_variance_summary(results, n_top=10):
    """
    Print a text summary of variance explained.
    
    Args:
        results (dict): Output from calculate_variance_explained()
        n_top (int): Number of top atoms to show
    """
    print("\n" + "="*70)
    print("VARIANCE EXPLAINED ANALYSIS")
    print("="*70)
    
    if results.get('scale_correction_applied', False):
        print(f"\n⚠️  Scale correction was applied (ratio: {results['scale_ratio']:.4f})")
        print("   S coefficients were automatically adjusted for proper reconstruction.")
    
    print(f"\nR² (Coefficient of Determination): {results['r_squared']:.4f}")
    print(f"R² (Correlation method): {results['r_squared_corr']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    
    if results['r_squared'] < 0:
        print("\n⚠️  WARNING: Negative R² detected!")
        print("   This means the model is worse than just using the mean.")
        print("   Possible causes:")
        print("   1. Lambda too high (over-sparsifying)")
        print("   2. Not enough iterations")
        print("   3. Wrong data scaling")
        print("   4. Dictionary not converged")
    elif results['r_squared'] < 0.3:
        print("\n⚠️  WARNING: Low R² detected!")
        print("   The model is not fitting the data well.")
    elif results['r_squared'] > 0.7:
        print("\n✓ Excellent fit!")
    elif results['r_squared'] > 0.5:
        print("\n✓ Good fit!")
    
    print(f"\nSum of Squares:")
    print(f"  Total SS: {results['ss_total']:.4e}")
    print(f"  Explained SS: {results['ss_explained']:.4e} ({100*results['ss_explained']/results['ss_total']:.2f}%)")
    print(f"  Residual SS: {results['ss_residual']:.4e} ({100*results['ss_residual']/results['ss_total']:.2f}%)")
    
    print(f"\n{'-'*70}")
    print(f"TOP {n_top} ATOMS BY CONTRIBUTION:")
    print(f"{'-'*70}")
    sorted_idx = results['sorted_atom_indices'][:n_top]
    for rank, idx in enumerate(sorted_idx, 1):
        var_pct = results['atom_variance_pct'][idx]
        print(f"  Rank {rank:2d}: Atom {idx+1:3d} - {var_pct:6.2f}% contribution")
    
    # Find atoms needed for thresholds (only if R² is positive)
    if results['r_squared'] > 0:
        cumvar = results['cumulative_variance_pct']
        idx_50 = np.argmax(cumvar >= 50) + 1 if np.any(cumvar >= 50) else None
        idx_90 = np.argmax(cumvar >= 90) + 1 if np.any(cumvar >= 90) else None
        idx_95 = np.argmax(cumvar >= 95) + 1 if np.any(cumvar >= 95) else None
        
        print(f"\n{'-'*70}")
        print("ATOMS NEEDED FOR VARIANCE THRESHOLDS:")
        print(f"{'-'*70}")
        if idx_50:
            print(f"  50% variance: {idx_50} atoms")
        if idx_90:
            print(f"  90% variance: {idx_90} atoms")
        if idx_95:
            print(f"  95% variance: {idx_95} atoms")
    
    print("="*70)


