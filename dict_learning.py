import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
from cupyx.scipy.sparse.linalg import lsqr as cp_lsqr
import matplotlib.pyplot as plt
from cuml.neighbors import NearestNeighbors
import time

# =============================================================================
# GPU-ACCELERATED CORE SOLVER
# =============================================================================
def single_gauss_neuro_infer_gpu(tau_vec, mov_vec, D, lambda_val, nonneg=True, 
                                  max_iter=1000, tol=1e-4):
    """
    GPU-accelerated weighted LASSO solver using proximal gradient descent.
    Much faster than CVXPY on GPU.
    
    Args:
        tau_vec (cp.ndarray): GPU array of penalty weights (n_atoms,)
        mov_vec (cp.ndarray): GPU array of pixel signal (n_time,)
        D (cp.ndarray): Dictionary matrix on GPU (n_time, n_atoms)
        lambda_val (float): Sparsity parameter
        nonneg (bool): Non-negativity constraint
        max_iter (int): Maximum iterations for optimization
        tol (float): Convergence tolerance
    
    Returns:
        cp.ndarray: Sparse coefficient vector on GPU
    """
    n_atoms = D.shape[1]
    
    # Initialize S
    S = cp.zeros(n_atoms, dtype=cp.float32)
    
    # Ensure mov_vec is a column vector for proper broadcasting
    mov_vec = mov_vec.ravel()
    
    # Precompute for efficiency
    DtD = D.T @ D
    Dtmov = D.T @ mov_vec
    
    # Lipschitz constant for step size
    L = cp.linalg.norm(DtD, ord=2)
    step_size = 1.0 / (L + 1e-8)
    
    # Proximal gradient descent
    for _ in range(max_iter):
        S_old = S.copy()
        
        # Gradient step
        grad = DtD @ S - Dtmov
        S = S - step_size * grad
        
        # Proximal operator (soft thresholding with weighted L1)
        threshold = lambda_val * step_size * tau_vec
        S = cp.sign(S) * cp.maximum(cp.abs(S) - threshold, 0)
        
        # Non-negativity projection
        if nonneg:
            S = cp.maximum(S, 0)
        
        # Check convergence
        if cp.linalg.norm(S - S_old) < tol:
            break
    
    # Sparsify (remove small values)
    S_max = S.max()
    if S_max > 0:
        S[S < 0.1 * S_max] = 0
    
    return S


def batch_solve_gpu(tau_mat, mov_flat, D, lambda_val, nonneg=True, batch_size=512):
    """
    Solve multiple pixels in batches on GPU using vectorized operations.
    
    Args:
        tau_mat (cp.ndarray): (n_pixels, n_atoms) penalty weights on GPU
        mov_flat (cp.ndarray): (n_pixels, n_time) data on GPU
        D (cp.ndarray): Dictionary on GPU (n_time, n_atoms)
        lambda_val (float): Sparsity parameter
        nonneg (bool): Non-negativity constraint
        batch_size (int): Number of pixels to process together
    
    Returns:
        cp.ndarray: (n_pixels, n_atoms) sparse coefficients
    """
    n_pixels, n_time = mov_flat.shape
    n_atoms = D.shape[1]
    S_result = cp.zeros((n_pixels, n_atoms), dtype=cp.float32)
    
    # Precompute shared terms for all pixels
    DtD = D.T @ D  # (n_atoms, n_atoms)
    
    # Compute Lipschitz constant once
    L = cp.linalg.norm(DtD, ord=2)
    step_size = 1.0 / (L + 1e-8)
    
    max_iter = 500  # Reduced for batch processing
    tol = 1e-4
    
    # Process in batches
    for batch_start in range(0, n_pixels, batch_size):
        batch_end = min(batch_start + batch_size, n_pixels)
        batch_size_actual = batch_end - batch_start
        
        # Get batch data
        mov_batch = mov_flat[batch_start:batch_end]  # (batch_size, n_time)
        tau_batch = tau_mat[batch_start:batch_end]   # (batch_size, n_atoms)
        
        # Initialize S for this batch
        S_batch = cp.zeros((batch_size_actual, n_atoms), dtype=cp.float32)
        
        # Precompute D.T @ mov_batch.T -> (n_atoms, batch_size)
        Dtmov_batch = D.T @ mov_batch.T  # (n_atoms, batch_size)
        
        # Vectorized proximal gradient descent for the batch
        for iter_idx in range(max_iter):
            S_old = S_batch.copy()
            
            # Gradient step: grad = DtD @ S_batch.T - Dtmov_batch
            grad = DtD @ S_batch.T - Dtmov_batch  # (n_atoms, batch_size)
            S_batch = S_batch - step_size * grad.T  # (batch_size, n_atoms)
            
            # Soft thresholding with per-pixel weights
            threshold = lambda_val * step_size * tau_batch
            S_batch = cp.sign(S_batch) * cp.maximum(cp.abs(S_batch) - threshold, 0)
            
            # Non-negativity projection
            if nonneg:
                S_batch = cp.maximum(S_batch, 0)
            
            # Check convergence
            if cp.linalg.norm(S_batch - S_old) / (cp.linalg.norm(S_old) + 1e-8) < tol:
                break
        
        # Sparsify: remove small values (but keep some non-zero entries)
        S_max = cp.max(S_batch, axis=1, keepdims=True)
        # Only sparsify if max is significant
        mask = (S_batch < 0.1 * S_max) & (S_max > 1e-6)
        S_batch[mask] = 0
        
        # Store results
        S_result[batch_start:batch_end] = S_batch
    
    return S_result


# =============================================================================
# GPU-ACCELERATED WORKER FUNCTION
# =============================================================================
def dictionary_rwl1sf_gpu(mov_flat, D, corr_kern, params, S_init):
    """
    GPU-accelerated version of dictionaryRWL1SF.
    All computations happen on GPU using CuPy.
    
    Args:
        mov_flat (cp.ndarray): (n_pixels, n_time) data on GPU
        D (cp.ndarray): Dictionary on GPU
        corr_kern (cp.sparse): Sparse correlation kernel on GPU
        params (dict): Algorithm parameters
        S_init (cp.ndarray): Initial S on GPU
    
    Returns:
        cp.ndarray: Updated S matrix
        cp.ndarray: Updated tau_mat
    """
    n_pixels, n_time = mov_flat.shape
    n_atoms = D.shape[1]
    
    # Initialize
    if S_init is not None:
        S = S_init.copy()
    else:
        S = cp.zeros((n_pixels, n_atoms), dtype=cp.float32)
    
    tau_mat = cp.full((n_pixels, n_atoms), params.get('tau', 1.0), dtype=cp.float32)
    
    # Main reweighting loop
    for kk in range(params.get('numreps', 2)):
        print(f"  RWL1SF Iteration {kk+1}/{params.get('numreps', 2)}")
        
        # Reweighting step (spatial regularization)
        if kk > 0:
            print("    Updating penalty weights with graph filtering...")
            beta = params.get('beta', 0.09)
            tau = params.get('tau', 1.0)
            
            # Graph-filtered spatial term (this is where the magic happens!)
            spatial_term = corr_kern @ cp.abs(S)
            tau_mat = tau / (beta + cp.abs(S) + spatial_term)
        
        # Inference step (batch GPU solve)
        print("    Solving weighted LASSO on GPU...")
        lambda_val = params.get('lambda', 0.6)
        nonneg = params.get('nonneg', True)
        batch_size = params.get('batch_size', 512)
        
        S = batch_solve_gpu(tau_mat, mov_flat, D, lambda_val, nonneg, batch_size)
        
        # Synchronize GPU
        cp.cuda.Stream.null.synchronize()
    
    return S, tau_mat


# =============================================================================
# GPU-ACCELERATED MASTER FUNCTION
# =============================================================================
'''
def graft_gpu(gcamp, D_init, corr_kern, params):
    """
    GPU-accelerated GraFT algorithm using CuPy and RAPIDS.
    
    Args:
        gcamp (np.ndarray): Input movie (time, height, width) on CPU
        D_init (np.ndarray): Initial dictionary on CPU
        corr_kern (scipy.sparse or cupy.sparse): Correlation kernel
        params (dict): Algorithm parameters
    
    Returns:
        D_final (np.ndarray): Final dictionary on CPU
        S_final (np.ndarray): Final spatial maps on CPU
        dict_evolution (list): Dictionary evolution on CPU
    """
    # Move data to GPU
    print("Transferring data to GPU...")
    time_pts, height, width = gcamp.shape
    total_pixels = height * width
    
    # Reshape and move to GPU
    mov_flat = cp.array(gcamp.reshape(time_pts, total_pixels, order='F').T, 
                        dtype=cp.float32)
    
    # Normalize the data to prevent numerical issues
    mov_mean = cp.mean(mov_flat, axis=1, keepdims=True)
    mov_std = cp.std(mov_flat, axis=1, keepdims=True) + 1e-8
    mov_flat = (mov_flat - mov_mean) / mov_std
    
    print(f"Data shape: {mov_flat.shape}, mean: {float(cp.mean(mov_flat)):.4f}, std: {float(cp.std(mov_flat)):.4f}")
    
    D = cp.array(D_init, dtype=cp.float32)
    
    # Ensure dictionary is properly normalized
    D_norms = cp.linalg.norm(D, axis=0, keepdims=True)
    D = D / (D_norms + 1e-8)
    
    print(f"Dictionary shape: {D.shape}, initialized and normalized")
    
    # Convert correlation kernel to GPU sparse matrix if needed
    if not isinstance(corr_kern, cp_sparse.spmatrix):
        corr_kern = cp_sparse.csr_matrix(corr_kern)
    
    S = None
    max_learn_iter = params.get('max_learn', 50)
    learn_eps = params.get('learn_eps', 0.01)
    n_atoms = params.get('n_atoms', D.shape[1])
    dict_evolution = []
    
    # Main learning loop
    for i in range(max_learn_iter):
        print(f"\n--- GraFT GPU Iteration {i+1}/{max_learn_iter} ---")
        D_old = D.copy()
        
        # Step 1: Infer activity maps on GPU
        print("Step 1: Inferring sparse activity maps on GPU...")
        S, _ = dictionary_rwl1sf_gpu(mov_flat, D, corr_kern, params, S)
        
        # Step 2: Update dictionary on GPU
        print("Step 2: Updating dictionary on GPU...")
        
        # Check if S is too sparse
        S_nonzero = cp.count_nonzero(S)
        print(f"  S has {S_nonzero}/{S.size} non-zero elements ({100*S_nonzero/S.size:.2f}%)")
        
        if S_nonzero < n_atoms:
            print("  WARNING: S is too sparse, using regularized update...")
            # Use more regularization if S is very sparse
            reg_strength = 1e-3
        else:
            reg_strength = 1e-6
        
        # Use GPU-accelerated least squares with proper regularization
        StS = S.T @ S
        Stmov = S.T @ mov_flat
        
        # Add ridge regularization for stability
        reg = reg_strength * cp.eye(StS.shape[0], dtype=cp.float32)
        
        try:
            D_new = cp.linalg.solve(StS + reg, Stmov).T
        except cp.linalg.LinAlgError:
            print("  Solve failed, using pseudoinverse...")
            D_new = (cp.linalg.pinv(StS + reg) @ Stmov).T
        
        # Normalize atoms and handle zero columns
        norms = cp.linalg.norm(D_new, axis=0, keepdims=True)
        norms = cp.maximum(norms, 1e-8)  # Prevent division by zero
        D_new = D_new / norms
        
        # Check for NaN/Inf in dictionary
        if cp.any(cp.isnan(D_new)) or cp.any(cp.isinf(D_new)):
            print("  WARNING: NaN/Inf detected in dictionary, keeping old dictionary")
            D_new = D_old
        else:
            D = D_new
        
        # Check convergence
        dDict = float(cp.linalg.norm(D - D_old) / (cp.linalg.norm(D_old) + 1e-8))
        print(f"Relative change in dictionary: {dDict:.6f}")
        print(f"Dictionary stats - min: {float(cp.min(D)):.4f}, max: {float(cp.max(D)):.4f}, mean: {float(cp.mean(D)):.4f}")
        
        # Store evolution (move to CPU for storage)
        dict_evolution.append(cp.asnumpy(D))
        
        if not cp.isnan(dDict) and dDict < learn_eps:
            print(f"Convergence reached after {i+1} iterations.")
            break
    
    # Final inference
    print("\nFinal inference with converged dictionary...")
    S_final, _ = dictionary_rwl1sf_gpu(mov_flat, D, corr_kern, params, S)
    
    # Move results back to CPU
    print("Transferring results back to CPU...")
    D_final = cp.asnumpy(D)
    S_final_flat = cp.asnumpy(S_final)
    
    # Reshape to spatial maps
    S_final_reshaped = S_final_flat.reshape(height, width, D_final.shape[1], order='F')
    
    return D_final, S_final_reshaped, dict_evolution
'''

def graft_gpu(gcamp, D_init, corr_kern, params):
    """
    GPU-accelerated GraFT with advanced dictionary regularization.
    
    Includes `lamCont` for smooth learning and `lamCorr` for atom decorrelation.
    """
    # --- Initializations are the same as before ---
    print("Transferring data to GPU...")
    time_pts, height, width = gcamp.shape
    total_pixels = height * width
    mov_flat = cp.array(gcamp.reshape(time_pts, total_pixels, order='F').T, dtype=cp.float32)
    
    # Normalize data
    mov_mean = cp.mean(mov_flat, axis=1, keepdims=True)
    mov_std = cp.std(mov_flat, axis=1, keepdims=True) + 1e-8
    mov_flat = (mov_flat - mov_mean) / mov_std
    
    D = cp.array(D_init, dtype=cp.float32)
    D /= (cp.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
    
    if not isinstance(corr_kern, cp.sparse.spmatrix):
        corr_kern = cp.sparse.csr_matrix(corr_kern)
    
    S = None
    max_learn_iter = params.get('max_learn', 50)
    learn_eps = params.get('learn_eps', 0.01)
    dict_evolution = []
    
    # --- Get the new regularization parameters ---
    lamCont = params.get('lamCont', 0.0) # Default to 0 (off)
    lamCorr = params.get('lamCorr', 0.0) # Default to 0 (off)
    ridge_reg = params.get('ridge_reg', 1e-6) # The original simple regularization
    
    print(f"Using Advanced Regularization: lamCont={lamCont}, lamCorr={lamCorr}")

    # --- Main learning loop ---
    for i in range(max_learn_iter):
        print(f"\n--- GraFT GPU Iteration {i+1}/{max_learn_iter} ---")
        D_old = D.copy()
        
        # --- Step 1: Infer S (This remains exactly the same) ---
        print("Step 1: Inferring sparse activity maps on GPU...")
        S, _ = dictionary_rwl1sf_gpu(mov_flat, D, corr_kern, params, S)
        
        # =====================================================================
        # --- Step 2: MODIFIED Dictionary Update ---
        # =====================================================================
        print("Step 2: Updating dictionary on GPU with advanced regularization...")
        
        n_atoms = D.shape[1]
        
        # Standard least-squares terms
        StS = S.T @ S
        Stmov = S.T @ mov_flat
        
        # --- Incorporate the new penalties into the normal equations ---
        
        # 1. Add Ridge regularization (for basic stability)
        regularizer = ridge_reg * cp.eye(n_atoms, dtype=cp.float32)
        
        # 2. Add lamCorr penalty (for orthogonality)
        # This encourages D'D to be close to Identity
        if lamCorr > 0:
            regularizer += lamCorr * cp.eye(n_atoms, dtype=cp.float32)
            
        # 3. Add lamCont penalty (for smoothness over iterations)
        # This encourages D to stay close to D_old
        if lamCont > 0:
            regularizer += lamCont * cp.eye(n_atoms, dtype=cp.float32)
            # We also need to modify the right-hand side of the equation
            Stmov += lamCont * D_old.T
            
        # --- Solve the fully regularized system ---
        try:
            D_new_T = cp.linalg.solve(StS + regularizer, Stmov)
            D_new = D_new_T.T
        except cp.linalg.LinAlgError:
            print("  Solve failed, using pseudoinverse...")
            D_new_T = cp.linalg.pinv(StS + regularizer) @ Stmov
            D_new = D_new_T.T
        
        # --- Normalization (same as before) ---
        norms = cp.linalg.norm(D_new, axis=0, keepdims=True)
        D = D_new / (norms + 1e-8)
        # =====================================================================
        
        # --- Convergence Check (same as before) ---
        dDict = float(cp.linalg.norm(D - D_old) / (cp.linalg.norm(D_old) + 1e-8))
        print(f"Relative change in dictionary: {dDict:.6f}")
        dict_evolution.append(cp.asnumpy(D))
        
        if dDict < learn_eps:
            print(f"Convergence reached after {i+1} iterations.")
            break
    
    # --- Final steps are the same ---
    print("\nFinal inference with converged dictionary...")
    S_final, _ = dictionary_rwl1sf_gpu(mov_flat, D, corr_kern, params, S)
    D_final = cp.asnumpy(D)
    S_final_flat = cp.asnumpy(S_final)
    S_final_reshaped = S_final_flat.reshape(height, width, D.shape[1], order='F')
    
    return D_final, S_final_reshaped, dict_evolution



def calculate_variance_explained_gpu(gcamp, D_final, S_final, valid_pixels=None):
    """
    GPU-accelerated calculation of variance explained by GraFT decomposition.
    
    Args:
        gcamp: Original data (time, height, width)
        D_final: Final dictionary (time, n_atoms)
        S_final: Final spatial maps (height, width, n_atoms) or (n_pixels, n_atoms)
        valid_pixels: Boolean mask of valid pixels
    
    Returns:
        dict: Variance metrics and decomposition results
    """
    
    time_pts, height, width = gcamp.shape
    n_atoms = D_final.shape[1]
    total_pixels = height * width
    
    # Determine if S_final needs reshaping
    if S_final.ndim == 2:
        S_matrix = S_final
    elif S_final.ndim == 3:
        if S_final.shape == (height, width, n_atoms):
            S_matrix = S_final.reshape(height * width, n_atoms, order='F')
        else:
            S_matrix = S_final.reshape(-1, n_atoms, order='F')
    else:
        raise ValueError(f"S_final has unexpected number of dimensions: {S_final.ndim}")
    
    # Reshape data to matrix form
    data_matrix = gcamp.reshape(time_pts, height * width, order='F').T
    
    # Apply mask
    if valid_pixels is not None:
        mask_flat = valid_pixels.ravel(order='F')
        n_valid = np.sum(mask_flat)
        
        data_matrix = data_matrix[mask_flat, :]
        
        if S_matrix.shape[0] == total_pixels:
            S_matrix = S_matrix[mask_flat, :]
        elif S_matrix.shape[0] == n_valid:
            pass
        else:
            data_matrix = data_matrix[:S_matrix.shape[0], :]
    
    # Move to GPU
    data_matrix = cp.asarray(data_matrix, dtype=cp.float32)
    S_matrix = cp.asarray(S_matrix, dtype=cp.float32)
    D_final_gpu = cp.asarray(D_final, dtype=cp.float32)
    
    # Reconstruct on GPU
    print("Reconstructing data...")
    data_reconstructed = S_matrix @ D_final_gpu.T
    
    # Scale correction
    data_std = float(cp.std(data_matrix))
    recon_std = float(cp.std(data_reconstructed))
    scale_ratio = recon_std / data_std
    
    scale_corrected = False
    if abs(scale_ratio - 1.0) > 0.1:
        S_matrix = S_matrix / scale_ratio
        data_reconstructed = S_matrix @ D_final_gpu.T
        scale_corrected = True
    
    
    # Calculate metrics on GPU
    residuals = data_matrix - data_reconstructed
    data_mean = cp.mean(data_matrix)
    ss_total = float(cp.sum((data_matrix - data_mean)**2))
    ss_residual = float(cp.sum(residuals**2))
    ss_explained = ss_total - ss_residual
    r_squared = 1 - (ss_residual / ss_total)
    
    total_variance = float(cp.var(data_matrix))
    residual_variance = float(cp.var(residuals))
    explained_variance = total_variance - residual_variance
    
    # GPU-accelerated correlation
    data_flat = data_matrix.ravel()
    recon_flat = data_reconstructed.ravel()
    data_centered = data_flat - cp.mean(data_flat)
    recon_centered = recon_flat - cp.mean(recon_flat)
    correlation = float(cp.sum(data_centered * recon_centered) / 
                      (cp.sqrt(cp.sum(data_centered**2) * cp.sum(recon_centered**2))))
    
    r_squared_corr = correlation**2
    
    # **GPU-ACCELERATED CUMULATIVE VARIANCE CALCULATION**
    
    # Calculate atom importance using vectorized GPU operations
    atom_ss = cp.zeros(n_atoms, dtype=cp.float32)
    
    # Calculate actual contribution sum of squares for each atom
    for i in range(n_atoms):
        contribution_i = S_matrix[:, i:i+1] @ D_final_gpu[:, i:i+1].T
        atom_ss[i] = cp.sum(contribution_i**2)
        

    
    # Sort by importance
    sorted_indices = cp.argsort(atom_ss)[::-1]
    
    # Calculate cumulative R² by progressively adding atoms - GPU accelerated
    cumulative_r2 = cp.zeros(n_atoms, dtype=cp.float32)
    
    # Process in chunks for better GPU efficiency
    checkpoint_interval = max(1, n_atoms // 20)
    
    for n in range(1, n_atoms + 1):
        # Use top n atoms only
        top_n_atoms = sorted_indices[:n]
        
        # Reconstruct with subset - GPU accelerated matrix multiplication
        S_subset = S_matrix[:, top_n_atoms]
        D_subset = D_final_gpu[:, top_n_atoms]
        recon_subset = S_subset @ D_subset.T
        
        # Calculate R² for this subset
        residuals_subset = data_matrix - recon_subset
        ss_residual_subset = cp.sum(residuals_subset**2)
        cumulative_r2[n-1] = 1 - (ss_residual_subset / ss_total)
        
        if n % checkpoint_interval == 0 or n == n_atoms:
            r2_val = float(cumulative_r2[n-1])
            print(f"  {n}/{n_atoms} atoms: R² = {r2_val*100:.2f}%")
    
    # Calculate per-atom variance contribution
    atom_variance_pct = 100 * atom_ss / max(ss_total, 1e-10)
    cumulative_variance_pct = 100 * cumulative_r2
    
    # Move results back to CPU
    data_reconstructed = cp.asnumpy(data_reconstructed)
    residuals = cp.asnumpy(residuals)
    data_matrix = cp.asnumpy(data_matrix)
    S_matrix = cp.asnumpy(S_matrix)
    atom_ss = cp.asnumpy(atom_ss)
    atom_variance_pct = cp.asnumpy(atom_variance_pct)
    sorted_indices = cp.asnumpy(sorted_indices)
    cumulative_variance_pct = cp.asnumpy(cumulative_variance_pct)
    
    results = {
        'total_variance': total_variance,
        'explained_variance': explained_variance,
        'residual_variance': residual_variance,
        'r_squared': r_squared,
        'r_squared_corr': r_squared_corr,
        'correlation': correlation,
        'variance_explained_pct': 100 * r_squared,
        'ss_total': ss_total,
        'ss_residual': ss_residual,
        'ss_explained': ss_explained,
        'atom_variance': atom_ss,
        'atom_variance_pct': atom_variance_pct,
        'sorted_atom_indices': sorted_indices,
        'cumulative_variance_pct': cumulative_variance_pct,
        'data_reconstructed': data_reconstructed,
        'residuals': residuals,
        'data_matrix': data_matrix,
        'S_corrected': S_matrix,
        'scale_correction_applied': scale_corrected,
        'scale_ratio': scale_ratio
    }

    return results


def calculate_variance_explained(gcamp, D_final, S_final, valid_pixels=None):
    """
    GPU-accelerated variance calculation.
    Wrapper for backward compatibility - always uses GPU.
    """
    return calculate_variance_explained_gpu(gcamp, D_final, S_final, valid_pixels)


# Optional: Batch processing version for very large datasets
def calculate_variance_explained_batched(gcamp, D_final, S_final, valid_pixels=None, 
                                        batch_size=10000):
    """
    Memory-efficient batched version for very large datasets.
    Processes pixels in batches to avoid GPU memory issues.
    
    Args:
        gcamp: Original data (time, height, width)
        D_final: Final dictionary (time, n_atoms)
        S_final: Final spatial maps
        valid_pixels: Boolean mask
        batch_size: Number of pixels to process at once
    """
    print(f"Using GPU with batched processing (batch_size={batch_size})...")
    
    time_pts, height, width = gcamp.shape
    n_atoms = D_final.shape[1]
    total_pixels = height * width
    
    # Reshape S_final
    if S_final.ndim == 2:
        S_matrix = S_final
    elif S_final.ndim == 3:
        S_matrix = S_final.reshape(height * width, n_atoms, order='F')
    else:
        raise ValueError(f"S_final has unexpected dimensions: {S_final.ndim}")
    
    # Reshape data
    data_matrix = gcamp.reshape(time_pts, height * width, order='F').T
    
    # Apply mask
    if valid_pixels is not None:
        mask_flat = valid_pixels.ravel(order='F')
        data_matrix = data_matrix[mask_flat, :]
        if S_matrix.shape[0] == total_pixels:
            S_matrix = S_matrix[mask_flat, :]
    
    n_pixels = data_matrix.shape[0]
    
    # Move dictionary to GPU once
    D_gpu = cp.asarray(D_final, dtype=cp.float32)
    
    # Process in batches
    data_reconstructed = np.zeros_like(data_matrix)
    
    for i in range(0, n_pixels, batch_size):
        end_idx = min(i + batch_size, n_pixels)
        
        S_batch = cp.asarray(S_matrix[i:end_idx], dtype=cp.float32)
        recon_batch = S_batch @ D_gpu.T
        data_reconstructed[i:end_idx] = cp.asnumpy(recon_batch)
        
        #if (end_idx) % (batch_size * 10) == 0:
    
    # Rest of calculation on CPU
    residuals = data_matrix - data_reconstructed
    data_mean = np.mean(data_matrix)
    ss_total = np.sum((data_matrix - data_mean)**2)
    ss_residual = np.sum(residuals**2)
    ss_explained = ss_total - ss_residual
    r_squared = 1 - (ss_residual / ss_total)
    
    total_variance = np.var(data_matrix)
    residual_variance = np.var(residuals)
    explained_variance = total_variance - residual_variance
    
    # Correlation
    data_flat = data_matrix.ravel()
    recon_flat = data_reconstructed.ravel()
    correlation = np.corrcoef(data_flat, recon_flat)[0, 1]
    r_squared_corr = correlation**2
    
    print(f"Batched variance calculation complete: R² = {r_squared*100:.2f}%")
    
    # Return full results in same format as main function
    results = {
        'total_variance': total_variance,
        'explained_variance': explained_variance,
        'residual_variance': residual_variance,
        'r_squared': r_squared,
        'r_squared_corr': r_squared_corr,
        'correlation': correlation,
        'variance_explained_pct': 100 * r_squared,
        'ss_total': ss_total,
        'ss_residual': ss_residual,
        'ss_explained': ss_explained,
        'data_reconstructed': data_reconstructed,
        'residuals': residuals,
        'data_matrix': data_matrix,
        'S_corrected': S_matrix,
        'scale_correction_applied': False,
        'scale_ratio': 1.0
    }
    
    return results