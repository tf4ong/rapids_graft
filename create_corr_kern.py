#!/usr/bin/env python3
"""
Faithful Python replication of MATLAB corr_kern creation from demo.m
with corrections for sparse matrix handling and memory errors.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, diags
import cupy as cp
import cudf
import cuml
from cuml.decomposition import PCA as cuPCA
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

# Import CuPy's sparse module, which is in a different location
try:
    import cupyx.scipy.sparse as cp_sparse
    CUPY_SPARSE_AVAILABLE = True
except ImportError:
    CUPY_SPARSE_AVAILABLE = False


# Check CUDA availability for RAPIDS
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("RAPIDS CUDA acceleration enabled")
except ImportError:
    CUDA_AVAILABLE = False
    print("RAPIDS not available, falling back to CPU")
    cp = np


class CorrKern:
    """
    Faithful Python replication of MATLAB corr_kern structure and processing.
    """
    
    def __init__(self, components=12, w_time=0, reduce_dim=True, corrType='embedding'):
        self.w_time = w_time
        self.reduce_dim = reduce_dim
        self.corrType = corrType
        self.n_components = components
        
    def create_embedding(self, data_obj, self_tune=7, dist_type='euclidean', 
                         kNN=49, verbose=0, use_cuda=True):
        """
        RAPIDS-accelerated replication of mkDataEmbedding.m functionality.
        """
        if verbose >= 2:
            print('Creating embedding with provided params.')
            
        use_gpu = CUDA_AVAILABLE and use_cuda
        
        if use_gpu:
            data_device = cp.asarray(data_obj, dtype=cp.float32)
        else:
            data_device = np.asarray(data_obj, dtype=np.float32)

        if self.reduce_dim:
            if data_device.ndim > 2:
                original_shape = data_device.shape
                data_2d = data_device.reshape(-1, original_shape[-1], order='F')
            else:
                data_2d = data_device

            if use_gpu:
                data_df = cudf.DataFrame(data_2d) 
                pca = cuPCA(n_components=self.n_components)
                data_transformed_df = pca.fit_transform(data_df)
                data_for_affinity = data_transformed_df
            else:
                pca = PCA(n_components=self.n_components)
                data_for_affinity = pca.fit_transform(data_2d)
        else:
            if data_device.ndim > 2:
                data_for_affinity = data_device.reshape(-1, data_device.shape[-1], order='F')
            else:
                data_for_affinity = data_device
                
        if use_gpu:
            K = self._calc_affinity_mat_cuda(data_for_affinity, self_tune, kNN)
        else:
            K = self._calc_affinity_mat(data_for_affinity, self_tune, kNN)
        
        # --- FIXED NORMALIZATION LOGIC ---
        if use_gpu:
            row_sums = K.sum(axis=1)
            inv_row_sums_flat = 1.0 / (cp.asarray(row_sums).flatten() + cp.finfo(cp.float32).eps)
            normalizer = cp_sparse.diags(inv_row_sums_flat)
            corr_kern_gpu = normalizer @ K
            return corr_kern_gpu.get() # Return to CPU at the very end
        else:
            row_sums = K.sum(axis=1)
            inv_row_sums_flat = 1.0 / (np.asarray(row_sums).flatten() + np.finfo(np.float32).eps)
            normalizer = diags(inv_row_sums_flat)
            corr_kern_cpu = normalizer @ K
            return corr_kern_cpu
    
       

    def estimate_knn_for_sparsity(self, data_obj,self_tune = 7, target_sparsity_percent=10.0, tol=0.5, max_iter=10,
                                min_k=1000, max_k=5000, verbose=True, use_cuda=True):
        """
        Estimate and find the kNN that gives a correlation kernel with the desired sparsity.

        Args:
            data_obj (np.ndarray): Input data of shape (frames, height, width).
            target_sparsity_percent (float): Target sparsity in % (e.g., 10.0 for 10%).
            tol (float): Acceptable error margin on sparsity.
            max_iter (int): Maximum binary search iterations.
            min_k (int): Minimum kNN value to test.
            max_k (int): Maximum kNN value to test.
            verbose (bool): Whether to print progress.
            use_cuda (bool): Use GPU or CPU.

        Returns:
            Tuple[int, float, csr_matrix]: (best_kNN, actual_sparsity_percent, kernel)
        """
        # Convert data from (F, H, W) to (H, W, F)
        if data_obj.ndim == 3:
            #data_obj = data_obj.transpose(1, 2, 0)
            #n_points = data_obj.shape[0] * data_obj.shape[1]
            data_to_embed = data_obj.transpose(1, 2, 0)
            n_points = data_to_embed.shape[0] * data_to_embed.shape[1]

        else:
             data_to_embed = data_obj.T
             n_points = data_obj.shape[0]

        low, high = min_k, max_k
        best_kNN = None
        best_kernel = None
        best_diff = float("inf")
        best_sparsity = 0.0

        for i in range(max_iter):
            kNN = (low + high) // 2
            if verbose:
                print(f"\n[Iter {i+1}] Trying kNN = {kNN}...")
            #.transpose(2, 0, 1)
            try:
                kernel = self.create_embedding(data_to_embed, self_tune= self_tune,
                                               kNN=kNN, use_cuda=use_cuda, verbose=0)
            except Exception as e:
                print(f"Error at kNN={kNN}: {e}")
                high = kNN - 1
                continue

            actual_sparsity = 100 * kernel.nnz / (n_points * n_points)
            diff = abs(actual_sparsity - target_sparsity_percent)

            if verbose:
                print(f"→ Sparsity: {actual_sparsity:.4f}% | Target: {target_sparsity_percent}% | Δ = {diff:.4f}")

            if diff < best_diff:
                best_diff = diff
                best_kNN = kNN
                best_kernel = kernel
                best_sparsity = actual_sparsity

            if diff <= tol:
                if verbose:
                    print("✅ Found acceptable kNN!")
                return best_kNN, best_sparsity, best_kernel

            if actual_sparsity < target_sparsity_percent:
                low = kNN + 1
            else:
                high = kNN - 1

        if verbose:
            print(f"\n⚠️ Max iterations reached. Closest match: kNN={best_kNN}, sparsity={best_sparsity:.4f}%")

        return best_kNN, best_sparsity, best_kernel

    def _calc_affinity_mat_cuda(self, data, self_tune, kNN):
        n_points = data.shape[0]
        nbrs = cuNearestNeighbors(n_neighbors=kNN + 1, algorithm='auto')
        nbrs.fit(data)
        distances, indices = nbrs.kneighbors(data)

        distances = cp.asarray(distances.values if hasattr(distances, 'values') else distances)[:, 1:]
        indices = cp.asarray(indices.values if hasattr(indices, 'values') else indices, dtype=cp.int32)[:, 1:]
        scales = distances[:, self_tune - 1] if self_tune <= kNN else distances[:, -1]

        i_indices = cp.repeat(cp.arange(n_points), kNN).astype(cp.int32)
        j_indices = indices.flatten()
        dist_values = distances.flatten()
        
        scales_i = scales[i_indices]
        scales_j = scales[j_indices]
        sigma = cp.sqrt(scales_i * scales_j)
        
        mask = sigma > 0
        affinity_values = cp.zeros_like(dist_values, dtype=cp.float32)
        affinity_values[mask] = cp.exp(-dist_values[mask]**2 / (2 * sigma[mask]**2))
        
        K_sparse = cp_sparse.coo_matrix((affinity_values, (i_indices, j_indices)), 
                                        shape=(n_points, n_points), dtype=cp.float32)
        K_symmetric = (K_sparse + K_sparse.T) / 2
        return K_symmetric.tocsr()

    def _calc_affinity_mat(self, data, self_tune, kNN):
        n_points = data.shape[0]
        nbrs = NearestNeighbors(n_neighbors=kNN+1, algorithm='auto').fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        scales = distances[:, self_tune-1] if self_tune <= kNN else distances[:, -1]
        
        i_indices = np.repeat(np.arange(n_points), kNN)
        j_indices = indices.flatten()
        dist_values = distances.flatten()
        
        scales_i = scales[i_indices]
        scales_j = scales[j_indices]
        sigma = np.sqrt(scales_i * scales_j)
        
        mask = sigma > 0
        affinity_values = np.zeros_like(dist_values, dtype=np.float32)
        affinity_values[mask] = np.exp(-dist_values[mask]**2 / (2 * sigma[mask]**2))
        
        K_sparse = coo_matrix((affinity_values, (i_indices, j_indices)), 
                              shape=(n_points, n_points), dtype=np.float32)
        K_symmetric = (K_sparse + K_sparse.T) / 2
        return K_symmetric.tocsr()

def create_corr_kern():
    return CorrKern(w_time=0, reduce_dim=True, corrType='embedding')

def run_demo():
    import time
    
    print("=" * 60)
    print("RAPIDS-Accelerated Correlation Kernel Demo")
    print("=" * 60)
    
    kernel = create_corr_kern()
    
    print("\n" + "=" * 60)
    print("Generating synthetic calcium imaging data...")
    print("=" * 60)
    
    height, width, n_frames = 320, 242, 500
    
    np.random.seed(42)
    data_3d = np.random.randn(height, width, n_frames).astype(np.float32)
    
    print(f"Data shape: {data_3d.shape}")
    
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    if CUDA_AVAILABLE and CUPY_SPARSE_AVAILABLE:
        print("Testing GPU acceleration...")
        start_time = time.time()
        corr_kern_gpu = kernel.create_embedding(data_3d, verbose=2, use_cuda=True)
        gpu_time = time.time() - start_time
        
        print(f"GPU Time: {gpu_time:.3f} seconds")
        print(f"Output shape: {corr_kern_gpu.shape}")
        # Use .nnz for sparse matrices
        print(f"Output sparsity: {corr_kern_gpu.nnz / (corr_kern_gpu.shape[0] * corr_kern_gpu.shape[1]) * 100:.3f}% non-zero")
        
        print("\nTesting CPU fallback...")
        start_time = time.time()
        corr_kern_cpu = kernel.create_embedding(data_3d, verbose=2, use_cuda=False)
        cpu_time = time.time() - start_time
        
        print(f"CPU Time: {cpu_time:.3f} seconds")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x faster with GPU")
        
        # FIXED: Commented out the memory-intensive comparison.
        # For matrices this large, comparing them directly is not feasible.
        # diff = np.abs(corr_kern_gpu.toarray() - corr_kern_cpu.toarray()).mean()
        # print(f"Mean absolute difference: {diff:.2e} (verification skipped)")
            
    else:
        print("RAPIDS or CuPy sparse not available - running CPU version...")
        start_time = time.time()
        corr_kern_cpu = kernel.create_embedding(data_3d, verbose=2, use_cuda=False)
        cpu_time = time.time() - start_time
        
        print(f"CPU Time: {cpu_time:.3f} seconds")
        print(f"Output shape: {corr_kern_cpu.shape}")
        print(f"Output sparsity: {corr_kern_cpu.nnz / corr_kern_cpu.size * 100:.3f}% non-zero")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
