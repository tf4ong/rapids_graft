import numpy as np
from skimage.restoration import denoise_wavelet
import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_wavelet
import time
import matplotlib.pyplot as plt
from utils import util 


def denoise_ci_in_time_fast(mov: np.ndarray, config: dict) -> np.ndarray:
    """
    Denoise a movie with shape (frames, height, width) efficiently using
    parameters from a configuration dictionary.
    """
    # Extract denoising parameters from the config dictionary
    # Use .get() to provide default values in case keys are missing
    denoise_params = config.get('denoising', {})
    smooth_lvl = denoise_params.get('smooth_level', 4)
    
    if smooth_lvl == 0:
        print("Smoothing level is 0, skipping denoising.")
        return mov

    wavelet_type = denoise_params.get('wavelet_type', 'sym4')
    denoising_method = denoise_params.get('denoising_method', 'BayesShrink')
    

    mov_transposed = mov.transpose(1, 2, 0)

    denoised_transposed = denoise_wavelet(
        mov_transposed,
        wavelet=wavelet_type,
        method=denoising_method,
        wavelet_levels=smooth_lvl,
        channel_axis=-1,
        convert2ycbcr=False
    )

    denoised_mov = denoised_transposed.transpose(2, 0, 1)

    return denoised_mov.astype(np.float32)

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # --- 1. Load the configuration from the YAML file ---
    print("Loading configuration from 'config.yaml'...")
    config = util.load_config('config.yaml')

    # --- 2. Create a sample movie with noise ---
    frames, height, width = 500, 100, 120
    print(f"Creating sample movie of shape ({frames}, {height}, {width})...")
    
    time_vec = np.linspace(0, 10 * np.pi, frames)
    clean_signal = np.sin(time_vec)
    clean_signal_reshaped = clean_signal.reshape(-1, 1, 1)
    clean_movie = np.ones((1, height, width)) * clean_signal_reshaped
    noise = np.random.randn(frames, height, width) * 0.5
    noisy_movie = (clean_movie + noise).astype(np.float32)

    # --- 3. Time the optimized version using the loaded config ---
    print("\nTiming the optimized version...")
    start_time_fast = time.time()
    # Pass the entire config dictionary to the function
    denoised_fast = denoise_ci_in_time_fast(noisy_movie.copy(), config)
    end_time_fast = time.time()
    duration_fast = end_time_fast - start_time_fast
    
    print(f"Optimized version took: {duration_fast:.4f} seconds")

    # --- 4. Display the results ---
    print("\nVisualizing the result for a single pixel...")
    pixel_h, pixel_w = 50, 60

    plt.figure(figsize=(14, 7))
    plt.title(f'Denoising Result for Pixel ({pixel_h}, {pixel_w})', fontsize=16)
    plt.plot(noisy_movie[:, pixel_h, pixel_w], label='Noisy Trace', color='gray', alpha=0.6)
    plt.plot(clean_movie[:, pixel_h, pixel_w], label='Original Clean Signal', color='black', linestyle='--', linewidth=2)
    plt.plot(denoised_fast[:, pixel_h, pixel_w], label='Denoised Trace', color='red', linewidth=2.5)
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()