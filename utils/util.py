import yaml
import numpy as np
import cupy as cp
import pandas as pd
import re

def load_config(path: str) -> dict:
    """
    Loads a YAML configuration file and processes it into a nested dictionary.

    Special processing includes converting 'x_select' and 'y_select' dictionaries
    into Python slice objects for easy NumPy array indexing.

    Args:
        path (str): The path to the config.yaml file.

    Returns:
        dict: A dictionary containing all configuration parameters.
    """
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{path}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
        
    # Combine the nested dictionaries into one for easier access
    # e.g., config['general']['save_dir'] becomes config['save_dir']
    flat_config = {}
    for key in config:
        flat_config.update(config[key])

    # Convert the selection dictionaries into proper slice objects
    if 'x_select' in flat_config:
        sel = flat_config['x_select']
        flat_config['x_select'] = slice(sel.get('start'), sel.get('end'))

    if 'y_select' in flat_config:
        sel = flat_config['y_select']
        flat_config['y_select'] = slice(sel.get('start'), sel.get('end'))

    return flat_config



def calculate_dff_cupy(movie: np.ndarray, 
                       method: str = 'mean', 
                       percentile: int = 20, 
                       epsilon: float = 1e-6) -> np.ndarray:
    """
    Calculates dF/F on the GPU using CuPy.

    The formula used is (F - F₀) / (F₀ + epsilon), where F₀ is the baseline.

    Args:
        movie (np.ndarray): The input movie as a NumPy array with shape 
                            (frames, height, width).
        method (str): The method to calculate the baseline F₀. 
                      Options: 'mean', 'median', 'percentile'. 
                      Defaults to 'percentile'.
        percentile (int): The percentile to use if method is 'percentile'. 
                          Defaults to 20.
        epsilon (float): A small value added to the denominator to prevent 
                         division by zero. Defaults to 1e-6.

    Returns:
        np.ndarray: The dF/F movie as a NumPy array on the CPU.
    """

    
    # 1. Move the data from CPU (NumPy) to GPU (CuPy)
    movie_gpu = cp.asarray(movie, dtype=cp.float32)
    
    # 2. Calculate the baseline fluorescence (F₀) on the GPU
    #    keepdims=True is essential for correct broadcasting later.
    if method == 'mean':
        f0_gpu = movie_gpu.mean(axis=0, keepdims=True)
    elif method == 'median':
        f0_gpu = cp.median(movie_gpu, axis=0, keepdims=True)
    elif method == 'percentile':
        f0_gpu = cp.percentile(movie_gpu, q=percentile, axis=0, keepdims=True)
    else:
        raise ValueError("Method must be one of 'mean', 'median', or 'percentile'.")

    # 3. Calculate dF/F = (F - F₀) / F₀ on the GPU
    #    The small epsilon prevents division by zero in dark background pixels.
    dff_gpu = (movie_gpu - f0_gpu) / (f0_gpu + epsilon)

    return dff_gpu

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
    
def extract_contour_coords(df_path):
    columns = ['left_x', 'left_y', 'right_x', 'right_y','left_center','right_center']
    convert = {i:eval for i in columns}
    df = pd.read_csv(df_path,index_col=0,converters=convert)
    df = df[(df['left_center'] != "(None, None)") & (df['right_center'] != "(None, None)")]
    df = df.reset_index()
    contour_coords = []
    exclude = ['FRP','PL','RSPv','ACAd']
    for index, row in df.iterrows():
        # Get the coordinates for the left and right contours
        if row['acronym'] not in exclude:
            left_x_coords = row['left_x']
            left_y_coords = row['left_y']
            right_x_coords = row['right_x']
            right_y_coords = row['right_y']
            if len(left_x_coords) > 0 and len(left_y_coords) > 0:
                contour_coords.append((left_x_coords, left_y_coords))
            if len(right_x_coords) > 0 and len(right_y_coords) > 0:
                contour_coords.append((right_x_coords, right_y_coords))
    return contour_coords


