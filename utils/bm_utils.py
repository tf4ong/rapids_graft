import numpy as np
from tqdm import tqdm 
import os
import tifffile as tf
import json
from skimage.draw import polygon, polygon_perimeter
from denoise import denoise_ci_in_time_fast
from cupyx.scipy.signal import sosfiltfilt
import cupy as cp
from cupyx.scipy.signal import butter, filtfilt


def get_trial_folder_path(base_path, trial_number):
    """
    Returns the folder path for the given trial number.
    
    Parameters:
    - base_path: The path where the trial folders are located.
    - trial_number: The trial number you're looking for.
    
    Returns:
    - The full path to the trial folder, or None if not found.
    """
    
    # Generate possible folder names
    possible_names = [f"Trial_{trial_number}", f"trial_{trial_number}"]
    
    # Check each possible name
    for name in possible_names:
        folder_path = os.path.join(base_path, name)
        if os.path.exists(folder_path):
            return folder_path
    
    # If none of the names exist, return None
    return None

def load_mask(path,image_shape,exclude = ['FRP','PL','RSPv','ACAd','MOB']):
    columns = ['left_x', 'left_y', 'right_x', 'right_y','left_center', 'right_center']
    convert = {i:eval for i in columns}
    allen_mask2 = pd.read_csv(path,index_col=0,converters=convert)
    mask = np.zeros(image_shape, dtype=np.uint8)
    for index, row in allen_mask2.iterrows():
        if row['acronym'] not in exclude:
            left_x_coords = row['left_x']
            left_y_coords = row['left_y']
            right_x_coords = row['right_x']
            right_y_coords = row['right_y']
            if len(left_x_coords) > 0 and len(left_y_coords) > 0:
                rr, cc = polygon(left_y_coords, left_x_coords)
                mask[rr, cc] = 1  # Set the pixels inside the contour to 1
                rr, cc = polygon_perimeter(left_y_coords, left_x_coords)
                mask[rr, cc] = 1
            if len(right_x_coords) > 0 and len(right_y_coords) > 0:
                rr, cc = polygon(right_y_coords, right_x_coords)
                mask[rr, cc] = 1  # Set the pixels inside the contour to 1
                rr, cc = polygon_perimeter(right_y_coords, right_x_coords)
                mask[rr, cc] = 1

    return mask, 


def extract_pixels(stk,valid_pixels):
    n,_,_ = stk.shape
    stk = stk.reshape(n,-1)
    stk= stk[:,valid_pixels]
    return stk

def load_offset(path):
    with open(path, 'rb') as f:
        offsets = json.load(f)
    return offsets

def get_brian_stk(trial_path):
    hemo_path = os.path.join(trial_path,'brain/hemo_dff.tif')
    gcamp_path = os.path.join(trial_path,'brain/gcamp_dff.tif')
    gcamp =tf.imread(gcamp_path)
    hemo = tf.imread(hemo_path)
    return gcamp, hemo



# Assume get_trial_folder_path, get_brian_stk, denoise_ci_in_time_fast exist

def session_process_with_indices(base_path, config, denoise=True):
    """
    Processes a session, concatenates the trial data, and also returns the
    indices needed to split the concatenated arrays back into individual trials.
    """
    gcamps_list = []
    hemos_list = []
    
    # --- NEW: Lists to store the length (number of frames) of each trial ---
    gcamp_lengths = []
    hemo_lengths = []
    
    pbar = tqdm(total=120, leave=True, position=0)
    for i in range(120):
        trial_path = get_trial_folder_path(base_path, i)
        gcamp, hemo = get_brian_stk(trial_path)
        
        # --- NEW: Store the shape *before* any processing that might change it ---
        gcamp_lengths.append(gcamp.shape[0])
        hemo_lengths.append(hemo.shape[0])
        
        if denoise:
            gcamp = denoise_ci_in_time_fast(gcamp, config)
            hemo = denoise_ci_in_time_fast(hemo, config)
            
        gcamps_list.append(gcamp)
        hemos_list.append(hemo)
        pbar.update(1)
    
    # Concatenate the arrays as before
    gcamps_concat = np.concatenate(gcamps_list)
    hemos_concat = np.concatenate(hemos_list)
    
    # --- NEW: Calculate the split indices ---
    # The split indices are the cumulative sum of the trial lengths.
    # We drop the last element because np.split doesn't need the final endpoint.
    gcamp_split_indices = np.cumsum(gcamp_lengths)[:-1]
    hemo_split_indices = np.cumsum(hemo_lengths)[:-1]
    
    # Return the concatenated arrays AND the indices needed to split them
    return gcamps_concat, hemos_concat, gcamp_split_indices, hemo_split_indices

def align_brain(stk, valid_pixels, offsets):
    aligned_tensor = np.roll(stk, shift=(offsets['offset_y'], offsets ['offset_x']), axis=(1, 2))
    #aligned_tensor=extract_pixels(aligned_tensor,valid_pixels)
    return aligned_tensor


def highpass_detrend_trials(gcamp_concat, gsplit_indices, fs, highpass=0.1, degree=1):
    """GPU highpass filter then polynomial detrend per trial."""
    gcamp_gpu = cp.asarray(gcamp_concat)
    split_indices_list = gsplit_indices.tolist() if hasattr(gsplit_indices, 'tolist') else list(gsplit_indices)
    list_of_trials = cp.split(gcamp_gpu, split_indices_list, axis=0)
    
    # Design highpass Butterworth filter
    sos = butter(4, highpass, btype='highpass', fs=fs, output='sos')
    
    detrended_trials_list = []
    
    for trial in list_of_trials:
        n_timepoints = trial.shape[0]
        original_shape = trial.shape
        trial_2d = trial.reshape(n_timepoints, -1)
        
        # Highpass filter each pixel
        filtered_2d = sosfiltfilt(sos, trial_2d, axis=0)
        
        # Polynomial detrend
        t = cp.arange(n_timepoints, dtype=cp.float32)
        X = cp.vstack([t**i for i in range(degree + 1)]).T
        XtX = X.T @ X
        XtY = X.T @ filtered_2d
        beta = cp.linalg.solve(XtX, XtY)
        trend = X @ beta
        detrended_2d = filtered_2d - trend
        
        detrended_trial = detrended_2d.reshape(original_shape)
        detrended_trials_list.append(detrended_trial)
    
    return cp.concatenate(detrended_trials_list, axis=0)

def demean_trials(gcamp_concat, gsplit_indices):

    list_of_trials = np.split(gcamp_concat, gsplit_indices, axis=0)

    demeaned_trials_list = []
    
    for i, trial in enumerate(list_of_trials):

        trial_mean = np.mean(trial, axis=0, keepdims=True)
        
        demeaned_trial = trial - trial_mean
        
        demeaned_trials_list.append(demeaned_trial)
        
    
    gcamp_demeaned_concat = np.concatenate(demeaned_trials_list, axis=0)
    
    
    return gcamp_demeaned_concat