import pandas as pd
import os
import re
import numpy as np
import subprocess
import corner
from matplotlib import pyplot as plt

# MCMC For Lensed Systems 

system_name = 'J1251'
model_ver = 'sie'
current_path = os.getcwd()
full_path = os.path.join(current_path, system_name, 'mcmc')
print(f'Full path: {full_path}')

def read_info(folder_path):
    info_file = os.path.join(folder_path, 'info.dat')
    z_lens, z_source = None, None
    if not os.path.exists(info_file):
        return None, None
    with open(info_file, 'r') as f:
        for line in f:
            if 'z_lens' in line.lower() and '=' in line:
                strs = re.findall(r'[\d\.]+', line.split('=')[1])
                if strs: z_lens = float(strs[0])
            elif 'z_source' in line.lower() and '=' in line:
                strs = re.findall(r'[\d\.]+', line.split('=')[1])
                if strs: z_source = float(strs[0])
    return z_lens, z_source

z_lens, z_source = read_info(f'./{system_name}')
print(f'z_lens: {z_lens}, z_source: {z_source}')

if z_lens is None:
    no_z_lens_flag = True
    z_lens = 0.5
else:    
    no_z_lens_flag = False

def calculate_distance(p1, p2):
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def assign_image_names(df, start_index):
    # Added extra letters just in case of >4 image systems
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    current_name_index = 0
    current_index = start_index
    assigned_indices = {current_index}
    
    while len(assigned_indices) < len(df):
        current_x, current_y = df.at[current_index, 'x'], df.at[current_index, 'y']
        next_index = None
        max_angle = -float('inf')
        
        for i in range(len(df)):
            if i in assigned_indices:
                continue
            dx = df.at[i, 'x'] - current_x
            dy = df.at[i, 'y'] - current_y
            angle = np.arctan2(dy, dx)
            if angle > max_angle:
                max_angle = angle
                next_index = i
        
        if next_index is not None:
            current_name_index += 1
            if current_name_index < len(names):
                df.at[next_index, 'Img'] = names[current_name_index]
            assigned_indices.add(next_index)
            current_index = next_index

def rms_extract(model_ver, model_path, obs_point, n_img, h0=False, time_delay=False):
    opt_result_file = os.path.join(model_path, f'{model_ver}_optresult.dat')
    with open(opt_result_file, 'r') as file: opt_result = file.readlines()
    last_optimize_index = next((i for i, line in reversed(list(enumerate(opt_result))) if 'optimize' in line), None)
    if last_optimize_index is None: raise ValueError("No 'optimize' line found.")
    opt_result = opt_result[last_optimize_index + 1:]
    lens_params_dict = {}
    for line in opt_result:
        if line.startswith('lens'):
            parts = re.split(r'\s+', line.strip())
            lens_params_dict[parts[1]] = [float(x) for x in parts[2:]]
    source_params = [[float(x) for x in re.split(r'\s+', line.strip())[1:]] for line in opt_result if line.startswith('point')]
    chi2_line = next((line for line in opt_result if 'chi^2' in line), None)
    if chi2_line is None: raise ValueError("No 'chi^2' line found.")
    chi2_value = float(chi2_line.split('=')[-1].strip().split()[0])
    hubble_val = None
    if h0:
        hubble_line = next((line for line in opt_result if 'hubble =' in line), None)
        if hubble_line: hubble_val = float(hubble_line.split('=')[-1].strip().split()[0])
    out_point_file = os.path.join(model_path, f'{model_ver}_point.dat')
    
    if not os.path.exists(out_point_file):
        return np.nan, [], np.nan, [], [], 0, chi2_value, source_params, lens_params_dict, hubble_val, None, None, None, None, None

    out_point = pd.read_csv(out_point_file, sep=r'\s+', header=None, skiprows=1, names=['x', 'y', 'mag', 'td', 'col5', 'col6', 'col7', 'col8'])
    num_pred_images = len(out_point)
    if num_pred_images == 0:
        return np.nan, [], np.nan, [], [], 0, chi2_value, source_params, lens_params_dict, hubble_val, None, None, None, None, out_point

    if num_pred_images > n_img:
        distance_matrix = np.zeros((n_img, num_pred_images))
        for i in range(n_img):
            for j in range(num_pred_images):
                distance_matrix[i, j] = calculate_distance(obs_point.iloc[i], out_point.iloc[j])

        matched_indices = np.unravel_index(np.argsort(distance_matrix, axis=None), distance_matrix.shape)
        matched_pred_indices = set()
        matched_obs_indices = set()
        matches = []
        for obs_idx, pred_idx in zip(*matched_indices):
            if obs_idx not in matched_obs_indices and pred_idx not in matched_pred_indices:
                matches.append((obs_idx, pred_idx))
                matched_obs_indices.add(obs_idx)
                matched_pred_indices.add(pred_idx)
            if len(matches) == n_img:
                break

        matches.sort(key=lambda x: x[0])
        matched_pred_indices = [pred_idx for _, pred_idx in matches]
        out_point = out_point.iloc[matched_pred_indices].reset_index(drop=True)
        out_point['Img'] = obs_point['Img'].values
    
    if num_pred_images < n_img:
        distance_matrix = np.zeros((n_img, num_pred_images))
        for i in range(n_img):
            for j in range(num_pred_images):
                distance_matrix[i, j] = calculate_distance(obs_point.iloc[i], out_point.iloc[j])

        matched_indices = np.unravel_index(np.argsort(distance_matrix, axis=None), distance_matrix.shape)
        matched_pred_indices = set()
        matched_obs_indices = set()
        matches = []
        for obs_idx, pred_idx in zip(*matched_indices):
            if obs_idx not in matched_obs_indices and pred_idx not in matched_pred_indices:
                matches.append((obs_idx, pred_idx))
                matched_obs_indices.add(obs_idx)
                matched_pred_indices.add(pred_idx)
            if len(matches) == num_pred_images:
                break

        matches.sort(key=lambda x: x[0])
        matched_pred_indices = [pred_idx for _, pred_idx in matches]
        out_point = out_point.iloc[matched_pred_indices].reset_index(drop=True)
        out_point['Img'] = [obs_point.at[obs_idx, 'Img'] for obs_idx, _ in matches]

    if num_pred_images == n_img:
        distance_matrix = np.zeros((n_img, n_img))
        for i in range(n_img):
            for j in range(n_img):
                distance_matrix[i, j] = calculate_distance(obs_point.iloc[i], out_point.iloc[j])
        
        matched_indices = np.unravel_index(np.argsort(distance_matrix, axis=None), distance_matrix.shape)
        matched_pred_indices = set()
        matched_obs_indices = set()
        matches = []
        for obs_idx, pred_idx in zip(*matched_indices):
            if obs_idx not in matched_obs_indices and pred_idx not in matched_pred_indices:
                matches.append((obs_idx, pred_idx))
                matched_obs_indices.add(obs_idx)
                matched_pred_indices.add(pred_idx)
            if len(matches) == n_img:
                break
        matches.sort(key=lambda x: x[0])
        matched_pred_indices = [pred_idx for _, pred_idx in matches]
        out_point = out_point.iloc[matched_pred_indices].reset_index(drop=True)
        out_point['Img'] = [obs_point.at[obs_idx, 'Img'] for obs_idx, _ in matches]
    
    obs_point = obs_point.sort_values(by='Img').reset_index(drop=True)
    out_point = out_point.sort_values(by='Img').reset_index(drop=True)

    image_rms = []
    pos_rms = np.nan

    num_matched_images = len(out_point)

    if num_matched_images > 0:
        for i in range(num_matched_images):
            obs_row = obs_point[obs_point['Img'] == out_point.at[i, 'Img']]
            if not obs_row.empty:
                dist = np.sqrt((obs_row.iloc[0]['x'] - out_point.at[i, 'x'])**2 + 
                               (obs_row.iloc[0]['y'] - out_point.at[i, 'y'])**2)
                image_rms.append(dist)

        if image_rms:
            pos_rms = np.sqrt(np.sum(np.array(image_rms)**2) / num_matched_images)

    image_rms = np.array(image_rms)

    flux_rms = []
    mag_rms = np.nan
    for i in range(len(out_point)):
        diff = abs(abs(out_point.at[i, 'mag']) - abs(obs_point.at[i, 'mag']))
        flux_rms.append(diff)
        mag_rms = np.sqrt(np.sum(np.array(flux_rms)**2) / len(out_point))
    flux_rms = np.array(flux_rms)

    percentage_errors = []
    for i in range(len(out_point)):
        perc_error = abs(abs((out_point.at[i, 'mag']) - abs(obs_point.at[i, 'mag']))) / abs(obs_point.at[i, 'mag']) * 100
        percentage_errors.append(perc_error)
    avg_percentage_error = np.mean(percentage_errors) if len(percentage_errors) > 0 else 0
    percentage_errors = np.array(percentage_errors)

    percentage_errors_td = []
    avg_percentage_error_td = None
    td_rms = None
    if time_delay:
        out_point['td_rms'] = np.nan
        for i in range(len(out_point)):
            diff = out_point.at[i, 'td'] - obs_point.at[i, 'td']
            out_point.at[i, 'td_rms'] = diff
        percentage_errors_td = [abs(out_point.at[i, 'td_rms'] / obs_point.at[i, 'td']) * 100 for i in range(len(out_point)) if obs_point.at[i, 'td'] != 0]
        for i in range(len(out_point)):
            if obs_point.at[i, 'td'] == 0:
                percentage_errors_td.insert(i, 0)
        avg_percentage_error_td = np.mean(percentage_errors_td) if len(percentage_errors_td) > 0 else 0
        percentage_errors_td = np.array(percentage_errors_td)
        td_rms = np.sqrt(np.sum(out_point['td_rms']**2) / len(out_point))
    else:
        td_rms = None
        percentage_errors_td = None
        avg_percentage_error_td = None
    
    td_vals = np.array(out_point['td']) if time_delay else None
    return pos_rms, image_rms, mag_rms, flux_rms, percentage_errors, avg_percentage_error, chi2_value, source_params, lens_params_dict, hubble_val, td_vals, td_rms, percentage_errors_td, avg_percentage_error_td, out_point

# Finding the FOV of the system
obs_point_file = f'./{system_name}/pos+flux_point.dat'

# Using raw string `r'\s+'` for separator instead of `\s+` to prevent 'Invalid escape sequence' warnings
obs_point = pd.read_csv(obs_point_file, sep=r'\s+', header=None, skiprows=1, names=['x', 'y', 'mag', 'pos_err', 'mag_err', 'td', 'td_err', 'parity'])
obs_point['Img'] = None  # Init the empty column for pandas compatibility

brightest_index = obs_point['mag'].idxmax()
obs_point.at[brightest_index, 'Img'] = 'A'
assign_image_names(obs_point, brightest_index)

n_img = len(obs_point)

x_min, x_max = obs_point['x'].min(), obs_point['x'].max()
y_min, y_max = obs_point['y'].min(), obs_point['y'].max()
x_min -= 1
x_max += 1
y_min -= 1
y_max += 1
print(f'FOV: x [{x_min}, {x_max}], y [{y_min}, {y_max}]')

# Define center of the lens
lens_center_x = (x_min + x_max) / 2
lens_center_y = (y_min + y_max) / 2
lens_center_x = round(lens_center_x, 3)
lens_center_y = round(lens_center_y, 3)

# Extract results
pos_rms, image_rms, mag_rms, flux_rms, percentage_errors, avg_percentage_error, chi2_value, source_params, lens_params_dict, hubble_val, td_vals, td_rms, percentage_errors_td, avg_percentage_error_td, out_point = rms_extract('sie', f'./{system_name}/', obs_point, n_img=n_img, h0=False, time_delay=False)

print(lens_params_dict)

# Create a sigfile.dat file for MCMC
sigfile_path = f'./{system_name}/mcmc/sigfile.dat'
with open(sigfile_path, 'w') as f:
    f.write(f"""5
0.5
0.005
0.005
0.005
0.05""")


# Create the point.input file 
input_file = f'./{system_name}/mcmc/point.input'
with open(input_file, 'w') as f:
    f.write(f"""

## setting primary parameters
omega     0.300000
lambda    0.700000
weos      -1.000000
hubble    0.700000
prefix    {full_path}/{model_ver}
xmin	  {x_min}
ymin	  {y_min}
xmax	  {x_max}
ymax	  {y_max}
pix_ext   0.005
pix_poi   0.005
maxlev	  1

## some examples of secondary parameters
chi2_splane    0 
chi2_restart   -1
chi2_usemag    0
hvary          0
flag_mcmcall   0

## define lenses and sources
startup 1 0 1
lens sie  {lens_params_dict['sie'][0]} {lens_params_dict['sie'][1]} {lens_params_dict['sie'][2]} {lens_params_dict['sie'][3]} {lens_params_dict['sie'][4]} {lens_params_dict['sie'][5]} 0.0 0.0
point {z_source} {source_params[0][1]} {source_params[0][2]}
end_startup

## for optimizations
## can be ignored unless you do opts
start_setopt
0 1 1 1 1 1 0 0
0 1 1
end_setopt

## execute commands
start_command
readobs_point {current_path}/{system_name}/pos_point.dat
mcmc_sigma {full_path}/sigfile.dat
mcmc 10000
quit
""")

# Run the MCMC using subprocess
subprocess.run(['/Users/ainsleylewis/glafic2/glafic', input_file], check=True)

# After running the MCMC create a corner plot of the results using the output files generated by glafic
# Find the number of parameters from the sigfile.dat second line
with open(sigfile_path, 'r') as f:
    f.readline()  # Skip the first line
    num_params = int(f.readline().strip())

# Generate parameter names as param1, param2, ..., paramN
param_names = [f'param{i+1}' for i in range(num_params)]

# Load MCMC file 
mcmc_output_file = f'./{system_name}/mcmc/{model_ver}_mcmc.dat'

# Read the MCMC output file 
mcmc_data = pd.read_csv(mcmc_output_file, delim_whitespace=True, names= ['chi2'] + param_names)
print(mcmc_data.head())

# Create corner plot
corner.corner(mcmc_data[param_names], labels=param_names, show_titles=True, title_fmt='.2f')
# Save the corner plot
corner_plot_path = f'./{system_name}/mcmc/{model_ver}_corner.png'
plt.savefig(corner_plot_path)
print(f'Corner plot saved to: {corner_plot_path}')
