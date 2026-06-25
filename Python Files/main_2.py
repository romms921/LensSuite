import glafic 
import pandas as pd
import os
import re
import numpy as np 

system_name = 'B2045'
model_ver = 'sie'

# Import system info
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

z_lens, z_source = read_info(f'./{system_name}')
print(f'z_lens: {z_lens}, z_source: {z_source}')

if z_lens is None:
    no_z_lens_flag = True
    z_lens = 0.5
else:    
    no_z_lens_flag = False


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


glafic.init(0.3, 0.7, -1.0, 0.7, f'./{system_name}/{model_ver}', x_min, y_min, x_max, y_max, 0.001, 0.001, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    0', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(1, 0, 1)
glafic.set_lens(1, 'sie', z_lens, 100, lens_center_x, lens_center_y, 0.17, 0, 0.0, 0.0)
glafic.set_point(1, z_source, lens_center_x, lens_center_y)

glafic.setopt_lens(1, 1, 1, 1, 1, 1, 1, 0, 0)
glafic.setopt_point(1, 0, 1, 1)


# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point(f'./{system_name}/pos_point.dat')
glafic.optimize()
glafic.findimg()

glafic.quit()


# Extract results
pos_rms, image_rms, mag_rms, flux_rms, percentage_errors, avg_percentage_error, chi2_value, source_params, lens_params_dict, hubble_val, td_vals, td_rms, percentage_errors_td, avg_percentage_error_td, out_point = rms_extract('sie', f'./{system_name}/', obs_point, n_img=n_img, h0=False, time_delay=False)

print(f'Position RMS: {pos_rms}')
print(f'Image RMS: {image_rms}')
print(f'Magnitude RMS: {mag_rms}')
print(f'Flux RMS: {flux_rms}')
print(f'Percentage Errors: {percentage_errors}')
print(f'Average Percentage Error: {avg_percentage_error}%')
print(f'Chi^2 Value: {chi2_value}')
print(f'Source Parameters: {source_params}')
print(f'Lens Parameters: {lens_params_dict}')
print(f'Hubble Value: {hubble_val}')
print(f'Time Delay Values: {td_vals}')
print(f'Time Delay RMS: {td_rms}')
print(f'Average Percentage Error in Time Delay: {avg_percentage_error_td}%')

# Writing the results to results.dat
# Check if results.dat already exists and if the system entry is present
# If the system entry is present check if the current chi2 value is better than the existing one
# If it is better then update the entry otherwise do not update
# results.dat format: 
# System	Z_Lens	Chi2	Pos_RMS	Mag_RMS	Avg_Flux_Err_%	Lens_Params	Source_Params	TD_RMS
# B2045	0.4785	143703.8	0.9476940236700873	1.112229762683952	inf	{'1': [0.47848160673036816, 173.21065771503658, 0.28055786448984077, 1.3811765157017193, 0.9, 41.86494972856973, 0.0, 0.0]}	[[1.0, 1.56, 0.43345262282721425, 1.241338152936244]]	7.374964118556781
# J2145	0.2957	1e+30			0.0	{'1': [0.2956571434622125, 152.83824159201694, 0.9128913748667467, 0.9899751821983067, 0.15207893203777895, 63.36724045282, 0.0, 0.0]}	[[1.0, 1.56, -0.9248634089774197, 1.8657431068258945]]	
results_file = 'results.dat'
new_entry = f"{system_name}\t{z_lens}\t{chi2_value}\t{pos_rms}\t{mag_rms}\t{avg_percentage_error}\t{lens_params_dict}\t{source_params}\t{td_rms}\t{no_z_lens_flag}\n"

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    existing_entry_index = next((i for i, line in enumerate(lines[1:], start=1) if line.startswith(system_name)), None)
    
    if existing_entry_index is not None:
        existing_chi2 = float(lines[existing_entry_index].split('\t')[2])
        if chi2_value < existing_chi2:
            lines[existing_entry_index] = new_entry
            print(f"Updated entry for {system_name} with better chi^2 value.")
        else:
            print(f"Existing entry for {system_name} has a better chi^2 value. No update made.")
    else:
        lines.append(new_entry)
        print(f"Added new entry for {system_name}.")
    
    with open(results_file, 'w') as f:
        f.writelines(lines)
else:
    with open(results_file, 'w') as f:
        f.write("System\tZ_Lens\tChi2\tPos_RMS\tMag_RMS\tAvg_Flux_Err_%\tLens_Params\tSource_Params\tTD_RMS\tNo_Z_Lens_Flag\n")
        f.write(new_entry)
    print(f"Created {results_file} and added entry for {system_name}.")