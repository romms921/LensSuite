import os
import re
import glob
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

WORKSPACE_DIR = "/Users/ainsleylewis/Documents/Astronomy/LensSuite"

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

def get_fov_from_obs(obs_file):
    x_coords, y_coords = [], []
    with open(obs_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: # skip first line which is header
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
                except ValueError:
                    pass
    if not x_coords:
        return -5.0, -5.0, 5.0, 5.0
    xmin = min(x_coords) - 1.0
    xmax = max(x_coords) + 1.0
    ymin = min(y_coords) - 1.0
    ymax = max(y_coords) + 1.0
    return xmin, ymin, xmax, ymax

def generate_glafic_script(particle, z_lens, z_source, obs_file, fov, prefix, script_path, use_amoeba=False, finalize=False):
    vdisp, x, y, ell, the, sx, sy = particle
    xmin, ymin, xmax, ymax = fov
    script_lines = [
        f"import glafic",
        f"glafic.init(0.3, 0.7, -1.0, 0.7, '{prefix}', {xmin}, {ymin}, {xmax}, {ymax}, 0.001, 0.001, 1, verb=0)",
        f"glafic.set_secondary('chi2_splane 0', verb=0)",
        f"glafic.set_secondary('chi2_checknimg 0', verb=0)",
        f"glafic.set_secondary('chi2_restart -1', verb=0)",
        f"glafic.set_secondary('chi2_usemag 1', verb=0)",
        f"glafic.set_secondary('hvary 0', verb=0)",
        f"glafic.set_secondary('ran_seed -122000', verb=0)",
        f"glafic.startup_setnum(1, 0, 1)",
        f"glafic.set_lens(1, 'sie', {z_lens}, {vdisp}, {x}, {y}, {ell}, {the}, 0.0, 0.0)",
        f"glafic.set_point(1, {z_source}, {sx}, {sy})"
    ]
    
    script_lines.append(f"glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)")
    script_lines.append(f"glafic.setopt_point(1, 0, 0, 0)")
    script_lines.append(f"glafic.model_init()")
    script_lines.append(f"glafic.readobs_point('{obs_file}')")
    
    script_lines.append(f"try:")
    script_lines.append(f"    chi2 = glafic.c2calc()")
    script_lines.append(f"except Exception:")
    script_lines.append(f"    chi2 = float('inf')")

    if finalize:
        # Mimic Glafic's optresult.dat structure so `rms_extract` can parse it correctly.
        script_lines.append(f"print('------------------------------------------')")
        script_lines.append(f"print('optimize ndim=0')")
        script_lines.append(f"print('run 1: 1 lens models calculated')")
        script_lines.append(f"print(f'chi^2 = {{chi2:e}}  [N_data(extend): 0]')")
        script_lines.append(f"print('------------------------------------------')")
        script_lines.append(f"print('lens 1 sie {z_lens} {vdisp} {x} {y} {ell} {the} 0.0 0.0')")
        script_lines.append(f"print('point 1 {z_source} {sx} {sy}')")
        script_lines.append(f"glafic.findimg()")
        script_lines.append(f"glafic.writecrit(1.0)")
        script_lines.append(f"glafic.writelens(1.0)")
    else:
        script_lines.append(f"print(f'chi^2 = {{chi2:e}}')")

    script_lines.append(f"glafic.quit()")
    
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines) + '\n')

def extract_chi2_from_output(output_text):
    match = re.search(r'chi\^2\s*=\s*([0-9\.e[+-]+)', output_text)
    if match:
        return float(match.group(1))
    return np.inf

class PSO:
    def __init__(self, bounds, num_particles=20):
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (num_particles, len(bounds)))
        self.velocities = np.random.uniform(-1, 1, (num_particles, len(bounds)))
        self.pbest_pos = np.copy(self.positions)
        self.pbest_scores = np.full(num_particles, np.inf)
        self.gbest_pos = None
        self.gbest_score = np.inf

    def update(self, scores):
        for i in range(self.num_particles):
            if scores[i] <= self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest_pos[i] = np.copy(self.positions[i])
            if self.gbest_pos is None or scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest_pos = np.copy(self.positions[i])
        
        w, c1, c2 = 0.5, 1.5, 1.5
        r1, r2 = np.random.rand(2)
        
        for i in range(self.num_particles):
            self.velocities[i] = (w * self.velocities[i] + 
                                  c1 * r1 * (self.pbest_pos[i] - self.positions[i]) + 
                                  c2 * r2 * (self.gbest_pos - self.positions[i]))
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])

def calculate_distance(p1, p2):
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

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
            lens_params_dict[parts[1]] = [float(x) for x in parts[3:]]
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

def process_system(system_dir):
    system_name = os.path.basename(system_dir)
    z_lens, z_source = read_info(system_dir)
    
    # We strictly need z_source, but z_lens can be evaluated dynamically
    if z_source is None:
        return None

    obs_files = glob.glob(os.path.join(system_dir, '*point.dat'))
    if not obs_files:
        return None
    obs_file = obs_files[0]
    
    obs_df = pd.read_csv(obs_file, sep=r'\s+', skiprows=1, header=None)
    num_cols = len(obs_df.columns)
    base_cols = ['x', 'y', 'mag', 'td', 'err_pos', 'err_mag', 'err_td']
    
    if num_cols > len(base_cols):
        extra_cols = [f'col{i}' for i in range(len(base_cols) + 1, num_cols + 1)]
        base_cols.extend(extra_cols)
        
    obs_df.columns = base_cols[:num_cols]
    
    for col in ['x', 'y', 'mag', 'td', 'err_pos', 'err_mag', 'err_td']:
        if col not in obs_df.columns:
            obs_df[col] = 0.0
            
    obs_df['Img'] = [chr(65+i) for i in range(len(obs_df))] # Image flags A, B, C...

    fov = get_fov_from_obs(obs_file)
    xmin, ymin, xmax, ymax = fov
    
    variable_z_lens = (z_lens is None)
    
    # PSO setup (7 dims naturally, 8 dims if z_lens is missing)
    bounds = [
        [10.0, 400.0], [xmin, xmax], [ymin, ymax],
        [0.0, 0.9], [-90.0, 90.0], [xmin, xmax], [ymin, ymax]
    ]
    
    # Add constraint bounds for z_lens to be between 0.01 and z_source
    if variable_z_lens:
        z_max = max(0.02, z_source - 0.01) # Failsafe limit
        bounds.append([0.01, z_max])
        
    pso = PSO(bounds, num_particles=20)
    num_iterations = 1000
    
    script_path = os.path.join(system_dir, f"temp_glafic_{system_name}.py")
    prefix = os.path.join(system_dir, 'SIE')

    for i in range(num_iterations):
        scores = []
        for p_idx in range(pso.num_particles):
            particle = pso.positions[p_idx]
            current_z_lens = particle[7] if variable_z_lens else z_lens
            generate_glafic_script(particle[:7], current_z_lens, z_source, obs_file, fov, prefix, script_path, use_amoeba=False)
            
            result = subprocess.run(["python3", script_path], capture_output=True, text=True)
            chi2 = extract_chi2_from_output(result.stdout)
            if chi2 > 1e10 or np.isnan(chi2):
                chi2 = np.inf
            scores.append(chi2)
        pso.update(scores)

    # Finalize with Best params
    best_particle = pso.gbest_pos
    best_z_lens = best_particle[7] if variable_z_lens else z_lens
    
    generate_glafic_script(best_particle[:7], best_z_lens, z_source, obs_file, fov, prefix, script_path, use_amoeba=False, finalize=True)
    final_res = subprocess.run(["python3", script_path], capture_output=True, text=True)

    optresult_path = f"{prefix}_optresult.dat"
    with open(optresult_path, "w") as f:
        f.write(final_res.stdout)
    
    if os.path.exists(script_path):
        os.remove(script_path)

    td_present = (obs_df['td'].abs().sum() > 0)
    try:
        ext_results = rms_extract('SIE', system_dir, obs_df, n_img=len(obs_df), time_delay=td_present)
        (pos_rms, image_rms, mag_rms, flux_rms, percentage_errors, avg_percentage_error,
         chi2_value, source_params, lens_params_dict, hubble_val, td_vals, td_rms,
         percentage_errors_td, avg_percentage_error_td, out_point) = ext_results

        return {
            'System': system_name,
            'Z_Lens': round(best_z_lens, 4),
            'Chi2': chi2_value,
            'Pos_RMS': pos_rms,
            'Mag_RMS': mag_rms,
            'Avg_Flux_Err_%': avg_percentage_error,
            'Lens_Params': str(lens_params_dict),
            'Source_Params': str(source_params),
            'TD_RMS': td_rms if td_present else 'N/A'
        }
    except Exception as e:
        return {
            'System': system_name,
            'Z_Lens': round(best_z_lens, 4),
            'Chi2': pso.gbest_score,
            'Pos_RMS': 'Error',
            'Mag_RMS': 'Error',
            'Avg_Flux_Err_%': 'Error',
            'Lens_Params': str(pso.gbest_pos[:5]),
            'Source_Params': str(pso.gbest_pos[5:7]),
            'TD_RMS': 'Error'
        }

if __name__ == '__main__':
    systems = []
    for d in os.listdir(WORKSPACE_DIR):
        full_path = os.path.join(WORKSPACE_DIR, d)
        if os.path.isdir(full_path) and d != 'Python Files':
            if os.path.exists(os.path.join(full_path, 'info.dat')):
                systems.append(full_path)

    print(f"Found {len(systems)} systems to optimize.")
    
    with Pool(6) as p:
        results = list(tqdm(p.imap(process_system, systems), total=len(systems), desc="Optimizing Systems", unit="sys"))

    valid_results = [r for r in results if r is not None]
    if valid_results:
        results_df = pd.DataFrame(valid_results)
        output_file = os.path.join(WORKSPACE_DIR, 'results.dat')
        results_df.to_csv(output_file, sep='\t', index=False)
        print(f"\nOptimization complete! Extracted information successfully saved to: {output_file}")