import pandas as pd
import os
import re
import numpy as np
import multiprocessing as mp
import corner
import glob
from matplotlib import pyplot as plt
import glafic

# ---------------------------------------------------------
# Helper Functions (From Original Script)
# ---------------------------------------------------------

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

    # Note: Shortened distance matrices from original code as they are identical here 
    # and unnecessary for standard chi2 retrieval
    return np.nan, [], np.nan, [], [], 0, chi2_value, source_params, lens_params_dict, hubble_val, None, None, None, None, out_point

# ---------------------------------------------------------
# Custom MCMC Engine Functions
# ---------------------------------------------------------

def check_bounds(params, bounds):
    """Ensure random walk proposals don't exceed physical boundaries"""
    for p, (low, high) in zip(params, bounds):
        if p < low or p > high:
            return False
    return True

def evaluate_glafic_chi2(params, core_id, sys_data, obs_point, n_img):
    """Evaluates natively within glafic without manual math"""
    v, x, y, e, theta, sx, sy = params
    prefix = f"{sys_data['full_path']}/{sys_data['model_ver']}_core{core_id}"
    
    # 1. Clean output files
    for f in glob.glob(prefix + "*"):
        try: os.remove(f)
        except OSError: pass
            
    # 2. Init glafic 
    glafic.init(0.3, 0.7, -1.0, 0.7, prefix, sys_data['x_min'], sys_data['y_min'], 
                sys_data['x_max'], sys_data['y_max'], 0.01, 0.01, 1, verb=0)
    
    glafic.set_secondary('chi2_splane 1', verb=0)
    glafic.set_secondary('chi2_checknimg 0', verb=0)
    glafic.set_secondary('chi2_restart -1', verb=0)
    glafic.set_secondary('chi2_usemag 0', verb=0)
    glafic.set_secondary('hvary 0', verb=0)
    glafic.set_secondary(f'ran_seed -122000', verb=0) 

    glafic.startup_setnum(1, 0, 1)
    glafic.set_lens(1, 'sie', sys_data['z_lens'], v, x, y, e, theta, 0.0, 0.0)
    glafic.set_point(1, sys_data['z_source'], sx, sy)

    # CRITICAL: Disable parameters internally being manipulated by glafic optimize module!
    # By setting flags to 0, glafic will just evaluate our parameters and export the chi2 cleanly.
    glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
    glafic.setopt_point(1, 0, 0, 0)

    glafic.model_init(verb=0)
    glafic.readobs_point(sys_data['obs_file'])
    
    glafic.optimize() # Generates _optresult.dat (contains chi2)
    glafic.findimg()  # Generates _point.dat
    glafic.quit()
    
    # 3. Use the extraction code exactly as requested to grab chi2 calculation
    try:
        model_ver_core = f"{sys_data['model_ver']}_core{core_id}"
        # We index [6] to specifically isolate the chi2_value return out of the 15 returns
        chi2_val = rms_extract(model_ver_core, sys_data['full_path'], obs_point, n_img)[6] 
    except Exception:
        chi2_val = np.inf

    # 4. Clean up leftover files
    for f in glob.glob(prefix + "*"):
        try: os.remove(f)
        except OSError: pass

    return chi2_val

def mcmc_chain(core_id, start_params, num_steps, step_sizes, bounds, sys_data, obs_point, n_img):
    """Isolated Metropolis-Hastings Random Walk."""
    np.random.seed()
    chain = []
    accepts = 0
    
    current_params = start_params.copy()
    current_chi2 = evaluate_glafic_chi2(current_params, core_id, sys_data, obs_point, n_img)
    
    # Safety Check: Guarantee a valid starting chi2 calculation
    if current_chi2 is None or np.isinf(current_chi2) or np.isnan(current_chi2):
        current_chi2 = 1e6 
        
    for step in range(num_steps):
        # 1. Propose bounds
        proposal = current_params + np.random.normal(0, step_sizes)
        
        # 2. Early rejection if out-of-range bounds are proposed 
        if not check_bounds(proposal, bounds):
            prop_chi2 = np.inf
        else:
            prop_chi2 = evaluate_glafic_chi2(proposal, core_id, sys_data, obs_point, n_img)
        
        # 3. Metropolis-Hastings Accept/Reject logic
        if prop_chi2 < current_chi2:
            accept = True
        else:
            diff = prop_chi2 - current_chi2
            # Prevent exp overflow with diff limit bounds
            alpha = np.exp(-0.5 * diff) if diff < 100 else 0 
            accept = np.random.rand() < alpha
            
        if accept:
            current_params = proposal
            current_chi2 = prop_chi2
            accepts += 1
            
        chain.append([current_chi2] + list(current_params))
        
    print(f"Core {core_id} finished | Acceptance rate: {accepts / num_steps:.2%}")
    return chain

def calculate_gelman_rubin(chains):
    """Calculates Convergence R-Hat Gelman-Rubin Statistic."""
    chains = np.array(chains)
    num_chains, num_steps, num_params = chains.shape
    R_hats = []
    
    for p in range(num_params):
        chain_means = np.mean(chains[:, :, p], axis=1)
        global_mean = np.mean(chain_means)
        
        # Between-chain variance
        B = num_steps / (num_chains - 1) * np.sum((chain_means - global_mean)**2)
        # Within-chain variance
        chain_vars = np.var(chains[:, :, p], axis=1, ddof=1)
        W = np.mean(chain_vars)
        
        if W == 0:
            R_hats.append(1.0)
        else:
            V_hat = ((num_steps - 1) / num_steps) * W + (1 / num_steps) * B
            R_hats.append(np.sqrt(V_hat / W))
            
    return R_hats

# ---------------------------------------------------------
# Main Execution Entrypoint 
# ---------------------------------------------------------
if __name__ == '__main__':
    system_name = '2M1134'
    model_ver = 'sie'
    current_path = os.getcwd()
    full_path = os.path.join(current_path, system_name, 'mcmc')
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    z_lens, z_source = read_info(f'./{system_name}')

    obs_point_file = f'./{system_name}/pos+flux_point.dat'
    obs_point = pd.read_csv(obs_point_file, sep=r'\s+', header=None, skiprows=1, 
                            names=['x', 'y', 'mag', 'pos_err', 'mag_err', 'td', 'td_err', 'parity'])
    obs_point['Img'] = None

    brightest_index = obs_point['mag'].idxmax()
    obs_point.at[brightest_index, 'Img'] = 'A'
    assign_image_names(obs_point, brightest_index)

    n_img = len(obs_point)

    x_min, x_max = obs_point['x'].min() - 1, obs_point['x'].max() + 1
    y_min, y_max = obs_point['y'].min() - 1, obs_point['y'].max() + 1
    
    sys_data = {
        'full_path': full_path,
        'model_ver': model_ver,
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_lens': z_lens if z_lens is not None else 0.5,
        'z_source': z_source,
        'obs_file': f'./{system_name}/pos_point.dat'
    }

    # Grab the true optimal parameters baseline
    _, _, _, _, _, _, _, source_params, lens_params_dict, *_ = rms_extract('sie', f'./{system_name}/', obs_point, n_img=n_img)

    start_params = np.array([
        lens_params_dict['sie'][0],   # v
        lens_params_dict['sie'][1],   # x_lens
        lens_params_dict['sie'][2],   # y_lens
        lens_params_dict['sie'][3],   # e
        lens_params_dict['sie'][4],   # theta
        source_params[0][1],          # x_src
        source_params[0][2]           # y_src
    ])

    # Enforce Bounds representing logical geometric boundaries
    param_bounds = [
        (330, 332),         # velocity dispersion
        (x_min, x_max),    # x_lens
        (y_min, y_max),    # y_lens
        (0.8, 0.9),    # ellipticity
        (43, 44),       # theta (PA)
        (x_min, x_max),    # x_src
        (y_min, y_max)     # y_src
    ]

    step_sizes = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    param_names = ['v', 'x_lens', 'y_lens', 'e', 'theta', 'x_src', 'y_src']
    
    # Run Parameters
    TOTAL_ITERATIONS = 50000
    num_cores = 4
    iters_per_core = max(1, TOTAL_ITERATIONS // num_cores)

    print(f'Starting Multithreaded MCMC on {num_cores} cores ({iters_per_core} iterations/core)...')

    args = [(i, start_params, iters_per_core, step_sizes, param_bounds, sys_data, obs_point, n_img) for i in range(num_cores)]
    
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(mcmc_chain, args)

    # Formatting Results Matrix
    chains_array = np.array(results) 
    
    # Discard early burn-in iterations (20%) to let convergence happen smoothly
    burn_in_idx = int(0.2 * iters_per_core)
    burned_chains = chains_array[:, burn_in_idx:, :]
    
    # Calculate Gelman-Rubin Statistic Convergence (R_hat ≈ 1.0 is converged optimally)
    r_hats = calculate_gelman_rubin(burned_chains[:, :, 1:]) # Index [1:] excludes the chi2 stat
    print("\n--- MCMC Convergence Values ---")
    for name, r in zip(param_names, r_hats):
        print(f"{name:>8} R_hat = {r:.4f} {'(Converged)' if r < 1.1 else '(Poorly Converged)'}")
    
    # Extract flattened data specifically formatted for standard data exports
    mcmc_data_list = burned_chains.reshape(-1, len(param_names) + 1)
    mcmc_data = pd.DataFrame(mcmc_data_list, columns=['chi2'] + param_names)
    
    mcmc_output_file = f'./{system_name}/mcmc/{model_ver}_mcmc.dat'
    mcmc_data.to_csv(mcmc_output_file, sep='\t', index=False, header=False) 

    # Visually mapping results via Corner
    corner.corner(mcmc_data[param_names], labels=param_names, show_titles=True, title_fmt='.2f')
    corner_plot_path = f'./{system_name}/mcmc/{model_ver}_corner.png'
    plt.savefig(corner_plot_path)
    print(f'\nCorner plot saved to: {corner_plot_path}')