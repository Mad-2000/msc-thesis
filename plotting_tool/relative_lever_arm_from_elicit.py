#%% Imports

import gzip
import pickle
import numpy as np

import json
from scipy.optimize import curve_fit
from scipy.ndimage import sobel

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from elicit.utils import el2np, sort_pointcloud
from elicit.plotting.pyqt import ScatterWidget


#%% Matplotlib


plt.rcParams.update({'font.size': 20})
RWTHblue = (0,103/255,166/255)
RWTHred = (161/255,16/255,53/255)
RWTHgreen = (87/255,171/255,39/255)
RWTHlightblue = (119/255,158/255,201/255)

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.minor.size'] = 8
matplotlib.rcParams['xtick.minor.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.minor.size'] = 8
matplotlib.rcParams['ytick.minor.width'] = 2
plt.rcParams.update({'font.size': 20})

# Load the gzipped data
def load_gz_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_labels_from_filename(file_path):
    filename = os.path.basename(file_path).replace('.gz', '')  # Get filename and remove extension
    try:
        parts = filename.split(' ')
        if 'vs' in parts:
            vs_index = parts.index('vs')
            x_label = parts[vs_index - 1]
            y_label = parts[vs_index + 1]
            return x_label, y_label
        else:
            return "X-axis", "Y-axis"  # Default fallback labels
    except Exception as e:
        print(f"Error extracting labels: {e}")
        return "X-axis", "Y-axis"

def plot_single_elicit_gz_file(file_path):
    """
    Load and plot a .gz file (elicit format) just like the GUI.
    Plots 2D voltage sweeps with proper voltage axes.
    """
    raw_data = load_gz_data(file_path)
    structured_data = el2np(raw_data)

    for key, value in structured_data.items():
        data = value['data']
        dims = value['dims']

        if len(dims) != 2:
            print(f"Skipping {key}: not a 2D dataset.")
            continue

        x_dim, y_dim = list(dims.keys())
        x_grid = np.array(dims[x_dim])
        y_grid = np.array(dims[y_dim])

        # Get 1D voltage axes from meshgrid
        x_vals = x_grid[:, 0]
        y_vals = y_grid[0, :]

        # Confirm matching dimensions
        if data.shape != (len(x_vals), len(y_vals)):
            print(f"Skipping {key}: shape mismatch.")
            continue

        # Plot
        plt.figure(figsize=(7, 6))
        plt.imshow(data, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
                   origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Current (nA)')
        plt.xlabel(f"{x_dim} (V)", fontsize=14)
        plt.ylabel(f"{y_dim} (V)", fontsize=14)
        plt.title(f"2D Plot - {key[0]}", fontsize=16)
        # plt.tight_layout()
        plt.show()

def process_and_plot_gz_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    # List all .gz files in the folder
    gz_files = [f for f in os.listdir(input_folder) if f.endswith('.gz')]
    
    for file_name in gz_files:
        file_path = os.path.join(input_folder, file_name)
        print(f"Processing: {file_name}")

        try:
            loaded_data = load_gz_data(file_path)
            data = el2np(loaded_data)  # Convert data using el2np

            x_label_from_file, y_label_from_file = extract_labels_from_filename(file_path)

            for i, (key, value) in enumerate(data.items()):
                try:
                    dims = value['dims']
                    plot_data = value['data']

                    # Check for 2D data
                    if len(dims) == 2:
                        plt.figure(figsize=(6, 5))
                        x_dim, y_dim = list(dims.keys())
                        x_data = dims[x_dim]
                        y_data = dims[y_dim]

                        # Plotting using imshow for heatmap visualization
                        plt.imshow(
                            plot_data.T * 1e9,  # Convert to nA
                            extent=[x_data.min(), x_data.max(), y_data.min(), y_data.max()],
                            origin='lower',
                            aspect='auto',
                            cmap='viridis'
                        )

                        # Use extracted labels for x and y axes
                        plt.xlabel(f"{x_label_from_file} (V)", fontsize=20)
                        plt.ylabel(f"{y_label_from_file} (V)", fontsize=20)

                        # Title with data key
                        # plt.title(f"2D Plot {i + 1}: {key[0]}")
                        plt.colorbar(label="Current (nA)")

                        # Save the plot instead of showing it
                        output_filename = f"{file_name.replace('.gz', '')}_plot_{i+1}.pdf"
                        output_path = os.path.join(output_folder, output_filename)
                        plt.savefig(output_path, 
                                    format='pdf', 
                                    bbox_inches='tight')
                        plt.close()
                        print(f"Saved: {output_path}")

                    else:
                        print(f"Skipping {file_name} Plot {i+1} due to unsupported dimensions: {dims.keys()}")

                except Exception as e:
                    print(f"Error processing plot {i+1} for {file_name}: {e}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

def plot_elicit_gz_file_with_correct_axis(file_path, 
                                          base_x=0.897216796875, 
                                          base_y=0.897216796875,
                                          output_folder=None,
                                          ylim = None,
                                          xlim = None
                                          ):
    """
    Load and plot a .gz file (elicit format) using real voltages based on base_x/y offsets.
    
    Args:
        file_path (str): Path to the .gz file
        base_x (float): Absolute voltage added to x offset
        base_y (float): Absolute voltage added to y offset
    """
    raw_data = load_gz_data(file_path)
    structured_data = el2np(raw_data)
    x_label_from_file, y_label_from_file = extract_labels_from_filename(file_path)

    for key, value in structured_data.items():
        data = value['data']
        dims = value['dims']

        if len(dims) != 2:
            print(f"Skipping {key}: not a 2D dataset.")
            continue

        x_dim, y_dim = list(dims.keys())
        x_grid = np.array(dims[x_dim])
        y_grid = np.array(dims[y_dim])

        # Apply base voltages to offset grid
        real_x_grid = x_grid + base_x
        real_y_grid = y_grid + base_y

        # Get 1D axes
        y_vals = real_y_grid[:, 0]
        x_vals = real_x_grid[0, :]

        # Confirm matching dimensions
        if data.shape != (len(y_vals), len(x_vals)):
            print(f"Skipping {key}: shape mismatch.")
            continue

        # Plot
        plt.figure(figsize=(6, 4))
        # plt.imshow(data*1e9, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        #            origin='lower', aspect='auto', cmap='viridis')
        plt.pcolormesh(real_x_grid, real_y_grid, data*1e12,
               shading='auto', cmap='viridis')

        plt.colorbar(label=' Transport Current (pA)')
        plt.xlabel(f"{x_label_from_file} (V)", fontsize=20)
        plt.ylabel(f"{y_label_from_file} (V)", fontsize=20)
        # plt.xlabel(f"Top Barrier (V)", fontsize=20)
        # plt.ylabel(f"Bottom Barrier (V)", fontsize=20)
        # plt.title(f"2D Plot - {key[0]}", fontsize=16)
        # plt.minorticks_on()
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)


        # Build filename only if output_folder is provided
        if output_folder:
            sweep_base = os.path.basename(file_path).replace('.gz', '')
            sweep_name = sweep_base.split()[1]+ "_"+ sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
            output_filename = f"{sweep_name}_rightaxis_plot2.pdf"
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved as: {output_path}")
        plt.show()
    return real_x_grid, real_y_grid, data

def save_xy_z_json(x_grid, y_grid, z_data, output_path='export.json'):
    data_dict = {
        'x': x_grid.tolist(),
        'y': y_grid.tolist(),
        'z': z_data.tolist()
    }
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved to {output_path}")


#%%  Analysis

def extract_edge_x(masked_data, X, Y):
    thresholds = []
    found = False
    for ii in range(len(X)-1):
        found = False
        for jj in range(len(Y)-1):
            if found: continue
            if np.isnan(masked_data[ii, jj]):
                if np.isnan(masked_data[ii+1, jj]): continue
                else: 
                    # thresholds.append([int(ii),X[ii]])
                    thresholds.append([X[ii],Y[jj]])
                    found = True
                        
    return np.array(thresholds)

def extract_edge_y(masked_data, X, Y):
    thresholds = []
    found = False
    for jj in range(len(Y)-1):
        found = False
        for ii in range(len(X)-1):
            if found: continue
            if np.isnan(masked_data[ii,jj]):
                if np.isnan(masked_data[ii,jj+1]): continue
                else: 
                    # thresholds.append([int(jj),Y[jj]])
                    thresholds.append([X[ii],Y[jj]])
                    found = True
                        
    return np.array(thresholds)



#%% Plot extraction of whole gz files in a folder (x offset and y offset)

# Define input and output folders
input_folder = r"z:\Data\Three layer SET\Tuning toolkit\Data\Batch 26\S14_C2"  # Replace with your actual input folder path
output_folder =  r"z:\Data\Plotting\Plots\with_mesa\S14_C2" # Replace with your actual output folder path

# Run the function to process and save plots
process_and_plot_gz_files(input_folder, output_folder)


#%% Sample relative lever arm extraction of LTB vs LBB - Better 2D plot and saving data as json for further processing 

#  Here LTB vs LBB is used given as a reference
file_path = r"z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 11_50_15_12 LTB vs LBB.gz"
output_folder =  r"z:\Data\Plotting\Plots\Sample Plots"

# Extract base name and sweep labels
sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
sweep_name = f"{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
json_filename = f"{sweep_name}.json"
json_output_path = os.path.join(output_folder, json_filename)


xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
                                 base_x=         0.897216796875,
                                 base_y =        0.897216796875,
                                 output_folder = output_folder,
# )
                                 xlim= [0.79,1.0] )


save_xy_z_json(xg, yg, zg, output_path=json_output_path)


# %% Loading json and masking the 2D data
json_filename = r"z:\Data\Plotting\Plots\lever_arm_extraction\LBBvsLTB.json"

with open(json_filename, "r") as file:
        data = json.load(file)
fig, ax = plt.subplots(1,1, figsize=(6,5))

x_label_from_file, y_label_from_file = extract_labels_from_filename(file_path)

X = np.array(data['x'])[3:-3, 3:-3]
Y = np.array(data['y'])[3:-3, 3:-3]

masked_z = np.array(data['z'])[3:-3, 3:-3]
masked_z[masked_z < 2e-11 ] = np.nan
masked_z = np.gradient(masked_z)[0]

ax.pcolormesh(X, Y, masked_z)
ax.minorticks_on()
ax.grid()


#%%  Saving linefit
thresholds_x = extract_edge_x(masked_z, X[:,0], Y[0,:])
thresholds_y = extract_edge_y(masked_z, X[:,0], Y[0,:])

plt.pcolormesh(X, Y, masked_z)
# plt.pcolormesh(X, Y, data['z'])
plt.scatter(thresholds_x[:,0], thresholds_x[:,1], color='red')
plt.scatter(thresholds_y[:,0], thresholds_y[:,1], color='blue')

plt.xlabel(f"{x_label_from_file} (V)", fontsize=20)
plt.ylabel(f"{y_label_from_file} (V)", fontsize=20)
plt.grid(which='major', linestyle='--', color='gray')
plt.grid(which='minor', linestyle=':', color='lightgray')
plt.minorticks_on()

sweep_base = os.path.basename(file_path).replace('.gz', '')
sweep_name = sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
output_filename = f"{sweep_name}_linefit_plot_new.pdf"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='pdf', bbox_inches='tight')

#%% Saving extracted relative lever arm data

# Manually inputing the corner from the edge plot
corner = [0.87, 0.82] 


# y_label_from_file = 'LTA'
idx = thresholds_x[:,1] < corner[0]
idy = thresholds_y[:,0] < corner[1]

p_x = np.polyfit(thresholds_x[idx,0], thresholds_x[idx,1], deg=1)
p_y = np.polyfit(thresholds_y[idy,1], thresholds_y[idy,0], deg=1)


plt.scatter(thresholds_x[idx,0], thresholds_x[idx,1], color='red')
plt.scatter(thresholds_y[idy,1], thresholds_y[idy,0], color='blue')
plt.plot(thresholds_x[idx,0], np.polyval(p_x, thresholds_x[idx,0]), color='red')
plt.plot(thresholds_y[idy,1], np.polyval(p_y, thresholds_y[idy,1]), color='blue')

plt.ylabel(f"{y_label_from_file} (V)")
plt.xlabel(f"{x_label_from_file} (V)")
plt.legend([
    f'{x_label_from_file}/{y_label_from_file}: {abs(p_x[0]):.3f} V/V',
    f'{y_label_from_file}/{x_label_from_file}: {abs(p_y[0]):.3f} V/V'
])

plt.grid(True)
plt.minorticks_on()
plt.grid(which='major', linestyle='--', color='gray')
plt.grid(which='minor', linestyle=':', color='lightgray')
sweep_base = os.path.basename(file_path).replace('.gz', '')
sweep_name = sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
output_filename = f"{sweep_name}_leverarmm_plot.pdf"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.show()

#%% Plotting the fitted line directly to the 2D plot

json_filename = r"z:\Data\Plotting\Plots\lever_arm_extraction\LBBvsLTB.json"

with open(json_filename, "r") as file:
        data = json.load(file)
fig, ax = plt.subplots(1,1, figsize=(6,5))

x_label_from_file, y_label_from_file = extract_labels_from_filename(file_path)

X = np.array(data['x'])[3:-3, 3:-3]
Y = np.array(data['y'])[3:-3, 3:-3]
masked_z = np.array(data['z'])[3:-3, 3:-3]

thresholds_x = extract_edge_x(masked_z, X[:,0], Y[0,:])
thresholds_y = extract_edge_y(masked_z, X[:,0], Y[0,:])

# Manually inputing the corner from the edge plot
corner = [0.87, 0.82] 

idx = thresholds_x[:,1] < corner[0]
idy = thresholds_y[:,0] < corner[1]

p_x = np.polyfit(thresholds_x[idx,0], thresholds_x[idx,1], deg=1)
p_y = np.polyfit(thresholds_y[idy,1], thresholds_y[idy,0], deg=1)


plt.pcolormesh(X, Y, masked_z)

plt.scatter(thresholds_x[idx,0], thresholds_x[idx,1], color='red')
plt.scatter( thresholds_y[idy,0],thresholds_y[idy,1], color='blue')

plt.plot(thresholds_x[idx,0], np.polyval(p_x, thresholds_x[idx,0]), color='red')
plt.plot( np.polyval(p_y, thresholds_y[idy,1]), thresholds_y[idy,1],color='blue')

plt.ylabel(f"{y_label_from_file} (V)", fontsize =20)
plt.xlabel(f"{x_label_from_file} (V)",  fontsize =20)
plt.legend([
    f'{x_label_from_file}/{y_label_from_file}: {abs(p_x[0]):.3f} V/V',
    f'{y_label_from_file}/{x_label_from_file}: {abs(p_y[0]):.3f} V/V'
])
plt.minorticks_on()
plt.savefig(r"z:\Data\Plotting\Plots\lever_arm_extraction\leverarm_LBB_LTB_thesis.pdf", 
            format='pdf', 
            bbox_inches='tight')
plt.show()

#%% #########  RLA of BS vs LP #############
json_filename = r"z:\Data\Plotting\Plots\lever_arm_extraction\BsvsLP.json"

with open(json_filename, "r") as file:
        data = json.load(file)
fig, ax = plt.subplots(1,1, figsize=(6,5))

x_label_from_file, y_label_from_file = extract_labels_from_filename(file_path)

X = np.array(data['x'])[3:-3, 3:-3]
Y = np.array(data['y'])[3:-3, 3:-3]

masked_z = np.array(data['z'])[3:-3, 3:-3]
masked_z[masked_z < 4e-11 ] = np.nan
masked_z = np.gradient(masked_z)[0]

ax.pcolormesh(masked_z)
# ax.pcolormesh(data['z'])

#%%  Saving linefit
thresholds_x = extract_edge_x1(masked_z, X[:,0], Y[0,:])
thresholds_y = extract_edge_y1(masked_z, X[:,0], Y[0,:])

plt.pcolormesh(X, Y, masked_z)

plt.scatter(thresholds_x[:,0], thresholds_x[:,1], color='red')
plt.scatter(thresholds_y[:,0], thresholds_y[:,1], color='blue')

plt.xlabel(f"{x_label_from_file} (V)", fontsize=20)
plt.ylabel(f"{y_label_from_file} (V)", fontsize=20)
plt.minorticks_on()
plt.grid()
sweep_base = os.path.basename(file_path).replace('.gz', '')

sweep_name = sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
output_filename = f"{sweep_name}_linefit_plot_new.pdf"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='pdf', bbox_inches='tight')

#%% Saving fit leverarm data
corner = [-0.02, 0.85] # BSvsLP
# y_label_from_file = 'LTA'
idx = thresholds_x[:,1] < corner[0]
idy = thresholds_y[:,0] < corner[1]

p_x = np.polyfit(thresholds_x[idx,0], thresholds_x[idx,1], deg=1)
p_y = np.polyfit(thresholds_y[idy,1], thresholds_y[idy,0], deg=1)

plt.scatter(thresholds_x[idx,0], thresholds_x[idx,1], color='red')
plt.scatter(thresholds_y[idy,1], thresholds_y[idy,0], color='blue')
plt.plot(thresholds_x[idx,0], np.polyval(p_x, thresholds_x[idx,0]), color='red')
plt.plot(thresholds_y[idy,1], np.polyval(p_y, thresholds_y[idy,1]), color='blue')

# plt.xlabel(f"{x_label_from_file} (V)")
plt.ylabel(f"{y_label_from_file} (V)")
plt.xlabel(f"{x_label_from_file} (V)")
# plt.ylim([0.7,0.9])
# plt.ylabel("LBA (V)")
plt.legend([f'{x_label_from_file}/{y_label_from_file}: {abs(p_x[0]):.3f} V/V',
            f'{y_label_from_file}/{x_label_from_file}: {abs(p_y[0]):.3f} V/V'])
plt.grid(True)
plt.minorticks_on()
plt.grid(which='major', linestyle='--', color='gray')
plt.grid(which='minor', linestyle=':', color='lightgray')
sweep_base = os.path.basename(file_path).replace('.gz', '')
sweep_name = sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
output_filename = f"{sweep_name}_leverarmm_plot.pdf"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.show()


#%% ############# RLA of BS vs LP ###############

json_filename = r"z:\Data\Plotting\Plots\lever_arm_extraction\BsvsLP.json"

with open(json_filename, "r") as file:
        data = json.load(file)
fig, ax = plt.subplots(1,1, figsize=(6,5))

x_label_from_file, y_label_from_file = extract_labels_from_filename(file_path)

X = np.array(data['x'])[3:-3, 3:-3]
Y = np.array(data['y'])[3:-3, 3:-3]

masked_z = np.array(data['z'])[3:-3, 3:-3]
masked_z[masked_z < 4e-11 ] = np.nan
masked_z = np.gradient(masked_z)[0]

ax.pcolormesh(masked_z)
# ax.pcolormesh(data['z'])

#%%  Saving linefit
thresholds_x = extract_edge_x1(masked_z, X[:,0], Y[0,:])
thresholds_y = extract_edge_y1(masked_z, X[:,0], Y[0,:])

plt.pcolormesh(X, Y, masked_z)

plt.scatter(thresholds_x[:,0], thresholds_x[:,1], color='red')
plt.scatter(thresholds_y[:,0], thresholds_y[:,1], color='blue')

plt.xlabel(f"{x_label_from_file} (V)", fontsize=20)
plt.ylabel(f"{y_label_from_file} (V)", fontsize=20)
plt.minorticks_on()
plt.grid()
sweep_base = os.path.basename(file_path).replace('.gz', '')

sweep_name = sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
output_filename = f"{sweep_name}_linefit_plot_new.pdf"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='pdf', bbox_inches='tight')

#%% Saving fit leverarm data
corner = [-0.02, 0.85] # BSvsLP
# y_label_from_file = 'LTA'
idx = thresholds_x[:,1] < corner[0]
idy = thresholds_y[:,0] < corner[1]

p_x = np.polyfit(thresholds_x[idx,0], thresholds_x[idx,1], deg=1)
p_y = np.polyfit(thresholds_y[idy,1], thresholds_y[idy,0], deg=1)

plt.scatter(thresholds_x[idx,0], thresholds_x[idx,1], color='red')
plt.scatter(thresholds_y[idy,1], thresholds_y[idy,0], color='blue')
plt.plot(thresholds_x[idx,0], np.polyval(p_x, thresholds_x[idx,0]), color='red')
plt.plot(thresholds_y[idy,1], np.polyval(p_y, thresholds_y[idy,1]), color='blue')

# plt.xlabel(f"{x_label_from_file} (V)")
plt.ylabel(f"{y_label_from_file} (V)")
plt.xlabel(f"{x_label_from_file} (V)")
# plt.ylim([0.7,0.9])
# plt.ylabel("LBA (V)")
plt.legend([f'{x_label_from_file}/{y_label_from_file}: {abs(p_x[0]):.3f} V/V',
            f'{y_label_from_file}/{x_label_from_file}: {abs(p_y[0]):.3f} V/V'])
plt.grid(True)
plt.minorticks_on()
plt.grid(which='major', linestyle='--', color='gray')
plt.grid(which='minor', linestyle=':', color='lightgray')
sweep_base = os.path.basename(file_path).replace('.gz', '')
sweep_name = sweep_base.split()[-1] + 'vs' + sweep_base.split()[-3]
output_filename = f"{sweep_name}_leverarmm_plot.pdf"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.show()













#%% 1D sweep from elicit 

# output_folder = r"Z:\Data\Three layer SET\24.01.25_TTF+elicit\Plots\S14_B4\elicit_plot"
file_path = r"Z:\Data\Three layer SET\24.01.25_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 16_34_58_06 LP.gz"
raw = load_gz_data(file_path)
data_dict = el2np(raw)
# x_lab, y_lab = extract_labels_from_filename(file_path)
base_x = 0.762939453125
chan = data_dict[('SET_L', 'x offset', 'y offset')]

x = np.asarray(chan['dims']['x offset']).flatten()
real_x_grid = x + base_x     # shape (N,)
y = chan['data'].flatten() * 1e12                        # convert A → nA

plt.figure(figsize=(6,4))
plt.plot(real_x_grid, y, '-', lw=1.2)
plt.xlabel('LP (V)')
plt.ylabel('Transport Current (pA)')
plt.xlim([0.65, 0.85])

plt.grid(True)
plt.minorticks_on()
plt.grid(which='major', linestyle='--', color='gray')
plt.grid(which='minor', linestyle=':', color='lightgray')
plt.savefig(rf"z:\Data\Plotting\Plots\with_mesa\S14_B4_spuriousdot_Plungersweep_{base_x}5.pdf", 
            format='pdf', 
            dpi = 500)
plt.show()
            


#%% ################## Coulomb Oscillation plots of the spurios dot for S14_B4 device #####################
# file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 12_29_44_69 LP vs LS.gz"
# output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# # Extract base name and sweep labels
# sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
# parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
# sweep_name = f"{parts[1]}_{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
# json_filename = f"{sweep_name}.json"
# json_output_path = os.path.join(output_folder, json_filename)


# xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
#                                  base_x=     0.69976806640625,  # VOLTAGE VALUES WERE EXTRACTED FROM ELICIT GUI
#                                  base_y =       -0.10009765625,
#                                  output_folder = output_folder,
#                                  xlim=[0.6,None])



# file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 13_32_10_61 LTA vs LTB.gz"
# output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# # Extract base name and sweep labels
# sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
# parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
# sweep_name = f"{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
# json_filename = f"{sweep_name}.json"
# json_output_path = os.path.join(output_folder, json_filename)


# xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
#                                  base_x=        0.7598876953125,
#                                  base_y =       0.77972412109375,
#                                  output_folder = output_folder,)


# file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 13_10_40_62 LTB vs LBB.gz"
# output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# # Extract base name and sweep labels
# sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
# parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
# sweep_name = f"{parts[1]}_{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
# json_filename = f"{sweep_name}.json"
# json_output_path = os.path.join(output_folder, json_filename)


# xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
#                                  base_x=        0.82489013671875,
#                                  base_y =        0.64971923828125,
#                                  output_folder = output_folder,)

# file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 13_28_31_10 LP vs LBB.gz"
# output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# # Extract base name and sweep labels
# sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
# parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
# sweep_name = f"{parts[1]}_{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
# json_filename = f"{sweep_name}.json"
# json_output_path = os.path.join(output_folder, json_filename)


# xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
#                                  base_x=      0.77972412109375,
#                                  base_y =      0.64971923828125,
#                                  output_folder = output_folder,
#                                  xlim=[0.7,None])


# file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 13_22_36_89 LP vs LTB.gz"
# output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# # Extract base name and sweep labels
# sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
# parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
# sweep_name = f"{parts[1]}_{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
# json_filename = f"{sweep_name}.json"
# json_output_path = os.path.join(output_folder, json_filename)


# xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
#                                  base_x=      0.762939453125,
#                                  base_y =       0.74981689453125,
#                                  output_folder = output_folder,
#                                  xlim=[0.68,None])

# file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 15_02_26_38 LP1 vs LTB.gz"
# output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# # Extract base name and sweep labels
# sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
# parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
# sweep_name = f"{parts[1]}_{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
# json_filename = f"{sweep_name}.json"
# json_output_path = os.path.join(output_folder, json_filename)


# xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
#                                  base_x=     0,
#                                  base_y =       0.77972412109375,
#                                  output_folder = output_folder,
#                                  xlim=[-0.08,None])

file_path = r"Z:\Data\Three layer SET\Tuning toolkit\S14_B4_TTF+elicit\S14_B4_TTF_elicit\2025-01-24 16_17_27_07 LTB vs LBB.gz"
output_folder =  r"z:\Data\Plotting\Plots\Coulomb Oscillations"

# Extract base name and sweep labels
sweep_base = os.path.basename(file_path).replace('.gz', '')  # → "2025-01-24 11_28_17_69 LP vs LS"
parts = sweep_base.split()  # → ['2025-01-24', '11_28_17_69', 'LP', 'vs', 'LS']
sweep_name = f"{parts[1]}_{parts[-1]}vs{parts[-3]}"  # → "LSvsLP"
json_filename = f"{sweep_name}.json"
json_output_path = os.path.join(output_folder, json_filename)


xg, yg, zg = plot_elicit_gz_file_with_correct_axis(file_path,
                                 base_x=    0.625,
                                 base_y =      0.56976318359375,
                                 output_folder = output_folder,
                                 xlim=[0.54,None])
