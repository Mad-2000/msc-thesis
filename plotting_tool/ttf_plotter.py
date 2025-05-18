#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
#%%
import os
os.environ["TTF_BASEPATH"] = ""
import tuning_toolkit.framework as ttf


#%% matplotlib parameters
plt.rcParams.update({'font.size': 20})
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
#%%  simple 1D and 2D plotting
def plot_1d(x, y, xlabel, ylabel="Current (nA)", title="", save_path=None):
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='minor', color='#CCCCCC', linestyle=':')
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_2d(data, x_label="X", y_label="Y", z_label="Current (nA)", title="", save_path=None):
    plt.figure(figsize=(6, 5))
    img = plt.imshow(data.T, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label=z_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

#%% ######## Combined Accumulation plots from a folder ##############
sample_name = "S14_B4"
accum_data = ttf.skeleton.load_all_dataitems_at_path(path = r"z:\Data\Three layer SET\Tuning toolkit\Data\3LSET\With_Mesa\accum_folder", 
                                                         keywords = ["accumulation-measurement"])
sample_name = "with_mesa"
accum_data = ttf.skeleton.load_all_dataitems_at_path(
    path=r"z:\Data\Three layer SET\Tuning toolkit\Data\3LSET\With_Mesa\accum_folder",
    keywords=["accumulation-measurement"]
)

# OPTIONAL: Custom device labels (must match number of data items)
device_labels = [  # You can manually modify this list
    "Dev 25", "Dev 26", "Dev B1", "Dev B2", "Dev C1"
][:len(accum_data)]  # Trim to actual number of entries

# === Create single plot
plt.figure(figsize=(6, 5))
ax = plt.gca()

# === Plot loop
for i, data_item in enumerate(accum_data):
    if "accum_data" in data_item.data:
        y = data_item.data["accum_data"].values[0] * 1e12  # Convert to pA
        x = data_item.data["accum_data"].coords["virtual_gate"].values

        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        label = device_labels[i] if i < len(device_labels) else f"Dev {i+1}"
        ax.plot(x, y, marker ='o',label=label)
    else:
        print(f"⚠️ Data item {i+1} missing 'accum_data'")

# === Style
plt.xlabel("Voltage (V)", fontsize ='20')
plt.ylabel("Transport Current (pA)", fontsize ='20')
# plt.title(f"{sample_name} - Combined Accumulation Curves")
plt.grid(True, which='major', color='#CCCCCC', linestyle='--')
plt.grid(True, which='minor', color='#CCCCCC', linestyle=':')
plt.minorticks_on()
plt.legend(fontsize=10)
plt.tight_layout()

# === Save
save_path = os.path.join(
    r"z:\Data\Plotting\Plots\with_mesa",
    f"{sample_name}_combined_accumulation.pdf"
)
plt.savefig(save_path, format='pdf', dpi=500)
plt.show()
print(f"✅ Saved combined accumulation plot to:\n{save_path}")

#%% ######## Combined Pinch offs from a folder ###################

from datetime import datetime
# Set root folder where all pinch-off data are collected
shared_folder = r"z:\Data\Three layer SET\Tuning toolkit\Data\3LSET\With_Mesa\pinch_off"
output_dir = r"z:\Data\Plotting\Plots\with_mesa"
os.makedirs(output_dir, exist_ok=True)

# Load all data items
pinch_off_data = ttf.skeleton.load_all_dataitems_at_path(
    path=shared_folder,
    keywords=["ls-pinch-off-measurement"]
)

device_labels = ["Dev 25", "Dev 26"]
# output_dir = r"Z:\Data\Three layer SET\Tuning toolkit\Plots\Pinchoff_Combined"
os.makedirs(output_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 4))

gate_label = None  # to track x-axis label

for i, data_item in enumerate(pinch_off_data[:2]):
    if "raw_data" not in data_item.data:
        print(f"Data item {i+1} has no 'raw_data'")
        continue

    raw_data = data_item.data["raw_data"] * 1e12  # convert to pA
    plot_obj = raw_data.plot(ax=ax , label=device_labels[i])

    if gate_label is None:
        gate_label = ax.get_xlabel()  # This will be your x-axis label, e.g., LTA or LP

# Finalize plot
ax.set_xlabel(f"{gate_label} (V)")
ax.set_ylabel("Transport Current (pA)")
ax.set_title(f"Pinch-Off {gate_label}")
ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
ax.minorticks_on()
ax.legend(fontsize=12)

# Save
save_path = os.path.join(output_dir, f"withmesa_PinchOff_{gate_label.replace(' ', '_')}_Dev25_Dev26_combo.pdf")
plt.tight_layout()
plt.savefig(save_path, format="pdf", bbox_inches="tight")
print(f"✅ Saved plot to {save_path}")
plt.show()


#%% Saving accumulation data of p files in  a json file

import json
import os

sample_name = "S13_D4"
experiment_type = "RSET"  # or RSET manually
save_filename = f"{sample_name}_pfile_pinch_off_rtb_data.json"

# Step 1: Load file or create clean dictionary
if os.path.exists(save_filename):
    with open(save_filename, "r") as f:
        pfile_PO_data = json.load(f)
    print(f"Loaded existing {save_filename}.")

    # 🛠️ Fix: if pfile_data is a list, convert it
    if isinstance(pfile_data, list):
        print("Old format detected, converting to new format...")
        pfile_PO_data = {"LSET": [], "RSET": []}

else:
    pfile_PO_data = {"LSET": [], "RSET": []}
    print(f"Starting new {save_filename}.")

# Step 2: Process
for i, item in enumerate(items):
    try:
        accum = item.data["accum_data"] * 1e12

        x = accum.coords["virtual_gate"].values
        y = accum.values.squeeze()

        pfile_PO_data[experiment_type].append({
            "x": x.tolist(),
            "y": y.tolist()
        })

        plt.figure(figsize=(5, 4))
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (pA)")
        plt.title(f"{experiment_type} Accumulation {len(pfile_data[experiment_type])}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except KeyError:
        print(f"Item {i+1} does not contain 'accum_data'.")

# Step 3: Save
with open(save_filename, "w") as f:
    json.dump(pfile_data, f, indent=4)

print(f"Saved updated {save_filename}")
print(f"Total LSET: {len(pfile_data['LSET'])} | Total RSET: {len(pfile_data['RSET'])}")




#%% Reading json
import json

# Load the existing pfile_data
filename = "S13_D4_pfile_data.json"

with open(filename, "r") as f:
    pfile_data = json.load(f)

#  Show current number of LSET entries
print(f"Before deletion: {len(pfile_data['LSET'])} LSET entries")

#  Delete the N-th entry (for example, index 2 for third entry)
del pfile_data["LSET"][1]  # Index starts from 0

#  Save back the updated file
with open(filename, "w") as f:
    json.dump(pfile_data, f, indent=4)

print(f"After deletion: {len(pfile_data['LSET'])} LSET entries")


#%%  Pinch off Plots

pinch_off_data = ttf.skeleton.load_all_dataitems_at_path(path = r"z:\Data\Three layer SET\Tuning toolkit\Data\Batch 26\S13_D4", 
                                                         keywords = ["rbb-pinch-off-measurement"])

sample_name='S13_D4'

for i, data_item in enumerate(pinch_off_data):
    if "raw_data" in data_item.data:
        plt.figure(figsize=(5, 4))
        (data_item.data["raw_data"]*1e12).plot()  # Plot the raw data

        ax = plt.gca()

        x_label = ax.get_xlabel()
        plt.ylabel('Current (pA)')
        plt.title(f'Pinch-Off of {x_label}')
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle=':')

        # Save each plot as a separate PDF
        plt.savefig(rf"Z:\Data\Three layer SET\24.01.25_TTF+elicit\Plots\{sample_name}_pinch_off_{x_label}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
    else:
        print(f"Data item {i + 1} does not contain 'raw_data'.")


#%% Saving pinchoff data into json file


# === User Configuration ===
sample_name = "S13_D4"
experiment_type = "RSET"  # or LSET manually
save_folder = r"z:\Data\Plotting\Plots\Plunger"
save_filename = os.path.join(save_folder, f"{sample_name}_rtb_pinch_off_data.json")
os.makedirs(save_folder, exist_ok=True)

# === Load or Initialize JSON Data ===
if os.path.exists(save_filename):
    with open(save_filename, "r") as f:
        pinch_off_data_json = json.load(f)
    print(f"✅ Loaded existing {save_filename}.")
else:
    pinch_off_data_json = {"LSET": [], "RSET": []}
    print(f"✅ Starting new {save_filename}.")

# === Load Pinch-Off Data from TTF ===
pinch_off_data = ttf.skeleton.load_all_dataitems_at_path(
    path = r"z:\Data\Three layer SET\Tuning toolkit\Data\Batch 26\S13_D4", 
    keywords = ["rtb-pinch-off-measurement__112"]
)

# === Process Each Pinch-Off Data Item ===
for i, data_item in enumerate(pinch_off_data):
    if "raw_data" in data_item.data:
        raw_data = data_item.data["raw_data"] * 1e12  # Convert to pA
        gate = data_item.data.get("basis_gate", f"Gate_{i+1}")

        # Extract x and y values
        x_values = [row[2] for row in data_item.data["processed_data"]]
        y_values = [row[0] for row in data_item.data["processed_data"]]

        # Add data to JSON structure
        pinch_off_data_json[experiment_type].append({
            "gate": gate,
            "x": x_values,
            "y": y_values
        })

        # Plotting
        plt.figure(figsize=(5, 4))
        plt.plot(x_values, y_values, linestyle='--', label=f"{gate}") 
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (pA)")
        plt.title(f"{experiment_type} Pinch-Off {gate}")
        plt.grid(True, linestyle=':', color='#CCCCCC')
        plt.tight_layout()

        # Save each plot as PDF
        plot_save_path = os.path.join(save_folder, f"{sample_name}_pinch_off_{experiment_type}_{gate}.pdf")
        plt.savefig(plot_save_path, format='pdf', bbox_inches='tight')
        print(f"✅ Saved plot: {plot_save_path}")
        plt.show()
    else:
        print(f"❌ Data item {i + 1} does not contain 'raw_data'.")

# === Save JSON Data ===
with open(save_filename, "w") as f:
    json.dump(pinch_off_data_json, f, indent=4)

print(f"✅ Saved updated {save_filename}")
print(f"Total LSET: {len(pinch_off_data_json['LSET'])} | Total RSET: {len(pinch_off_data_json['RSET'])}")


# %%  2D sweep plots


corner_data = ttf.skeleton.get_dataitem(r"Z:\Inbox\2025-01-24_10-52-55-589571\data\2025-01-24_11-41-02-659353__lp-vs-ls-corner-evaluator__93ff1f8108753291c8521347eb6837c6.p")
scan = corner_data.ancestry[-2]["data"]
scan.plot()
plt.show()


corner2D_data = ttf.skeleton.load_all_dataitems_at_path(path = r"Z:\Data\Three layer SET\24.01.25_TTF+elicit\TTF_data\2025-01-24_10-52-55-589571\data", keywords = ["corner-evaluator"])

scan1 = corner2D_data.ancestry[-2]["data"]
scan1.plot()
plt.show()

#%% Extracting Corner values
# List all available Corner Evaluator measurements
corner_evaluator_entries = []

for i, data_item in enumerate(corner2D_data):
    try:
        # Extract metadata for each item
        metadata = data_item.metadata if hasattr(data_item, 'metadata') else {}
        name = metadata.get('name', 'Unknown')

        # Check if it's a Corner Evaluator
        if "Corner Evaluator" in name:
            corner_evaluator_entries.append((i, name))

    except AttributeError:
        continue

# Display the list of Corner Evaluators found
for index, name in corner_evaluator_entries:
    print(f"Index: {index}, Name: {name}")

for i, data_item in enumerate(corner2D_data):
    try:
        # Access the scan data from ancestry
        plt.figure(figsize=(6, 4))
        scan2 = (data_item.ancestry[-2]["data"]*1e9)
        
        # Plot the scan
        
        scan2.plot()  # Plot the scan and get the axis object
        ax = plt.gca()

        x_label = ax.get_xlabel() if ax.get_xlabel() else "X-axis"
        y_label = ax.get_ylabel() if ax.get_ylabel() else "Y-axis"


        # Set a dynamic title using the extracted labels
        plt.title("")
        # plt.title(f'{y_label} vs {x_label}')

        plt.savefig(rf"Z:\Data\Three layer SET\24.01.25_TTF+elicit\Plots\{sample_name}_2Dplot_{y_label}vs{x_label}.pdf", format='pdf', bbox_inches='tight')

        plt.show()

    except (IndexError, KeyError) as e:
        print(f"Error processing data item {i + 1}: {e}")

