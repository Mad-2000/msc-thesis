#%% Imports
import os
import sqlite3
import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
from scipy.ndimage import convolve

import json
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import qcodes
import qcodes as qc
from qcodes import initialise_or_create_database_at

import qumada as qm
from qumada.utils import *
from qumada.utils.load_from_sqlite_db import *
from qumada.utils.plotting import *


#%% Matplotlib Parameteres

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (5,4)

matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.minor.size'] = 8
matplotlib.rcParams['xtick.minor.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.minor.size'] = 8
matplotlib.rcParams['ytick.minor.width'] = 2


# %% Analysis Functions using PANDAS 

def print_table_schema(db_path, table_name):
    """
    Prints the schema of a table from a SQLite database.

    Parameters:
    - db_path: str, path to the SQLite .db file.
    - table_name: str, name of the table to inspect.
    """
    try:
        conn = sqlite3.connect(db_path)
        schema_query = f"PRAGMA table_info('{table_name}');"
        schema_df = pd.read_sql_query(schema_query, conn)
        conn.close()

        print(f"Schema of table '{table_name}':")
        print(schema_df)
        return schema_df

    except Exception as e:
        print(f"Error reading schema from {db_path}, table {table_name}: {e}")
        return None
    
def extract_sample_name(db_path):
    """
    Extracts sample name like 'S14_B4' from full database path like 'S14_B4_20241004.db'.
    
    Parameters:
    ----------
    db_path : str
        Full path to the database file.

    Returns:
    ----------
    sample_name : str
        Extracted sample name.
    """
    file_name = os.path.basename(db_path)              # 'S14_B4_20241004.db'
    file_root = os.path.splitext(file_name)[0]          # 'S14_B4_20241004'
    parts = file_root.split('_')                        # ['S14', 'B4', '20241004']

    if len(parts) >= 2:
        sample_name = f"{parts[0]}_{parts[1]}"          # 'S14_B4'
    else:
        sample_name = file_root                         # fallback

    return sample_name

def find_valid_result_tables(db_path):
    """
    Finds all non-empty result tables from a QCoDeS database and includes experiment names.

    Parameters:
    ----------
    db_path : str
        Path to the QCoDeS SQLite database.

    Returns:
    ----------
    valid_tables_df : pd.DataFrame
        DataFrame containing 'experiment_name', 'run_name', 'result_table_name', and 'row_count'
        for tables that have rows.
    """

    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        experiments.name AS experiment_name,
        runs.name AS run_name,
        runs.result_table_name
    FROM runs
    JOIN experiments ON runs.exp_id = experiments.exp_id;
    """

    runs_with_experiments = pd.read_sql_query(query, conn)
    valid_tables = []

    for _, row in runs_with_experiments.iterrows():
        table_name = row['result_table_name']
        try:
            count_query = f"SELECT COUNT(*) AS row_count FROM '{table_name}';"
            count_result = pd.read_sql_query(count_query, conn)

            if count_result['row_count'].iloc[0] >= 0:
                valid_tables.append({
                    'experiment_name': row['experiment_name'],
                    'run_name': row['run_name'],
                    'result_table_name': table_name,
                    'row_count': count_result['row_count'].iloc[0]
                })

        except Exception as e:
            print(f"⚠️ Error checking table '{table_name}': {e}")

    conn.close()
    valid_tables_df = pd.DataFrame(valid_tables)
    return valid_tables_df

def ids_curve(V, I0, a, Vth):   # MOSFET accumulation exponential fit 
    return I0 * np.exp(a * (V - Vth))

def log_ids_curve(V, log_I0, a, Vth): # MOSFET accumulation log fit
    return log_I0 + a * (V - Vth)

def plot_accumulation(
    db_path,  # <-- database connection
    table_names,
    experiment_type='LSET',
    savepath=None,
    plot_leakage=True,
    scale_axis=True,
    hysteresis=False,
    **kwargs
):
    """
    Plots accumulation curves from tables using the plot_multiple_datasets framework.

    Parameters:
    ----------
    conn : sqlite3.Connection
        The connection to the SQLite database.
    table_names : list
        List of table names to plot.
    experiment_type : str
        'LSET' or 'RSET' depending on the experiment.
    savepath : str or None
        If provided, saves plots to this folder.
    plot_leakage : bool
        Whether to also plot the leakage current.
    scale_axis : bool
        Whether to scale the axis (µA, mV etc.).
    hysteresis : bool
        Whether to plot foresweep/backsweep separately.
    kwargs:
        Additional arguments passed to plotting functions.
    """

    datasets_transport = []
    datasets_leakage = []
    legends_transport = []
    legends_leakage = []

    file_name = os.path.basename(db_path)  # Output: 'S14_B4_20241004.db'
    # Optionally, remove the file extension to get a cleaner title
    plot_title = os.path.splitext(file_name)[0]

    for table_name in table_names:
        try:
            conn = sqlite3.connect(db_path)
            data = pd.read_sql_query(f"SELECT * FROM '{table_name}';", conn)

            if experiment_type == 'LSET':
                x = data['keithley_left_volt']
                y_transport = data['lockin_up_R'] * 1e12  # to pA
                y_leakage = data['keithley_left_curr'] * 1e12  # to pA
                x_label = "keithley_left_volt"
                y_label_transport = "lockin_up_R"
                y_label_leakage = "keithley_left_curr"
            elif experiment_type == 'RSET':
                x = data['keithley_right_volt']
                y_transport = data['lockin_down_R'] * 1e12  # to pA
                y_leakage = data['keithley_right_curr'] * 1e12  # to pA
                x_label = "keithley_right_volt"
                y_label_transport = "lockin_down_R"
                y_label_leakage = "keithley_right_curr"
            else:
                raise ValueError("experiment_type must be 'LSET' or 'RSET'")

            # Build "dataset" structure
            dataset_transport = pd.DataFrame({x_label: x, y_label_transport: y_transport})
            datasets_transport.append(dataset_transport)
            legends_transport.append(f"{table_name}_transport")

            if plot_leakage:
                dataset_leakage = pd.DataFrame({x_label: x, y_label_leakage: y_leakage})
                datasets_leakage.append(dataset_leakage)
                legends_leakage.append(f"{table_name}_Leakage")

        except Exception as e:
            print(f"Error loading data from table {table_name}: {e}")

    # Plot transport accumulation
    fig_transport, ax_transport = plt.subplots(figsize=(6, 5))
    for i, data in enumerate(datasets_transport):
        ax_transport.plot(
            data.iloc[:,0],
            data.iloc[:,1],
            marker='o',
            linestyle='-',
            label=legends_transport[i]
        )
    ax_transport.set_xlabel('Voltage (V)')
    ax_transport.set_ylabel('Current (pA)' if experiment_type=='LSET' else 'Current (nA)')
    ax_transport.legend()
    ax_transport.grid(True)
    plt.tight_layout()

    if savepath:
        fig_transport.savefig(f"{savepath}/{plot_title}_accumulation_transport.png", dpi=300)

    plt.show()

    # Plot leakage currents
    if plot_leakage:
        fig_leak, ax_leak = plt.subplots(figsize=(6, 5))
        for i, data in enumerate(datasets_leakage):
            ax_leak.plot(
                data.iloc[:,0],
                data.iloc[:,1],
                marker='o',
                linestyle='-',
                label=legends_leakage[i]
            )
        ax_leak.set_xlabel('Voltage (V)')
        ax_leak.set_ylabel('Leakage Current (pA)' if experiment_type=='LSET' else 'Leakage Current (nA)')
        ax_leak.legend()
        ax_leak.grid(True)
        plt.tight_layout()

        if savepath:
            fig_leak.savefig(f"{savepath}/{plot_title}accumulation_leakage.pdf", dpi=300, format = 'pdf')

        plt.show()

def plot_selected_accumulation(
    conn, 
    experiment_type='LSET', 
    savepath=None, 
    plot_leakage=True,
    keywords=["linear_1d_sweep_gates_measure_ohmics", "accumulation"],
    **kwargs
):
    """
    Plots accumulation/ohmics experiments automatically from database based on keywords.

    Parameters:
    ----------
    conn : sqlite3.Connection
        SQLite database connection.
    experiment_type : str
        'LSET' or 'RSET'.
    savepath : str or None
        Where to save plots, if needed.
    plot_leakage : bool
        Whether to also plot leakage.
    keywords : list
        List of strings to search for in table names.
    kwargs:
        Passed to plotting.
    """

    # --- Step 1: Find matching tables ---
    try:
        all_tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_list = all_tables['name'].tolist()
    except Exception as e:
        print(f"Error loading table names: {e}")
        return

    selected_tables = [t for t in table_list if any(kw in t.lower() for kw in keywords)]

    if not selected_tables:
        print("No matching tables found with given keywords.")
        return

    print(f"Found {len(selected_tables)} matching tables: {selected_tables}")

    # --- Step 2: Plot them ---
    plot_accumulation(
        conn=conn,
        table_names=selected_tables,
        experiment_type=experiment_type,
        savepath=savepath,
        plot_leakage=plot_leakage,
        **kwargs
    )

def plot_multiple_accumulation_with_pfiles(db_table_pairs, 
                                           json_pfile_paths=None, 
                                           save_to_file = None, 
                                           accu_yrange = None, 
                                           accu_xrange = None,
                                           leak_xrange = None,
                                           leak_yrange = None):
    """
    Plots accumulation for:
    - multiple SQLite database tables (transport and leakage)
    - and optionally also p-file accumulation curves from JSON.

    Parameters:
    - db_table_pairs (list of tuples): (db_file, table_name, experiment_type) entries.
    - json_pfile_paths (list of str, optional): List of json files (one per SET) to also plot.
    """

    fig, (ax_accum, ax_leak) = plt.subplots(1, 2, figsize=(10, 6))

    # Set up common plot features
    for ax in (ax_accum, ax_leak):
        ax.set_xlabel('Voltage (V)')
        ax.set_xlim([0, 1])
        ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
        ax.minorticks_on()

    ax_accum.set_ylabel('Current (pA)')
    ax_accum.set_title('Accumulation')
    if accu_xrange is not None:
        ax_accum.set_xlim(accu_xrange)
    if accu_yrange is not None:
        ax_accum.set_ylim(accu_yrange)
    if leak_xrange is not None:
        ax_leak.set_xlim(leak_xrange)
    if leak_yrange is not None:
        ax_leak.set_ylim(leak_yrange) # for accumulation current, for example
    


    ax_leak.set_ylabel('Leakage Current (pA)')
    ax_leak.set_title('Leakage')
     

    # Counter for device numbering
    dev_counter = 1

    # === 1. Plot DB files
    for db_file, table_name, experiment_type in db_table_pairs:
        sample_name = extract_sample_name(db_file)
        try:
            conn = sqlite3.connect(db_file)
            data = pd.read_sql_query(f"SELECT * FROM '{table_name}';", conn)
            conn.close()
            # data = data.dropna(subset=['lockin_up_R'])
            # data = data.dropna(subset=['lockin_down_R'])

            legend_label = f"dev {dev_counter}"
            dev_counter += 1

            if experiment_type == 'LSET':
                # data = data.dropna(subset=['lockin_up_R'
                #                            ])
                x = data['keithley_left_volt']
                y_accum = data['lockin_up_R'] * 1e12
                y_leak = data['keithley_left_curr'] * 1e12

            elif experiment_type == 'RSET':
                # data = data.dropna(subset=['lockin_down_R' 
                #                            ])
                x = data['keithley_right_volt']
                y_accum = data['lockin_down_R'] * 1e12
                y_leak = data['keithley_right_curr'] * 1e12
            else:
                continue

            ax_accum.plot(x, y_accum, marker='o', 
                          markersize = 6, 
                          linestyle='-',
                            label=legend_label)
            ax_leak.plot(x, y_leak, 
                         marker= 'o',
                          markersize = 4, 
                          linestyle='-', 
                          label=legend_label)
            

        except Exception as e:
            print(f"Error processing table {table_name} in {sample_name}: {e}")

    # === 2. Plot P-file data if provided
    if json_pfile_paths is not None:
        for json_path in json_pfile_paths:
            try:
                with open(json_path, "r") as f:
                    pfile_data = json.load(f)

                # Plot LSET
                for curve in pfile_data.get("LSET", []):
                    x = curve["x"]
                    y = curve["y"]
                    legend_label = f"dev {dev_counter}"
                    ax_accum.plot(x, y, marker='o',markersize = 6,
                                   linestyle='-', 
                                   label=legend_label)
                    dev_counter += 1

                # Plot RSET
                for curve in pfile_data.get("RSET", []):
                    x = curve["x"]
                    y = curve["y"]
                    legend_label = f"dev {dev_counter}"
                    ax_accum.plot(x, y, 
                                  marker='o', markersize = 4,
                                  linestyle='-', 
                                  label=legend_label)
                    dev_counter += 1

            except Exception as e:
                print(f"Error reading pfile {json_path}: {e}")

    # === Finish
    ax_accum.legend(fontsize=8)
    ax_leak.legend(fontsize=8)
    plt.tight_layout()
    if save_to_file is not None:
        plt.savefig(save_to_file, format= 'pdf',  dpi=500)
    plt.show()

def fit_and_save_accumulation_fits(db_table_pairs, 
                                   save_json_path, 
                                   save_plot_folder=None,
                                   json_pfile_paths= None):
    """
    Fit accumulation curves with an exponential model and save parameters.
    
    Parameters:
    ----------
    db_table_pairs : list of tuples
        (db_file, table_name, experiment_type) entries.
    save_json_path : str
        Path to save the JSON file with fitted parameters.
    save_plot_folder : str or None
        Folder path to save fitted plots. If None, plots won't be saved.
    """

    # Load existing JSON if exists
    if os.path.exists(save_json_path):
        with open(save_json_path, "r") as f:
            try:
                fitted_data = json.load(f)
                if isinstance(fitted_data, dict):
                    fitted_data = [fitted_data]
            except json.JSONDecodeError:
                fitted_data = []
    else:
        fitted_data = []

    # fitting db
    for db_file, table_name, experiment_type in db_table_pairs:
        try:
            conn = sqlite3.connect(db_file)
            data = pd.read_sql_query(f"SELECT * FROM '{table_name}';", conn)
            conn.close()

            # Extract voltage and current
            if experiment_type == 'LSET':
                data = data.dropna(subset=['keithley_left_volt', 'lockin_up_R'])
                V_data = data['keithley_left_volt'].values
                I_data = data['lockin_up_R'].values
            elif experiment_type == 'RSET':
                data = data.dropna(subset=['keithley_right_volt', 'lockin_down_R'])
                V_data = data['keithley_right_volt'].values
                I_data = data['lockin_down_R'].values
            else:
                print(f"Unknown experiment type: {experiment_type}")
                continue

            # Sort
            sorted_indices = np.argsort(V_data)
            V_data = V_data[sorted_indices]
            I_data = I_data[sorted_indices]

            # Remove low-current region (threshold)
            current_threshold = 5e-12

            mask = (I_data > current_threshold) & (I_data < 300e-12)

            V_data_fit = V_data[mask]
            I_data_fit = I_data[mask]

            plt.figure(figsize=(7,5))
            plt.scatter(V_data, I_data*1e12, label="Data", color="red")
            plt.ylim([-10, 200])
            plt.show()
            plt.close()

            p0_input = input("Enter initial guess for [I0, a, Vth] separated by commas (e.g., 3e-11, 10, 0.5):\n")
            p0_values = [float(x.strip()) for x in p0_input.split(",")]


            # Fit
            sigma = np.sqrt(np.abs(I_data_fit))
            sigma[sigma == 0] = 1e-15

            popt, pcov = curve_fit(
                ids_curve,
                V_data_fit,
                I_data_fit,
                # p0=[30e-12, 10, 0.3],
                p0=p0_values,
                sigma=sigma,
                absolute_sigma=True,
                bounds=([0, 0, 0], [200, np.inf,2]),
                maxfev=10000
            )

            I0_fit, a_fit, Vth_fit = popt
            
            errors = np.sqrt(np.diag(pcov))
            I0_err, a_err, Vth_err = errors

            # Build fit dictionary
            sample_name = os.path.splitext(os.path.basename(db_file))[0]
            fit_result = {
                "sample_name": sample_name,
                "table_name": table_name,
                "SET": experiment_type,
                "I0": I0_fit,
                "I0_err": I0_err,
                "a": a_fit,
                "a_err": a_err,
                "Vth": Vth_fit,
                "Vth_err": Vth_err
            }

            # Check duplicate
            if not any(d['sample_name'] == sample_name and d['table_name'] == table_name for d in fitted_data):
                fitted_data.append(fit_result)
                print(f"✅ Added fit for {sample_name} ({table_name})")
            else:
                print(f"⚠️ Fit already exists for {sample_name} ({table_name})")

            # Plot (optional)
            V_fit = np.linspace(min(V_data), max(V_data), 100)
            I_fit = ids_curve(V_fit, *popt)

            plt.figure(figsize=(7,5))
            plt.scatter(V_data, I_data*1e12, label="Data", color="red")
            plt.plot(V_fit, I_fit*1e12, label="Fit", color="blue")

            # Choose an operating point (near start of rise)
            V_op = V_data_fit[0]  # or choose median point of fit
            I_op = ids_curve(V_op, *popt)
            dIdV = a_fit * I_op

            # Tangent intercept with x-axis
            Vth_tangent = V_op - I_op / dIdV

            # Tangent line
            V_tangent = np.linspace(V_op - 0.1, V_op + 0.1, 10)
            I_tangent = dIdV * (V_tangent - V_op) + I_op

            # plt.plot(V_tangent, I_tangent * 1e12, 'r--', 
            #         label=f"Tangent Intercept: $V_{{th}}^t$ = {Vth_tangent:.3f} V")

            # 4. Vertical line at Vth
            plt.axvline(x=Vth_fit, linestyle='--', 
                        color='cyan', 
                        label=f"$V_{{th}}$={Vth_fit:.3f} V from Fit")
            
            plt.xlabel("Voltage (V)")
            plt.ylabel("Transport Current (pA)")
            # plt.title(f"Accumulation Fit: {sample_name}")
            plt.legend()
            plt.grid(True)
            plt.ylim(0,500)
            plt.tight_layout()

            if save_plot_folder:
                plot_save_path = os.path.join(save_plot_folder, 
                                              f"{sample_name}_{table_name}_fit_title_new2.pdf")
                plt.savefig(plot_save_path, format='pdf', dpi=500)
            plt.show()
            plt.close()

        except Exception as e:
            print(f"Error fitting {db_file} {table_name}: {e}")
    # pfiles
    if json_pfile_paths is not None:
        for json_path in json_pfile_paths:
            try:
                with open(json_path, "r") as f:
                    pfile_data = json.load(f)

                for ptype in ["LSET", "RSET"]:
                    for idx, curve in enumerate(pfile_data.get(ptype, [])):
                        x = np.array(curve["x"])
                        y = np.array(curve["y"]) * 1e-12  # assume stored as pA, convert to A

                        # Sort and mask range
                        sort_idx = np.argsort(x)
                        x = x[sort_idx]
                        y = y[sort_idx]
                        mask = (y > 5e-12) & (y < 200e-12)
                        x_fit = x[mask]
                        y_fit = y[mask]

                        if len(x_fit) < 3:
                            print(f"⚠️ Skipping {ptype} curve #{idx+1} in {json_path}: too few points")
                            continue

                        # Fit
                        sigma = np.sqrt(np.abs(y_fit))
                        sigma[sigma == 0] = 1e-15
                        popt, pcov = curve_fit(
                            ids_curve,
                            x_fit,
                            y_fit,
                            p0=[30e-12, 10, 0.8],
                            sigma=sigma,
                            absolute_sigma=True,
                            bounds=([0, 0, 0], [200, np.inf, 2]),
                            maxfev=10000
                        )
                        I0, a, Vth = popt
                        I0_err, a_err, Vth_err = np.sqrt(np.diag(pcov))

                        # Save
                        fit_result = {
                            "sample_name": os.path.basename(json_path).replace(".json", ""),
                            "table_name": f"{ptype}_{idx+1}",
                            "SET": ptype,
                            "I0": I0,
                            "I0_err": I0_err,
                            "a": a,
                            "a_err": a_err,
                            "Vth": Vth,
                            "Vth_err": Vth_err
                        }
                        fitted_data.append(fit_result)
                        print(f"✅ Fit from {json_path} [{ptype} #{idx+1}]")

                        # Plot
                        V_fit = np.linspace(min(x), max(x), 100)
                        I_fit = ids_curve(V_fit, *popt)

                        plt.figure(figsize=(6, 5))
                        plt.scatter(x, y * 1e12, label="Data", color="red")
                        plt.plot(V_fit, I_fit * 1e12, label="Fit", color="blue")
                        plt.axvline(x=Vth, linestyle='--', color='cyan', label=f"$V_{{th}}$ = {Vth:.3f} V")
                        plt.xlabel("Voltage (V)")
                        plt.ylabel("Current (pA)")
                        plt.title(f"Fit: {json_path} {ptype} #{idx+1}")
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()

                        if save_plot_folder:
                            plot_save_path = os.path.join(
                                save_plot_folder, f"{os.path.basename(json_path).replace('.json','')}_{ptype}_{idx+1}_fit.pdf"
                            )
                            plt.savefig(plot_save_path, dpi=300)
                        plt.show()
                        plt.close()

            except Exception as e:
                print(f"Error processing P-file {json_path}: {e}")

    # Save updated JSON
    with open(save_json_path, "w") as f:
        json.dump(fitted_data, f, indent=4)

    print(f"\n✅ All fits saved to {save_json_path}")

def plot_vth_from_fit_json_by_dev_number(
    json_path,
    save_path=None,
    title=None,
    figsize=(10, 5)
):
    """
    Plot Vth values from a fitted parameters JSON file indexed by device number.
    Colors are separated for LSET and RSET types.

    Parameters:
    ----------
    json_path : str
        Path to the fitted parameters JSON file.
    save_path : str or None
        If provided, saves the plot at this path (PDF recommended).
    title : str
        Title of the plot.
    figsize : tuple
        Figure size for the plot.
    """

    with open(json_path, "r") as f:
        fit_data = json.load(f)

    dev_indices = np.linspace(11, 24, 14)
    vths = []
    colors = []
    markers = []

    for i, entry in enumerate(fit_data, start=1):
        vth = entry["Vth"]
        set_type = entry.get("SET", "LSET").upper()

        # dev_indices.append(i)
        vths.append(vth)
        if set_type == "LSET":
            colors.append("tab:blue")
            markers.append("o")
        else:
            colors.append("tab:orange")
            markers.append("s")

    # Plotting
    plt.figure(figsize=figsize)

    for i, (x, y, c, m) in enumerate(zip(dev_indices, vths, colors, markers)):
        plt.scatter(x, y, color=c, marker=m, s=60)

    # Mean    
    mean_vth = np.mean(vths)
    median_vth = np.median(vths)
    plt.axhline(y=mean_vth, color='green', linestyle='--', linewidth=2,
                label=f"Mean $V_{{th}}$ = {mean_vth:.3f} V")
    plt.axhline(y=median_vth, color='red', linestyle='--', linewidth=2,
                label=f"Median $V_{{th}}$ = {median_vth:.3f} V")
        
    # Create custom handles for LSET and RSET
    lset_handle = plt.Line2D([0], [0], marker='o', color='w', label='LSET',
                            markerfacecolor='tab:blue', markersize=8)
    rset_handle = plt.Line2D([0], [0], marker='s', color='w', label='RSET',
                            markerfacecolor='tab:orange', markersize=8)

    # Mean and Median lines already plotted, get their handles
    mean_handle = plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2,
                            label=f"Mean $V_{{th}}$ = {mean_vth:.3f} V")
    median_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                            label=f"Median $V_{{th}}$ = {median_vth:.3f} V")

    plt.xticks(dev_indices)
    plt.xlabel("SET")
    plt.ylabel("Turn-on Voltage (V)")
    plt.ylim([0, 1.8])
    plt.title(title)
    plt.grid(True, linestyle=':', color='gray')


    # Combine all and display
    plt.legend(handles=[lset_handle, rset_handle, mean_handle, median_handle])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()

def plot_vth_from_fit_json_with_SG(
    json_path,
    save_path=None,
    title=None,
    figsize=(10, 5)
):
    import matplotlib.patches as mpatches

    with open(json_path, "r") as f:
        fit_data = json.load(f)

    dev_indices = np.linspace(11,24,14)
    vths = []
    colors = []
    markers = []
    highlight_flags = []

    for i, entry in enumerate(fit_data, start=1):
        vth = entry["Vth"]
        set_type = entry.get("SET", "LSET").upper()
        needs_screening = entry.get("needs_screening", False)

        # dev_indices.append(i)
        vths.append(vth)
        highlight_flags.append(needs_screening)

        if set_type == "LSET":
            colors.append("tab:blue")
            markers.append("o")
        else:
            colors.append("tab:orange")
            markers.append("s")

    plt.figure(figsize=figsize)

    for i, (x, y, c, m, h) in enumerate(zip(dev_indices, vths, colors, markers, highlight_flags)):
        if h:
            plt.scatter(x, y, color=c, 
                        marker=m, s=120, 
                        edgecolors='lime', 
                        linewidths=4, 
                        label="Screened" if i==0 else "")
        else:
            plt.scatter(x, y, color=c, marker=m, s=60)

    # Mean and Median
    mean_vth = np.mean(vths)
    median_vth = np.median(vths)
    plt.axhline(y=mean_vth, color='green', linestyle='--', linewidth=2,
                label=f"Mean $V_{{th}}$ = {mean_vth:.3f} V")
    plt.axhline(y=median_vth, color='red', linestyle='--', linewidth=2,
                label=f"Median $V_{{th}}$ = {median_vth:.3f} V")

    plt.xticks(dev_indices)
    plt.xlabel("SET")
    plt.ylabel("Activation Voltage (V)")
    plt.ylim([0, 1.8])
    if title:
        plt.title(title)
    plt.grid(True, linestyle=':', color='gray')

    # Legend for LSET, RSET, Screening
    lset_handle = plt.Line2D([0], [0], marker='o', color='w', label='LSET',
                            markerfacecolor='tab:blue', markersize=8)
    rset_handle = plt.Line2D([0], [0], marker='s', color='w', label='RSET',
                            markerfacecolor='tab:orange', markersize=8)
    screen_handle = mpatches.Patch(facecolor='white', edgecolor='lime', 
                                   label='with +ve SG', linewidth=4)
    # Mean and Median lines already plotted, get their handles
    mean_handle = plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2,
                            label=f"Mean $V_{{th}}$ = {mean_vth:.3f} V")
    median_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                            label=f"Median $V_{{th}}$ = {median_vth:.3f} V")

    plt.legend(handles=[lset_handle, rset_handle, screen_handle, 
                        mean_handle, median_handle], loc='best')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Plot saved to: {save_path}")

    plt.show()


#%% ############################  Qcodes Functions for db file analysis ###########################################
# QCODES Plotter from Sala and Offermann

def find_max_run_id(current_connection: qcodes.dataset.ConnectionPlus):
    "Find maximum run id available in db."
    max_run_id = 1
    for i in range(1000):
        try:
            _ = qcodes.dataset.data_set.load_by_id(max_run_id, current_connection)
            max_run_id += 1
        except:
            return max_run_id
    return max_run_id

def load_and_extract_dataset_entries(run_id: int, connection: qcodes.dataset.ConnectionPlus) -> dict:
    """Load dataset and extract relevant information that are relevant to the client.
    
    Unfortunately we need to extract or serialize the dataset before loading it into the store.
    If additional information of a run should be available to the plotting functions, add it here.
    """
    dataset = qcodes.dataset.data_set.load_by_id(run_id, connection)
    
    extract = {}
    extract["db_path"] = connection.path_to_dbfile
    extract["data"] = dataset.get_parameter_data()
    extract["run_id"] = dataset.run_id
    extract["exp_name"] = dataset.exp_name
    extract["name"] = dataset.name
    extract["metadata"] = dataset.metadata
    extract["guid"] = dataset.guid
    extract["number_of_results"] = dataset.number_of_results
    extract["run_timestamp_raw"] = dataset.run_timestamp_raw
    extract["completed_timestamp_raw"] = dataset.completed_timestamp_raw
    extract["max_run_id"] = find_max_run_id(connection)
    extract["parameters"] = {(x:=param_spec._to_dict())["name"]: x for param_spec in dataset.get_parameters()}
    
    return extract

def readout_database(data_selection, db_path_input_value, run_id_value):
    # run_id_value = run_id_value + 1 # if not getting exp_id starting from 0
    run_id_value = run_id_value +2
    current_connection = qcodes.dataset.connect(db_path_input_value)
    
    data_selection = load_and_extract_dataset_entries(run_id_value, current_connection)
    current_connection.close()

    return data_selection

def auto_detect_gate_from_run_name(meas, set_side):
    """
    Detect gate name from measurement metadata and data structure, 
    assuming 'plunger' is mentioned in exp_name/run_name.

    Parameters:
    - meas: QCoDeS measurement dictionary from readout_database
    - set_side: 'left' or 'right'

    Returns:
    - Detected gate name as string, or None
    """
    exp_name = meas.get("exp_name", "").lower()
    data_key = "keithley_right_curr" if set_side.lower() == "right" else "keithley_left_curr"
    all_keys = meas["data"][data_key].keys()

    # Search for 'plunger' and a gate name pattern in exp_name
    if "plunger" in exp_name:
        # Look for any DAC voltage key
        for key in all_keys:
            if re.match(r'dac\d+_Slot\d+_Chan\d+_volt', key):
                return key

    return None  # fallback if not detected

def generate_device_color_map(db_table_pairs):
    """
    Generates a consistent color map for each device in the list.

    Parameters:
    - db_table_pairs : list of (db_path, exp_id, set_side)
    
    Returns:
    - device_color_map : dict with (db_path, exp_id) as keys and color as value.
    """
    num_devices = len(db_table_pairs)
    colors = plt.cm.viridis(np.linspace(0, 1, num_devices))  # Generate unique colors
    
    device_color_map = {}
    for idx, (db_path, exp_id, set_side) in enumerate(db_table_pairs):
        device_key = (db_path)  # Unique identifier for each device
        device_color_map[device_key] = colors[idx]
    
    return device_color_map

# --- Accumulation Plot --

def plot_accumulation_v3(db_path, 
                         exp_id, 
                         set_side='rset', 
                         xlim=None, 
                         ylim_transport=None, 
                         ylim_leakage=None,
                         save_path=None):

    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    # === Select transport block (lockin or mfli)
    if set_side.lower() == 'rset':
        trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
        leak_block = meas["data"].get("keithley_right_curr", None)
    elif set_side.lower() == 'lset':
        trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
        leak_block = meas["data"].get("keithley_left_curr", 'keithley_left_curr')

    else:
        raise ValueError("set_side must be 'lset' or 'rset'.")

    if trans_block is None:
        print(f"❌ No transport data available for {set_side.upper()}")
        return

    # === Extract x and y for transport
    x_trans_key = next(k for k in trans_block if "volt" in k)
    y_trans_key = next(k for k in trans_block if "lockin" in k or "current" in k)
    x_trans = trans_block[x_trans_key]
    y_trans = trans_block[y_trans_key]

    # === Extract x and y for leakage if available
    x_leak = y_leak = None
    if leak_block is not None:
        x_leak_key = next((k for k in leak_block if "volt" in k), None)
        if x_leak_key is not None and "keithley" in leak_block:
            x_leak = leak_block[x_leak_key]
            y_leak = leak_block["keithley_right_curr" if set_side == "right" else "keithley_left_curr"]

    # Determine if leakage data is available
    has_leakage = x_leak is not None and y_leak is not None
    # fig, (ax_trans, ax_leak) = plt.subplots(1, 2, figsize=(10, 6))

    # === Plotting
    if has_leakage:
        fig, (ax_trans, ax_leak) = plt.subplots(1, 2, figsize=(7, 4))
    else:
        fig, ax_trans = plt.subplots(figsize=(5, 4))

    # --- Transport plot (left)
    ax_trans.plot(x_trans, y_trans * 1e12, label="Transport Current", color='blue')
    ax_trans.set_xlabel("Voltage (V)")
    ax_trans.set_ylabel("Current (pA)")
    ax_trans.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_trans.minorticks_on()
    ax_trans.legend()
    if xlim:
        ax_trans.set_xlim(xlim)
    if ylim_transport:
        ax_trans.set_ylim(ylim_transport)

    # --- Leakage plot (right), only if available
    if has_leakage:
        ax_leak.plot(x_leak, y_leak * 1e12, label="Leakage Current", color='red')
        ax_leak.set_xlabel("Voltage (V)")
        ax_leak.set_ylabel("Current (pA)")
        ax_leak.grid(True, linestyle=':', color='#CCCCCC')
        ax_leak.minorticks_on()
        ax_leak.legend()
        if xlim:
            ax_leak.set_xlim(xlim)
        if ylim_leakage:
            ax_leak.set_ylim(ylim_leakage)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")
    plt.show()

def plot_multiple_accumulation_qcodes(db_table_pairs, 
                                      json_pfile_paths=None, 
                                      save_to_file=None, 
                                      accu_xrange=None, 
                                      accu_yrange=None,
                                      leak_xrange=None,
                                      leak_yrange=None):
    """
    Plot accumulation (transport) and leakage for multiple QCoDeS databases.

    Parameters:
    - db_table_pairs : list of (db_path, exp_id, set_side)
    - json_pfile_paths : list of JSON files with pfile curves
    - save_to_file : optional PDF output path
    """
    fig, (ax_accum, ax_leak) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

    # ax_accum.set_title("Transport Current")
    ax_accum.set_xlabel("Voltage (V)")
    ax_accum.set_ylabel("Transport Current (pA)")
    ax_accum.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_accum.minorticks_on()

    # ax_leak.set_title("Leakage Current")
    ax_leak.set_xlabel("Voltage (V)")
    ax_leak.set_ylabel("Leakage Current (nA)")
    ax_leak.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_leak.minorticks_on()

    if accu_xrange:
        ax_accum.set_xlim(accu_xrange)
    if accu_yrange:
        ax_accum.set_ylim(accu_yrange)
    if leak_xrange:
        ax_leak.set_xlim(leak_xrange)
    if leak_yrange:
        ax_leak.set_ylim(leak_yrange)

    dev_counter = 11

    # === Plot DB files using QCoDeS ===
    for db_path, exp_id, set_side in db_table_pairs:
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            sample_name = extract_sample_name(db_path)
            label = f"Dev {dev_counter}"

            if set_side.lower() == 'rset':
                trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
                leak_block = meas["data"].get("keithley_right_curr", None)
            elif set_side.lower() == 'lset':
                trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
                leak_block = meas["data"].get("keithley_left_curr", None)

            if set_side.lower() == 'rset':
                x1 = meas["data"]["keithley_right_curr"]["keithley_right_volt"]
                y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
                x2 = meas["data"]["lockin_down_R"]["keithley_right_volt"]
                y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
            elif set_side.lower() == 'lset':
                x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
                y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
                x2 = meas["data"]["lockin_up_R"]["keithley_left_volt"]
                y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
            else:
                raise ValueError("set_side must be 'LSET' or 'RSET'")
            
            # color = device_color_map.get(db_path, "black")

            ax_accum.plot(x2, y2 * 1e12, 
                          marker='o', 
                          linestyle='-', markersize=5, label=label)
            ax_leak.plot(x1, y1 * 1e9, 
                         marker='s', 
                         linestyle='--', markersize=4, label=label)

            dev_counter += 1
        except Exception as e:
            print(f"❌ Error reading {db_path}, exp_id {exp_id}: {e}")

    # === Plot P-file data if given ===
    if json_pfile_paths:
        for json_path in json_pfile_paths:
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                dev_counter = 23
                colors_rset = ['magenta', 'black']

                for curve in data.get("LSET", []) + data.get("RSET", []):
                    x = np.array(curve["x"])
                    y = np.array(curve["y"])
                    label = f"Dev {dev_counter}"
                    color = colors_rset[(dev_counter - 1) % len(colors_rset)]
                    ax_accum.plot(x, y, 
                                  linestyle='-', 
                                  marker='o', 
                                  markersize=5, 
                                  label=label, color = color)
                    dev_counter += 1
            except Exception as e:
                print(f"❌ Error reading JSON file {json_path}: {e}")

    ax_accum.legend(fontsize=7, loc='best')
    ax_leak.legend(fontsize=7, loc='best')
    fig.tight_layout()

    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, format='pdf', dpi=300)
        print(f"✅ Plot saved to: {save_to_file}")

    plt.show()

def plot_multiple_accumulation_qcodes_with_mfli(db_table_pairs, 
                                      save_to_file=None, 
                                      accu_xrange=None, 
                                      accu_yrange=None,
                                      leak_xrange=None,
                                      leak_yrange=None):
    """
    Plot accumulation (transport) and leakage for multiple QCoDeS databases.

    Parameters:
    - db_table_pairs : list of (db_path, exp_id, set_side)
    - json_pfile_paths : list of JSON files with pfile curves
    - save_to_file : optional PDF output path
    """
    fig, (ax_accum, ax_leak) = plt.subplots(1, 2, figsize=(10, 6), sharex=True)

    ax_accum.set_title("Transport Current")
    ax_accum.set_xlabel("Voltage (V)")
    ax_accum.set_ylabel("Current (pA)")
    ax_accum.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_accum.minorticks_on()

    ax_leak.set_title("Leakage Current")
    ax_leak.set_xlabel("Voltage (V)")
    ax_leak.set_ylabel("Current (nA)")
    ax_leak.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_leak.minorticks_on()

    if accu_xrange:
        ax_accum.set_xlim(accu_xrange)
    if accu_yrange:
        ax_accum.set_ylim(accu_yrange)
    if leak_xrange:
        ax_leak.set_xlim(leak_xrange)
    if leak_yrange:
        ax_leak.set_ylim(leak_yrange)

    dev_counter = 1

    # === Plot DB files using QCoDeS ===
    for db_path, exp_id, set_side in db_table_pairs:
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            sample_name = extract_sample_name(db_path)
            label = f"Dev {dev_counter}"

            # if set_side.lower() == 'rset':
            #     trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
            #     leak_block = meas["data"].get("keithley_right_curr", None)
            # elif set_side.lower() == 'lset':
            #     trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
            #     leak_block = meas["data"].get("keithley_left_curr", None)
            # y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
            # y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
            # x1 = meas["data"]["keithley_right_curr"].get("time", meas["data"]["keithley_right_curr"].get("clock"))
            # x2 = meas["data"]["lockin_down_R"].get("time", meas["data"]["lockin_down_R"].get("clock"))

            if set_side.lower() == 'rset':
                x1 = meas["data"]["keithley_right_curr"]["keithley_right_volt"]
                y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
                x2 = meas["data"]["mfli2_current"]["keithley_right_volt"]
                y2 = meas["data"]["mfli2_current"]["mfli2_current"]
            elif set_side.lower() == 'lset':
                x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
                y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
                x2 = meas["data"]["mfli1_current"]["keithley_left_volt"]
                y2 = meas["data"]["mfli1_current"]["mfli1_current"]
            else:
                raise ValueError("set_side must be 'LSET' or 'RSET'")

            ax_accum.plot(x2, y2 * 1e12, 
                          marker='o', 
                          linestyle='-', markersize=5, label=label)
            ax_leak.plot(x1, y1 * 1e12, 
                         marker='s', 
                         linestyle='--', markersize=4, label=label)

            dev_counter += 1
        except Exception as e:
            print(f"❌ Error reading {db_path}, exp_id {exp_id}: {e}")

    ax_accum.legend(fontsize=7, loc='best')
    ax_leak.legend(fontsize=7, loc='best')
    fig.tight_layout()

    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, format='pdf', dpi=300)
        print(f"✅ Plot saved to: {save_to_file}")

    plt.show()

# --- Time Stability Plot ---
def plot_time_stability_v2(db_path, 
                           exp_id, 
                           set_side='rset', 
                           save_path=None,
                           ylim=None):
    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    # === Determine transport and leakage sources
    if set_side.lower() == 'rset':
        trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
        leak_block = meas["data"].get("keithley_right_curr", None)
    elif set_side.lower() == 'lset':
        trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
        leak_block = meas["data"].get("keithley_left_curr", None)
    else:
        raise ValueError("set_side must be 'rset' or 'lset'.")

    # === Time and signal for transport
    x_trans = trans_block.get("time", trans_block.get("clock", None))
    y_trans_key = next((k for k in trans_block if "lockin" in k or "current" in k), None)
    y_trans = trans_block[y_trans_key] if y_trans_key else None

    # === Time and signal for leakage (if exists)
    x_leak = y_leak = None
    if leak_block is not None:
        x_leak = leak_block.get("time", leak_block.get("clock", None))
        y_leak_key = next((k for k in leak_block if "curr" in k), None)
        y_leak = leak_block[y_leak_key] if y_leak_key else None

    # === Plotting
    fig, ax = plt.subplots(figsize=(5, 4))

    if x_leak is not None and y_leak is not None:
        ax.plot(x_leak - x_leak[0], 
                y_leak * 1e12, 
                label="Leakage Current", color='red')

    if x_trans is not None and y_trans is not None:
        ax.plot(x_trans - x_trans[0], 
                y_trans * 1e12,
                  label="Transport Current", color='blue')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (pA)")
    ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax.minorticks_on()
    ax.set_ylim(ylim)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")

    plt.show()

def plot_multi_time_stability_qcodes(db_table_list, 
                                          device_labels=None, 
                                          save_path=None,
                                          figsize=(6, 1.8)):
    """
    Stacked time-stability plots with labeled devices and visible y-ticks.

    Parameters:
    - db_table_list: list of (db_path, exp_id, set_side)
    - device_labels: list of strings (same length) e.g. ["Dev 1", "Dev 2"]
    - save_path: optional file path to save the plot
    """
    num_devices = len(db_table_list)
    fig, axes = plt.subplots(num_devices, 1, figsize=(figsize[0], figsize[1]*num_devices), sharex=True)

    if num_devices == 1:
        axes = [axes]

    for idx, ((db_path, exp_id, set_side), ax) in enumerate(zip(db_table_list, axes)):
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            dev_label = device_labels[idx] if device_labels else f"Dev {idx+1}"

            if set_side.lower() == 'rset':
                y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
                y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
                x1 = meas["data"]["keithley_right_curr"].get("time", meas["data"]["keithley_right_curr"].get("clock"))
                x2 = meas["data"]["lockin_down_R"].get("time", meas["data"]["lockin_down_R"].get("clock"))
            elif set_side.lower() == 'lset':
                y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
                y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
                x1 = meas["data"]["keithley_left_curr"].get("time", meas["data"]["keithley_left_curr"].get("clock"))
                x2 = meas["data"]["lockin_up_R"].get("time", meas["data"]["lockin_up_R"].get("clock"))
            else:
                raise ValueError("set_side must be 'LSET' or 'RSET'")

            # Align to t=0
            x0 = x1[0]
            ax.plot(x1 - x0, y1 * 1e9, color='black', linestyle='--', linewidth=0.7, label='Leakage')
            ax.plot(x2 - x0, y2 * 1e9, color='red', linewidth=1, label='Transport')

            ax.set_ylabel("nA")
            ax.set_title(dev_label, loc='left', fontsize=9)
            ax.grid(True, linestyle=':', linewidth=0.5, color='#AAAAAA')
            ax.tick_params(labelsize=8)
            # ax.xlim([0,30])

        except Exception as e:
            ax.set_title(f"❌ {dev_label} (Err)", color='red', fontsize=8)
            ax.set_yticks([])

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.supylabel("Current (nA)", fontsize=12)
    plt.tight_layout(h_pad=0.7)
    plt.legend()
    plt.xlim([0,30])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Plot saved to {save_path}")

    plt.show()

# --- 1D Gate Pinch-off Sweep ---

def plot_gate_with_sweep_direction(db_path, 
                                      exp_id,  
                                      set_side='right', 
                                      save_path=None, 
                                      ylim=None,
                                      xlabel=None):
    """
    Plot a plunger sweep with forward/backward color coding in two subplots:
    - Right: Leakage current
    - Left: Transport current
    """
    # Load measurement
    meas = readout_database("data", db_path, exp_id - 1)

    for key in meas["data"]["lockin_down_R"]:
        if "volt" in key:
            gate_name = key
            break

    # title = meas["exp_name"]
    sample_name = extract_sample_name(db_path)
    clean_name = f"{sample_name}"
    print("Available keys in keithley_right_curr:", meas["data"]["keithley_right_curr"].keys())
    print("Available keys in lockin_down_R:", meas["data"]["lockin_down_R"].keys())

    # Extract data
    try:
        if set_side.lower() == 'right':
            y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
            y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
            x1 = meas["data"]["keithley_right_curr"][gate_name]
            x2 = meas["data"]["lockin_down_R"][gate_name]
        elif set_side.lower() == 'left':
            y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
            y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
            x1 = meas["data"]["keithley_left_curr"][gate_name]
            x2 = meas["data"]["lockin_up_R"][gate_name]
        else:
            raise ValueError("set_side must be 'left' or 'right'.")
    except KeyError as e:
        print(f"Key error: {e}")
        return

    # Separate forward/backward sweeps
    x1_separated, y1_separated, dir1 = separate_up_down(x1, y1)
    x2_separated, y2_separated, dir2 = separate_up_down(x2, y2)

    # Create 2 subplots: leakage and transport
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    colors = {1: 'blue', -1: 'red'}  # forward: blue, backward: red

    # Transport current =
    for xs, ys, d in zip(x2_separated, y2_separated, dir2):
        d = int(d) if d != 0 else 1  # fallback to forward if direction is 0
        label = f'Forward' if d == 1 else 'Backward'
        ax1.plot(xs, ys * 1e9, color=colors[d], linestyle='-', label=label)

    ax1.set_ylabel("Transport Current (nA)")
    # ax1.set_title("Transport Current")
    xlabel_text = f"{xlabel} Voltage (V)" if xlabel else "Gate Voltage (V)"
    ax1.set_xlabel(xlabel_text)
    ax1.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax1.legend(fontsize=8)

    # Leakage current =
    for xs, ys, d in zip(x1_separated, y1_separated, dir1):
        d = int(d) if d != 0 else 1
        label = f'Forward' if d == 1 else 'Backward'
        ax2.plot(xs, ys * 1e9, color='black', linestyle='--', label=label)

    ax2.set_ylabel("Leakage Current (nA)")
    ax2.set_xlabel(xlabel_text)
    # ax2.set_title("Leakage Current")
    ax2.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax2.legend(fontsize=8)



    if ylim:
        # ax1.set_ylim([-0.1, 0.6])

        ax2.set_ylim([-0.04, 12])

    plt.tight_layout()
    

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # gate_save_path = os.path.join(save_folder, 
        #                               f"{clean_name}_{exp_id}_{xlabel}_sweeps.pdf")
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {gate_save_path}")

    plt.show()

def plot_gate_with_sweep_direction_v2(
    db_path, 
    exp_id,  
    set_side='rset', 
    save_path=None,
    ylim_trans =None ,
    ylim_leakage=None,
    xlabel=None
):
    """
    Plot a plunger sweep with forward/backward color coding:
    - Left: Transport current (from lockin or mfli)
    - Right: Leakage current (if available)
    """

    # Load data
    meas = readout_database("data", db_path, exp_id - 1)
    sample_name = extract_sample_name(db_path)

    # --- Identify transport and leakage blocks
    if set_side.lower() == 'rset':
        trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
        leak_block = meas["data"].get("keithley_right_curr", None)
    elif set_side.lower() == 'lset':
        trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
        leak_block = meas["data"].get("keithley_left_curr", None)
    else:
        raise ValueError("set_side must be 'rset' or 'lset'.")

    if trans_block is None:
        print(f"❌ No transport data found for {set_side}")
        return

    # --- Auto-detect gate voltage key
    gate_name = next((k for k in trans_block if "volt" in k.lower()), None)
    if gate_name is None:
        print("❌ No gate voltage key found in transport block")
        return

    y_trans_key = next((k for k in trans_block if "lockin" in k or "current" in k), None)
    if y_trans_key is None:
        print("❌ No transport current key found")
        return

    # --- Extract data
    x_trans = trans_block[gate_name]
    y_trans = trans_block[y_trans_key]

    if leak_block:
        x_leak_key = next((k for k in leak_block if "volt" in k.lower()), None)
        y_leak_key = next((k for k in leak_block if "curr" in k), None)
        x_leak = leak_block.get(x_leak_key, None)
        y_leak = leak_block.get(y_leak_key, None)
    else:
        x_leak = y_leak = None

    # --- Separate forward/backward
    x_trans_sep, y_trans_sep, dir_trans = separate_up_down(x_trans, y_trans)
    if x_leak is not None and y_leak is not None:
        x_leak_sep, y_leak_sep, dir_leak = separate_up_down(x_leak, y_leak)
    else:
        x_leak_sep = y_leak_sep = dir_leak = []

    # --- Setup subplots
    if x_leak is not None and y_leak is not None:
        fig, (ax_trans, ax_leak) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    else:
        fig, ax_trans = plt.subplots(1, 1, figsize=(5,4))
        ax_leak = None

    colors = {1: 'blue', -1: 'red'}

    # --- Plot transport
    for xs, ys, d in zip(x_trans_sep, y_trans_sep, dir_trans):
        d = int(d) if d != 0 else 1
        label = f"Forward" if d == 1 else "Backward"
        ax_trans.plot(xs, ys * 1e12, color=colors[d], linestyle='-' if d == 1 else '-', label=label)

    ax_trans.set_ylabel("Transport Current (pA)")
    xlabel_text = f"{xlabel} (V)" if xlabel else "Gate Voltage (V)"
    ax_trans.set_xlabel(xlabel_text)
    ax_trans.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_trans.minorticks_on()
    ax_trans.legend(fontsize=8)
    ax_trans.set_ylim(ylim_trans)

    # --- Plot leakage if available
    if ax_leak:
        for xs, ys, d in zip(x_leak_sep, y_leak_sep, dir_leak):
            d = int(d) if d != 0 else 1
            label = f"Forward" if d == 1 else "Backward"
            ax_leak.plot(xs, ys * 1e12, color=colors[d], linestyle='--' if d == 1 else '--', label=label)

        ax_leak.set_ylabel("Leakage Current (pA)")
        ax_leak.set_xlabel(xlabel_text)
        ax_leak.grid(True, which='both', linestyle=':', color='#CCCCCC')
        ax_leak.minorticks_on()
        ax_leak.legend(fontsize=8)
        if ylim_leakage:
            ax_leak.set_ylim(ylim_leakage)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")

    plt.show()

def plot_multiple_plunger_sweeps_v2(
        db_file_list, 
        json_file=None, 
        save_to_file=None, 
        ylim=None
):
    """
    Plot multiple plunger sweeps from QCoDeS databases and JSON data (with user-specified DAC keys).
    
    Parameters:
    - db_file_list : list of (db_path, exp_id, dac_key, set_side, xlabel)
    - json_file : str : Path to the JSON file containing plunger sweeps.
    - save_to_file : str : Path to save the combined plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # === Plot from QCoDeS Databases ===
    for db_path, exp_id, dac_key, set_side, xlabel in db_file_list:
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            sample_name = extract_sample_name(db_path)

            # Use the specified DAC key for plunger voltage
            if set_side.lower() == 'rset':
                trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
                leak_block = meas["data"].get("keithley_right_curr", None)
            elif set_side.lower() == 'lset':
                trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
                leak_block = meas["data"].get("keithley_left_curr", None)
            else:
                continue

            # Ensure DAC key exists
            if dac_key not in trans_block:
                print(f"❌ DAC key '{dac_key}' not found in {sample_name} {set_side.upper()}")
                continue

            # Extract transport data
            x_trans = trans_block[dac_key]
            y_trans_key = next((k for k in trans_block if "lockin" in k or "current" in k), None)
            if y_trans_key is None:
                print(f"❌ No transport current key found for {sample_name} {set_side.upper()}")
                continue

            y_trans = trans_block[y_trans_key] * 1e12  # Convert to pA

            # Separate forward/backward
            x_trans_sep, y_trans_sep, dir_trans = separate_up_down(x_trans, y_trans)
            if len(x_trans_sep) < 2:
                continue

            colors = {1: 'blue', -1: 'red'}
            for xs, ys, d in zip(x_trans_sep, y_trans_sep, dir_trans):
                d = int(d) if d != 0 else 1
                ax.plot(xs, ys, color=colors[d], linestyle='-' if d == 1 else '--', 
                        label=f"{sample_name} {set_side.upper()} ({xlabel})")

        except Exception as e:
            print(f"❌ Error reading {db_path}, exp_id {exp_id}: {e}")

    # === Plot from JSON file ===
    if json_file:
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)

            for key, data in json_data.items():
                plunger_voltage = data.get("plunger_voltage")
                transport_current = data.get("transport_current")
                sample_name = data.get("sample_name")
                set_side = data.get("set_side")
                fixed_voltage_x = data.get("fixed_voltage_x")

                ax.plot(
                    plunger_voltage, transport_current, 
                    linestyle='-',  
                    label=f"{sample_name} {set_side.upper()} (JSON)"
                )
        except Exception as e:
            print(f"❌ Error reading JSON file: {e}")

    # === Finalize Plot ===
    ax.set_xlabel("Plunger Voltage (V)")
    ax.set_ylabel("Transport Current (pA)")
    ax.set_title("Combined Plunger Sweeps (User-Specified DAC Keys)")
    ax.grid(True, linestyle=':', color='#CCCCCC')
    ax.minorticks_on()
    ax.legend(fontsize=9)
    ax.set_xlim([0,2])
    ax.set_ylim([-10,200])

    plt.tight_layout()
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, format='pdf', dpi=300)
        print(f"✅ Saved combined plot to {save_to_file}")

    plt.show()

def plot_multiple_plunger_sweeps_with_leakage_v4(
        db_file_list, 
        json_file=None, 
        save_to_file=None, 
        device_labels=None,
        ylim_transport=None,
        ylim_leakage=None
):
    """
    Plot multiple plunger sweeps and their corresponding leakage in separate plots.

    Parameters:
    - db_file_list : list of (db_path, exp_id, dac_key, set_side, xlabel)
    - json_file : str : Path to the JSON file containing plunger sweeps.
    - save_to_file : str : Path to save the combined plot.
    - device_labels : list of strings for legend labels.
    """
    fig, (ax_trans, ax_leak) = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(db_file_list)))  # Use a color map for unique colors

    # === Plot from QCoDeS Databases ===
    for idx, (db_path, exp_id, dac_key, set_side, xlabel) in enumerate(db_file_list):
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            sample_name = extract_sample_name(db_path)
            color = colors[idx]  # Use consistent color for transport and leakage

            # Use the specified DAC key for plunger voltage
            if set_side.lower() == 'rset':
                trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
                leak_block = meas["data"].get("keithley_right_curr", {}).get("keithley_right_curr", None)
            elif set_side.lower() == 'lset':
                trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
                leak_block = meas["data"].get("keithley_left_curr", {}).get("keithley_left_curr", None)
            else:
                continue

            # Ensure DAC key exists
            if dac_key not in trans_block:
                print(f"❌ DAC key '{dac_key}' not found in {sample_name} {set_side.upper()}")
                continue

            # Extract transport data
            x_trans = trans_block[dac_key]
            y_trans_key = next((k for k in trans_block if "lockin" in k or "current" in k), None)
            if y_trans_key is None:
                print(f"❌ No transport current key found for {sample_name} {set_side.upper()}")
                continue

            y_trans = trans_block[y_trans_key] * 1e12  # Convert to pA
            label = device_labels[idx] if device_labels else f"Dev {idx+1}"

            # Detect forward/backward for transport
            x_trans_sep, y_trans_sep, dir_trans = separate_up_down(x_trans, y_trans)
            if len(dir_trans) > 1:
                for xs, ys, direction in zip(x_trans_sep, y_trans_sep, dir_trans):
                    linestyle = '-' if direction == 1 else '--'
                    ax_trans.plot(xs, ys, label=f"{label} (Forward)" if direction == 1 else f"{label} (Backward)", 
                                  linestyle=linestyle, color=color)
            else:
                ax_trans.plot(x_trans, y_trans, label=label, linestyle='-', color=color)

            # Plot leakage if available
            if leak_block is not None:
                y_leak = leak_block * 1e9  # Convert to pA
                x_leak = x_trans  # Use the same x-axis for leakage

                # Detect forward/backward for leakage
                x_leak_sep, y_leak_sep, dir_leak = separate_up_down(x_leak, y_leak)
                if len(dir_leak) > 1:
                    for xs, ys, direction in zip(x_leak_sep, y_leak_sep, dir_leak):
                        linestyle = '-' if direction == 1 else '--'
                        ax_leak.plot(xs, ys, label=f"{label} (Leak Forward)" if direction == 1 else f"{label} (Leak Backward)", 
                                     linestyle=linestyle, color=color)
                else:
                    ax_leak.plot(x_leak, y_leak, label=f"{label} (Leak)", linestyle='-', color=color)

        except Exception as e:
            print(f"❌ Error reading {db_path}, exp_id {exp_id}: {e}")

    # === Plot from JSON file ===
    if json_file:
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)

            for idx, (key, data) in enumerate(json_data.items()):
                plunger_voltage = data.get("plunger_voltage")
                transport_current = data.get("transport_current")
                label = device_labels[idx + len(db_file_list)] if device_labels else f"Dev {idx+1} (JSON)"
                color = colors[idx % len(colors)]  # Use consistent color

                ax_trans.plot(plunger_voltage, transport_current, linestyle='-', label=label)
        except Exception as e:
            print(f"❌ Error reading JSON file: {e}")

    # === Finalize Transport Plot ===
    ax_trans.set_xlabel("Plunger (V)")
    ax_trans.set_ylabel("Transport Current (pA)")
    ax_trans.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_trans.minorticks_on()
    ax_trans.legend(fontsize=8)
    ax_trans.set_xlim([0, 2])
    if ylim_transport:
        ax_trans.set_ylim(ylim_transport)

    # === Finalize Leakage Plot ===
    ax_leak.set_xlabel("Plunger (V)")
    ax_leak.set_ylabel("Leakage Current (nA)")
    ax_leak.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_leak.minorticks_on()
    ax_leak.legend(fontsize=8)
    ax_leak.set_xlim([0, 2])
    if ylim_leakage:
        ax_leak.set_ylim(ylim_leakage)

    plt.tight_layout()
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, format='pdf', dpi=300)
        print(f"✅ Saved combined plot to {save_to_file}")

    plt.show()

def plot_multiple_topbarrier_sweeps_with_leakage_v4(
        db_file_list, 
        json_file=None, 
        save_to_file=None, 
        device_labels=None,
        ylim_transport=None,
        ylim_leakage=None,
        xlim_trans =None,
        xlim_leak =None
):
    """
    Plot multiple plunger sweeps and their corresponding leakage in separate plots.

    Parameters:
    - db_file_list : list of (db_path, exp_id, dac_key, set_side, xlabel)
    - json_file : str : Path to the JSON file containing plunger sweeps.
    - save_to_file : str : Path to save the combined plot.
    - device_labels : list of strings for legend labels.
    """
    fig, (ax_trans, ax_leak) = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(db_file_list)))  # Use a color map for unique colors

    # === Plot from QCoDeS Databases ===
    for idx, (db_path, exp_id, dac_key, set_side, xlabel) in enumerate(db_file_list):
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            sample_name = extract_sample_name(db_path)
            color = colors[idx]  # Use consistent color for transport and leakage

            # Use the specified DAC key for plunger voltage
            if set_side.lower() == 'rset':
                trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
                leak_block = meas["data"].get("keithley_right_curr", {}).get("keithley_right_curr", None)
            elif set_side.lower() == 'lset':
                trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
                leak_block = meas["data"].get("keithley_left_curr", {}).get("keithley_left_curr", None)
            else:
                continue

            # Ensure DAC key exists
            if dac_key not in trans_block:
                print(f"❌ DAC key '{dac_key}' not found in {sample_name} {set_side.upper()}")
                continue

            # Extract transport data
            x_trans = trans_block[dac_key]
            y_trans_key = next((k for k in trans_block if "lockin" in k or "current" in k), None)
            if y_trans_key is None:
                print(f"❌ No transport current key found for {sample_name} {set_side.upper()}")
                continue

            y_trans = trans_block[y_trans_key] * 1e12  # Convert to pA
            label = device_labels[idx] if device_labels else f"Dev {idx+1}"

            # Detect forward/backward for transport
            x_trans_sep, y_trans_sep, dir_trans = separate_up_down(x_trans, y_trans)
            if len(dir_trans) > 1:
                for xs, ys, direction in zip(x_trans_sep, y_trans_sep, dir_trans):
                    linestyle = '--' if direction == 1 else '-'
                    ax_trans.plot(xs, ys, label=f"{label} (Forward)" if direction == 1 else f"{label} (Backward)", 
                                  linestyle=linestyle, color=color)
            else:
                ax_trans.plot(x_trans, y_trans, label=label, linestyle='-', color=color)

            # Plot leakage if available
            if leak_block is not None:
                y_leak = leak_block * 1e9  # Convert to pA
                x_leak = x_trans  # Use the same x-axis for leakage

                # Detect forward/backward for leakage
                x_leak_sep, y_leak_sep, dir_leak = separate_up_down(x_leak, y_leak)
                if len(dir_leak) > 1:
                    for xs, ys, direction in zip(x_leak_sep, y_leak_sep, dir_leak):
                        linestyle = '-' if direction == 1 else '--'
                        ax_leak.plot(xs, ys, label=f"{label} (Leak Forward)" if direction == 1 else f"{label} (Leak Backward)", 
                                     linestyle=linestyle, color=color)
                else:
                    ax_leak.plot(x_leak, y_leak, label=f"{label} (Leak)", linestyle='-', color=color)

        except Exception as e:
            print(f"❌ Error reading {db_path}, exp_id {exp_id}: {e}")

    # === Plot from JSON file ===
    if json_file:
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)

            for data in json_data.get("RSET", []):
                x_values = np.array(data.get("x", []))
                y_values = np.array([y * 1e12 for y in data.get("y", [])])
                ax_trans.plot(x_values,
                               y_values, 
                               linestyle='-', color='red', 
                               label='Dev 14 (Backward)')
        except Exception as e:
            print(f"❌ Error reading JSON file: {e}")

    # === Finalize Transport Plot ===
    ax_trans.set_xlabel("Top Barrier (V)")
    ax_trans.set_ylabel("Transport Current (pA)")
    ax_trans.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_trans.minorticks_on()
    ax_trans.legend(fontsize=8)
    ax_trans.set_xlim(xlim_trans)
    if ylim_transport:
        ax_trans.set_ylim(ylim_transport)

    # === Finalize Leakage Plot ===
    ax_leak.set_xlabel("Top Barrier (V)")
    ax_leak.set_ylabel("Leakage Current (nA)")
    ax_leak.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_leak.minorticks_on()
    ax_leak.legend(fontsize=8)
    ax_leak.set_xlim(xlim_leak)
    if ylim_leakage:
        ax_leak.set_ylim(ylim_leakage)

    plt.tight_layout()
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, format='pdf', dpi=300)
        print(f"✅ Saved combined plot to {save_to_file}")

    plt.show()

def plot_multiple_bottombarrier_sweeps_with_leakage_v4(
        db_file_list, 
        json_file=None, 
        save_to_file=None, 
        device_labels=None,
        ylim_transport=None,
        ylim_leakage=None
):
    """
    Plot multiple plunger sweeps and their corresponding leakage in separate plots.

    Parameters:
    - db_file_list : list of (db_path, exp_id, dac_key, set_side, xlabel)
    - json_file : str : Path to the JSON file containing plunger sweeps.
    - save_to_file : str : Path to save the combined plot.
    - device_labels : list of strings for legend labels.
    """
    fig, (ax_trans, ax_leak) = plt.subplots(1, 2, figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(db_file_list)))  # Use a color map for unique colors

    # === Plot from QCoDeS Databases ===
    for idx, (db_path, exp_id, dac_key, set_side, xlabel) in enumerate(db_file_list):
        try:
            meas = readout_database("data", db_path, exp_id - 1)
            sample_name = extract_sample_name(db_path)
            color = colors[idx]  # Use consistent color for transport and leakage

            # Use the specified DAC key for plunger voltage
            if set_side.lower() == 'rset':
                trans_block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
                leak_block = meas["data"].get("keithley_right_curr", {}).get("keithley_right_curr", None)
            elif set_side.lower() == 'lset':
                trans_block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
                leak_block = meas["data"].get("keithley_left_curr", {}).get("keithley_left_curr", None)
            else:
                continue

            # Ensure DAC key exists
            if dac_key not in trans_block:
                print(f"❌ DAC key '{dac_key}' not found in {sample_name} {set_side.upper()}")
                continue

            # Extract transport data
            x_trans = trans_block[dac_key]
            y_trans_key = next((k for k in trans_block if "lockin" in k or "current" in k), None)
            if y_trans_key is None:
                print(f"❌ No transport current key found for {sample_name} {set_side.upper()}")
                continue

            y_trans = trans_block[y_trans_key] * 1e12  # Convert to pA
            label = device_labels[idx] if device_labels else f"Dev {idx+1}"

            # Detect forward/backward for transport
            x_trans_sep, y_trans_sep, dir_trans = separate_up_down(x_trans, y_trans)
            if len(dir_trans) > 1:
                for xs, ys, direction in zip(x_trans_sep, y_trans_sep, dir_trans):
                    linestyle = '-' if direction == 1 else '--'
                    ax_trans.plot(xs, ys, label=f"{label} (Forward)" if direction == 1 else f"{label} (Backward)", 
                                  linestyle=linestyle, color=color)
            else:
                ax_trans.plot(x_trans, y_trans, label=label, linestyle='-', color=color)

            # Plot leakage if available
            if leak_block is not None:
                y_leak = leak_block * 1e9  # Convert to pA
                x_leak = x_trans  # Use the same x-axis for leakage

                # Detect forward/backward for leakage
                x_leak_sep, y_leak_sep, dir_leak = separate_up_down(x_leak, y_leak)
                if len(dir_leak) > 1:
                    for xs, ys, direction in zip(x_leak_sep, y_leak_sep, dir_leak):
                        linestyle = '-' if direction == 1 else '--'
                        ax_leak.plot(xs, ys, label=f"{label} (Leak Forward)" if direction == 1 else f"{label} (Leak Backward)", 
                                     linestyle=linestyle, color=color)
                else:
                    ax_leak.plot(x_leak, y_leak, label=f"{label} (Leak)", linestyle='-', color=color)

        except Exception as e:
            print(f"❌ Error reading {db_path}, exp_id {exp_id}: {e}")

    # === Finalize Transport Plot ===
    ax_trans.set_xlabel("Bottom Barrier (V)")
    ax_trans.set_ylabel("Transport Current (pA)")
    ax_trans.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_trans.minorticks_on()
    ax_trans.legend(fontsize=8)
    ax_trans.set_xlim([-0.7, 1.5])
    if ylim_transport:
        ax_trans.set_ylim(ylim_transport)

    # === Finalize Leakage Plot ===
    ax_leak.set_xlabel("Bottom Barrier (V)")
    ax_leak.set_ylabel("Leakage Current (nA)")
    ax_leak.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax_leak.minorticks_on()
    ax_leak.legend(fontsize=8)
    ax_leak.set_xlim([-0.7, 2])
    # if ylim_leakage:
    #     ax_leak.set_ylim(ylim_leakage)

    # plt.tight_layout()
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, format='pdf', dpi=300)
        print(f"✅ Saved combined plot to {save_to_file}")
    # plt.figure(figsize=(5, 4))
    # plt.plot(y_slice, z_slice_y)

    # plt.xlabel("Bottom Barrier Voltage (V)")
    # plt.ylabel("Transport Current (pA)")
    # # plt.title(f"1D Slice at Y={fixed_voltage_y} V")
    # plt.grid(True, which='both', linestyle=':', color='#CCCCCC')
    # plt.minorticks_on()
    # plt.show()

    plt.show()

# --- 2D Barrier vs Barrier Sweep ---
def plot_2d_barrier_barrier(db_path, 
                            exp_id, 
                            set_side='rset',
                            TB_gate=None,
                            BB_gate=None,
                            xlabel = None,
                            ylabel = None,
                            ylim = None,
                            xlim = None,
                            save_path =None):
    """
    Plot a 2D barrier-barrier sweep for a SET device.

    Parameters:
    - db_path : str
        Path to the QCoDeS .db file
    - exp_id : int
        Experiment ID in the database
    - set_side : 'left' or 'right'
        Which SET to plot
    - TB_gate : str
        Top barrier gate column name
    - BB_gate : str
        Bottom barrier gate column name
    """
    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    try:
        if set_side.lower() == 'rset':
            signal = meas["data"]["lockin_down_R"]
        elif set_side.lower() == 'lset':
            signal = meas["data"]["lockin_up_R"]
        else:
            raise ValueError("set_side must be 'lset' or 'rset'.")

        z = signal["lockin_down_R"] if set_side.lower() == 'rset' else signal["lockin_up_R"]
        x = signal[BB_gate]
        y = signal[TB_gate]

    except KeyError as e:
        print(f"❌ Missing data column: {e}")
        return

    # Infer shape for reshaping (assumes 2D meshgrid sweep)
    try:
        x_unique = len(np.unique(x))
        y_unique = len(np.unique(y))
        x_reshaped = x.reshape(y_unique, x_unique)
        y_reshaped = y.reshape(y_unique, x_unique)
        z_reshaped = (z * 1e12).reshape(y_unique, x_unique)
    except Exception as e:
        print(f"❌ Error reshaping data: {e}")
        return
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    c = ax.pcolormesh(x_reshaped, y_reshaped, z_reshaped, shading='auto')
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Transport Current (pA)")

    ax.set_xlabel(f"{xlabel} (V)")
    ax.set_ylabel(f"{ylabel} (V)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # ax.set_title(f"{set_side.upper()} SET Barrier-Barrier Sweep\n{title}")
    ax.minorticks_on()
    # ax.grid(True, which= 'both', linestyle=':', color='#CCCCCC')
    # plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")
    plt.show()

def plot_2d_barrier_barrier_v2(db_path, exp_id, set_side='rset', save_path=None):

    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    # === Select measurement block
    if set_side.lower() == 'rset':
        data_block = meas["data"]["lockin_down_R"]
        z_key = "lockin_down_R"
    elif set_side.lower() == 'lset':
        data_block = meas["data"]["lockin_up_R"]
        z_key = "lockin_up_R"
    else:
        raise ValueError("set_side must be 'lset' or 'rset'")

    # === Auto-detect gate voltages (first two DAC keys)
    gate_keys = [k for k in data_block if "dac" in k and "volt" in k]
    if len(gate_keys) < 2:
        print("❌ Could not find two DAC voltage axes.")
        return

    x = data_block[gate_keys[0]]
    y = data_block[gate_keys[1]]
    z = data_block[z_key] * 1e12  # Convert to pA
    print(data_block)

    try:
        # === Infer square reshaping
        length = int(np.sqrt(len(z)))
        x = x.reshape(length, length)
        y = y.reshape(length, length)
        z = z.reshape(length, length)
        # print(x,y,z)
    except Exception as e:
        print(f"❌ Reshape failed: {e}")
        return
    # print(z)

    # === Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.pcolormesh(x, y, z, shading='auto', cmap='viridis')
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Transport Current (pA)")

    ax.set_xlabel("Bottom Barrier (V)")
    ax.set_ylabel("Top Barrier (V)")
    ax.grid(True, linestyle=':', color='#CCCCCC')
    ax.minorticks_on()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")
    plt.show()

def plot_2d_barrier_barrier_v3(db_path, exp_id, set_side='rset', save_path=None):

    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    # === Select measurement block
    if set_side.lower() == 'rset':
        data_block = meas["data"]["lockin_down_R"]
        z_key = "lockin_down_R"
    elif set_side.lower() == 'lset':
        data_block = meas["data"]["lockin_up_R"]
        z_key = "lockin_up_R"
    else:
        raise ValueError("set_side must be 'lset' or 'rset'")

    # === Auto-detect gate voltages (first two DAC keys)
    gate_keys = [k for k in data_block if "dac" in k and "volt" in k]
    if len(gate_keys) < 2:
        print("❌ Could not find two DAC voltage axes.")
        return

    x = data_block[gate_keys[0]]
    y = data_block[gate_keys[1]]
    z = data_block[z_key] * 1e12  # Convert to pA

    # === Automatically determine the correct 2D shape ===
    unique_x = len(np.unique(x))
    unique_y = len(np.unique(y))
    print(f"Detected Grid Shape: ({unique_y}, {unique_x})")

    try:
        x = x.reshape((unique_y, unique_x))
        y = y.reshape((unique_y, unique_x))
        z = z.reshape((unique_y, unique_x))
    except Exception as e:
        print(f"❌ Reshape failed: {e}")
        return

    # === Plotting the 2D map ===
    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.pcolormesh(x, y, z, shading='auto', cmap='viridis')
    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Transport Current (pA)")

    ax.set_xlabel("Top Barrier (V)")
    ax.set_ylabel("Bottom Barrier (V)")
    # ax.set_title(f"{set_side.upper()} SET 2D Barrier-Barrier Map")
    ax.minorticks_on()
    ax.grid(True, linestyle=':', color='#CCCCCC')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")
    plt.show()

# --- 2D Barrier vs Plunger Sweep ---
def plot_2d_barrier_plunger(db_path, exp_id, set_side='RSET', plunger_dac = None):
    meas = readout_database("data", db_path, exp_id - 1)
    title = meas["exp_name"]

    if set_side.lower() == 'RSET':
        z = meas["data"]["lockin_down_R"]["lockin_down_R"]
        y = meas["data"]["lockin_down_R"][f"{plunger_dac}"]
        x = meas["data"]["lockin_down_R"]["dac_Slot1_Chan1_volt"]
    elif set_side.lower() == 'LSET':
        z = meas["data"]["lockin_up_R"]["lockin_up_R"]
        y = meas["data"]["lockin_up_R"][f"{plunger_dac}"]
        x = meas["data"]["lockin_up_R"]["dac_Slot4_Chan1_volt"]
    else:
        raise ValueError("set_side must be 'left' or 'right'.")

    length = int(np.sqrt(len(z)))
    fig, ax = plt.subplots()
    p = ax.pcolor(x.reshape(length,length), y.reshape(length,length), (z*1e12).reshape(length,length))
    ax.set_xlabel("Barrier Voltage (V)")
    ax.set_ylabel("Plunger Voltage (V)")
    cb = fig.colorbar(p, ax=ax)
    cb.set_label("Lock-in Current (pA)")
    ax.set_title(f"{set_side.upper()} SET Barrier-Plunger Sweep: {title}")
    plt.show()

def plot_2d_barrier_plunger_v2(db_path, exp_id, set_side='rset',
                            barrier_gate=None, plunger_gate=None,
                            save_path=None):

    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    # === Choose block
    set_side = set_side.lower()
    if set_side == 'rset':
        block = meas["data"].get("lockin_down_R", meas["data"].get("mfli2_current"))
        z_key = "lockin_down_R" if "lockin_down_R" in block else "mfli2_current"
    elif set_side == 'lset':
        block = meas["data"].get("lockin_up_R", meas["data"].get("mfli1_current"))
        z_key = "lockin_up_R" if "lockin_up_R" in block else "mfli1_current"
    else:
        raise ValueError("set_side must be 'lset' or 'rset'")

    # === Use user-specified gates if provided, else auto-detect
    if barrier_gate is not None and plunger_gate is not None:
        x_key, y_key = barrier_gate, plunger_gate
    else:
        volt_keys = [k for k in block if "volt" in k.lower()]
        if len(volt_keys) < 2:
            print("❌ Not enough voltage axes found.")
            return
        x_key, y_key = volt_keys[:2]  # Auto fallback
        print(f"⚠️ Using auto-detected keys: X = {x_key}, Y = {y_key}, Z = {z_key}")

    try:
        x = np.array(block[x_key])
        y = np.array(block[y_key])
        z = np.array(block[z_key]) * 1e12  # convert to pA

        # Reshape
        length = int(np.sqrt(len(z)))
        x = x[:length**2].reshape(length, length)
        y = y[:length**2].reshape(length, length)
        z = z[:length**2].reshape(length, length)
        # print(x,y)

    except Exception as e:
        print(f"❌ Error reshaping: {e}")
        return

    # === Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    pcm = ax.pcolormesh(x, y, z, shading='auto', cmap='viridis')
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("Current (pA)")

    ax.set_xlabel(f"Barrier (V)")
    ax.set_ylabel(f"Plunger (V)")
    ax.minorticks_on()
    ax.grid(True, linestyle=':', color='#CCCCCC')
    # ax.minorticks_on()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")

    plt.show()

def plot_1d_slices_from_2d_v2(db_path, 
                           exp_id, 
                           set_side='rset', 
                           TB_gate=None,
                           BB_gate=None,
                           fixed_voltage_x=None,
                           fixed_voltage_y=None,  
                           save_path=None,
                           save_json_path=None):
    """
    Plot two 1D slices from a 2D barrier-barrier scan at a specified voltage.
    """
    # Load the measurement
    meas = readout_database("data", db_path, exp_id - 1)
    title = meas.get("exp_name", f"exp_id {exp_id}")

    # === Select measurement block
    if set_side.lower() == 'rset':
        data_block = meas["data"]["lockin_down_R"]
        z_key = "lockin_down_R"
    elif set_side.lower() == 'lset':
        data_block = meas["data"]["lockin_up_R"]
        z_key = "lockin_up_R"
    else:
        raise ValueError("set_side must be 'lset' or 'rset'")

    # === Auto-detect gate voltages (first two DAC keys)
    x = data_block[BB_gate]
    y = data_block[TB_gate]
    z = data_block[z_key] * 1e12  # Convert to pA

    try:
        x_unique = len(np.unique(x))
        y_unique = len(np.unique(y))
        x = x.reshape(y_unique, x_unique)
        y = y.reshape(y_unique, x_unique)
        z = z.reshape(y_unique, x_unique)
    except Exception as e:
        print(f"❌ Error reshaping data: {e}")
        return

    # === Extract 1D Slice at X (along Y-axis)
    if fixed_voltage_y is not None:
        idx_x = np.abs(y[0, :] - fixed_voltage_y).argmin()
        y_slice = x[:, idx_x]
        z_slice_y = z[:, idx_x]
        plt.figure(figsize=(5, 4))
        plt.plot(y_slice, z_slice_y)
        plt.xlabel("Barrier Voltage (V)")
        plt.ylabel("Transport Current (pA)")
        plt.title(f"1D Slice at Y={fixed_voltage_y} V")
        plt.grid(True, which='both', linestyle=':', color='#CCCCCC')
        plt.minorticks_on()
        plt.show()

    # === Extract 1D Slice at Y (along X-axis)
    if fixed_voltage_x is not None:
        idx_y = np.abs(x[:, 0] - fixed_voltage_x).argmin()
        x_slice = y[idx_y, :]
        z_slice_x = z[idx_y, :]
        plt.figure(figsize=(5, 4))
        plt.plot(x_slice, z_slice_x)
        plt.xlabel("Plunger Voltage (V)")
        plt.ylabel("Transport Current (pA)")
        plt.title(f"1D Slice at X={fixed_voltage_x} V")
        plt.grid(True, which='both', linestyle=':', color='#CCCCCC')
        plt.minorticks_on()
        plt.show()
        sample_name =extract_sample_name(db_path)
    
        # === Save to JSON (Only if save_json_path is provided)
        if save_json_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
            
            # Load existing data if the JSON file exists
            if os.path.exists(save_json_path):
                with open(save_json_path, "r", encoding="utf-8") as json_file:
                    try:
                        data = json.load(json_file)
                    except json.JSONDecodeError:
                        print("❌ Existing JSON file is corrupted. Creating a new one.")
                        data = {}
            else:
                data = {}

            # Construct the new sample data
            new_sample_data = {
                "sample_name": sample_name,
                "exp_id": exp_id,
                "set_side": set_side,
                "fixed_voltage_x": fixed_voltage_x,
                "plunger_voltage": x_slice.tolist(),
                "transport_current": z_slice_x.tolist()
            }

            # Check for duplicates based on the unique combination of sample name, exp_id, and set_side
            unique_key = f"{sample_name}_{exp_id}_{set_side}"
            data[unique_key] = new_sample_data

            # Save updated data to JSON in text mode
            with open(save_json_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)
            
            print(f"✅ Saved 1D slice data to {save_json_path}")

    
    # === Save Plot (Only if save_path is provided)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"✅ Saved plot to {save_path}")

def plot_1d_slice_from_json(json_path):
    """
    Load a saved 1D slice from a JSON file and plot it.

    Parameters:
    - json_path : str : Path to the JSON file containing the 1D slice data.
    """
    try:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
        
        exp_id = data.get("exp_id")
        set_side = data.get("set_side")
        fixed_voltage_x = data.get("fixed_voltage_x")
        plunger_voltage = data.get("plunger_voltage")
        transport_current = data.get("transport_current")
        
        # === Plotting the loaded data ===
        plt.figure(figsize=(6, 4))
        plt.plot(plunger_voltage, transport_current, label=f"Slice at X={fixed_voltage_x} V")
        plt.xlabel("Plunger Voltage (V)")
        plt.ylabel("Transport Current (pA)")
        plt.title(f"1D Slice from JSON - {set_side.upper()} (Exp ID: {exp_id})")
        plt.grid(True, linestyle=':', color='#CCCCCC')
        plt.minorticks_on()
        plt.tight_layout()
        plt.show()

        print(f"✅ Successfully plotted 1D slice from {json_path}")

    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in {json_path}")
    except Exception as e:
        print(f"❌ Error loading or plotting JSON data: {e}")

