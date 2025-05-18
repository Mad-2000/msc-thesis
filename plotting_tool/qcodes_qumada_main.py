#%% Imports

from plotting_functions import *


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



# %%  #############################   PANDAS  ######################################

#%%  #################   Devices Without Fanout FET Accumulation   #################
accu_yrange = [0,200]
accu_xrange = [0,2]
leak_xrange = [0,1]
leak_yrange = [-10,12000]


db_table_pairs_nofanout = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A2_20240723.db', 
     'results-1-1', 
     'LSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A3_20240729.db', 
     'results-1-1', 
     'LSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A3_20240729.db',
      'results-2-1', 
      'RSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_B4_20240809.db', 
     'results-1-1', 
     'LSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_B4_20240809.db', 
     'results-2-1', 
     'RSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db', 
     'results-1-1', 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db', 
     'results-2-2', 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S10_C4_20240809.db', 
     'results-1-1', 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S10_C4_20240809.db', 
     'results-2-1', 
     'RSET'),
]

# plot_multiple_accumulation_with_pfiles(db_table_pairs_nofanout, 
#                                        save_to_file=rf'z:\Data\Plotting\Plots\accumulation_withoutfanoutFET2.pdf',
#                                        accu_xrange = accu_xrange, 
#                                        accu_yrange = accu_yrange,
#                                         leak_xrange = leak_xrange,
#                                         leak_yrange = leak_yrange)

# plot_multiple_accumulation_with_pfiles(db_table_pairs_nofanout, )

#%%   #################  Devices With dependent Fanout FET ################



save_to_file_fanout = rf'z:\Data\Plotting\Plots\accumulation_fanoutFET7.pdf'
json_pfile_paths_fanout = [r'z:\Data\Plotting\S13_D4_pfile_data.json']



db_table_pairs_fanout = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_C2_20240723.db', 
     2, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db', 
     2, 
     'LSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_B2_20241002.db', 
    #  'results-1-5', 
    #  'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     3, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db', 
     3, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
     3, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db', 
     33, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B3_20240912.db', 
     4, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B3_20240912.db', 
     12, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D2_20240823.db', 
     2, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     8, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_C1_20240723.db', 
     7, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     14, 
     'RSET'),
     ]

accu_yrange = [0,200]
accu_xrange = [0,2]
leak_xrange = [0,2]
leak_yrange = [-100,50]
db_sample_accum= [     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     8, 
     'LSET')]
plot_multiple_accumulation_with_pfiles(db_table_pairs_fanout, 
                                       json_pfile_paths_fanout, 
                                       save_to_file_fanout,
                                       accu_xrange = accu_xrange, 
                                       accu_yrange = accu_yrange,
                                        leak_xrange = leak_xrange,
                                        leak_yrange = leak_yrange)

#%%    Curve Fitting 


db_fit_sample=[
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     'results-1-8', 
     'LSET'),]

json_pfile_paths_fanout = [r'z:\Data\Plotting\S13_D4_pfile_data.json']

fit_and_save_accumulation_fits(
    db_fit_sample,    
    save_json_path=r"z:\Data\Plotting\accumulation_fits_batch26_better2.json",
    save_plot_folder=r"z:\Data\Plotting\Plots\Fits",
    # json_pfile_paths=json_pfile_paths_fanout
)
#%%    Accumulation Threshold vth with +ve SG
json_path = r"z:\Data\Plotting\accumulation_fits_batch26_better.json"

# Optional: Save path
save_path1 = r"z:\Data\Plotting\Plots\Fits\vth_summary_plot_nottitle.pdf"
save_path = r"z:\Data\Plotting\Plots\Fits\vth_summary_plot_with_SG.pdf"
plot_vth_from_fit_json_by_dev_number(json_path, save_path1)
plot_vth_from_fit_json_with_SG(json_path, save_path)


#%%   ################################   QCODES   ###############################
#%% Accessing all db files ina folder

# === Define Directory Path ===
db_directory = r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile'

# === List All .db Files ===
db_files = [f for f in os.listdir(db_directory) if f.endswith('.db')]

# === Run Your Function on Each File ===
for db_file in db_files:
    db_path = os.path.join(db_directory, db_file)
    print(f"Processing: {db_path}")
    
    # Run your function (replace 'find_valid_result_tables' with the actual function name)
    result_tables = find_valid_result_tables(db_path)
    print(f"Valid Result Tables in {db_path}:\n", result_tables)



#%%  ################ Without Fanout FET ###########################
db_table_pairs_nofanout = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A2_20240723.db', 
     0, 
     'LSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A2_20240723.db', 
     1, 
     'RSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A3_20240729.db', 
     0, 
     'LSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_A3_20240729.db',
      1, 
      'RSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_B4_20240809.db', 
     0, 
     'LSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_B4_20240809.db', 
     1, 
     'RSET'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db', 
     0, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db', 
     3, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S10_C4_20240809.db', 
     0, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S10_C4_20240809.db', 
     3, 
     'RSET'),
]



save_to_file_nofanout = rf'z:\Data\Plotting\Plots\combined_accumulation_without_fanoutFET_qcodes.pdf'
# json_pfile_paths_fanout = [r'z:\Data\Plotting\S13_D4_pfile_data.json']

accu_yrange = [0,200]
accu_xrange = [0,2]
leak_xrange = [0,1]
leak_yrange = [-0.1,15]


plot_multiple_accumulation_qcodes(db_table_pairs_nofanout,
                                   json_pfile_paths= None,
                                    save_to_file=save_to_file_nofanout,
                                    accu_xrange=accu_xrange,
                                    accu_yrange=accu_yrange,
                                    leak_xrange=leak_xrange,
                                    leak_yrange=leak_yrange,
                                  )
#%% Time Stability

# (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db', 
#      4, 
#      'RSET')
# stability

db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240730.db'
exp_id = 2
set_side = 'rset'
plot_time_stability_v2(db_path, 
                       exp_id,
                         set_side, 
                       save_path=rf"z:\Data\Plotting\Plots\without_fanout_FET\stability_{set_side}_{exp_id}.pdf")
#%%  Plunger sweep

save_folder = "z:\Data\Plotting\Plots\without_fanout_FET"
S9_C3_sweeps = [
    (r'c:\Users\Suresh\Documents\Three Layer SET\Data\Three layer SET\09.11.2024\Batch26_qcodes\GDSET_noSmile\S9_C3_20240730.db', 
     4, 
     "dac2_Slot0_Chan3_volt", 
     "right",  
     save_folder),

    (r'c:\Users\Suresh\Documents\Three Layer SET\Data\Three layer SET\09.11.2024\Batch26_qcodes\GDSET_noSmile\S9_C3_20240730.db', 
     5, 
     "dac2_Slot0_Chan3_volt", 
     "right",  
     save_folder),]

for db_path, exp_id, gate, side, save_folder in S9_C3_sweeps:
    plot_gate_with_sweep_direction(
        db_path=db_path,
        exp_id=exp_id,
        gate_name=gate,
        set_side=side,
        save_folder=save_folder,
        ylim=(-50, 150),
        xlabel="Barrier"
    )

#%% BB plot
db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db'
exp_id = 12
set_side = 'rset'
plot_2d_barrier_barrier_v2(db_path, 
                        exp_id, 
                        set_side,
                        # TB_gate='dac2_Slot0_Chan3_volt',

                        # BB_gate='dac2_Slot0_Chan1_volt',
                        save_path=rf"z:\Data\Plotting\Plots\Sample Plots\BBsweep_{set_side}_{exp_id}.pdf")


# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db'
# exp_id = 10
# set_side = 'rset'
# # plot_2d_barrier_barrier_v2(db_path, 
# #                         exp_id, 
# #                         set_side,
# #                         # TB_gate='dac2_Slot0_Chan3_volt',

# #                         # BB_gate='dac2_Slot0_Chan1_volt',
# #                         save_path=rf"z:\Data\Plotting\Plots\Sample Plots\BBsweep_{set_side}_{exp_id}.pdf")

# plot_2d_barrier_barrier(db_path, 
#                         exp_id, 
#                         set_side,
#                         TB_gate='dac2_Slot0_Chan3_volt',

#                         BB_gate='dac2_Slot0_Chan1_volt',
#                         save_path=rf"z:\Data\Plotting\Plots\Sample Plots\BBsweep_{set_side}_{exp_id}.pdf")


#%% BP plot
# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db'
# exp_id = 10
# set_side = 'rset'
# sample_name=extract_sample_name(db_path)
# plot_2d_barrier_barrier(db_path, 
#                         exp_id, 
#                         set_side,
#                         BB_gate='dac2_Slot0_Chan3_volt',
#                         TB_gate='dac2_Slot0_Chan2_volt',
#                         xlim = [0.45,None],
#                         ylim=[0.95, None],
#                         xlabel= 'Bottom Barrier',
#                         ylabel='Plunger',
#                         save_path=rf"z:\Data\Plotting\Plots\wihout_fanout_FET\{sample_name}_BP2sweep_{set_side}_{exp_id}.pdf")


db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240730.db'

exp_id = 9
set_side = 'rset'
sample_name=extract_sample_name(db_path)
plot_2d_barrier_barrier(db_path, 
                        exp_id, 
                        set_side,
                        BB_gate='dac2_Slot0_Chan3_volt',
                        TB_gate='dac2_Slot0_Chan2_volt',
                        # xlim = [0.45,None],
                        # ylim=[0.95, None],
                        xlabel= 'Bottom Barrier',
                        ylabel='Plunger',
                        save_path=rf"z:\Data\Plotting\Plots\wihout_fanout_FET\withoutfanout_{sample_name}_BP2sweep_{set_side}_{exp_id}.pdf")





#%%  ################## with depenedent Fanout FET #################################

#%% Combined Accumulation Plots 
db_table_pairs_fanout_qcodes = [
    (r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_C2_20240723.db', 
     1, 
     'RSET'),
     (r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db', 
     0, 
     'LSET'),
     (r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     2, 
     'RSET'),
     (r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db', 
     2, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
     2, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db', 
     32, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B3_20240912.db', 
     3, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B3_20240912.db', 
     11, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D2_20240823.db', 
     1, 
     'RSET'), 
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     7, 
     'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_C1_20240723.db', 
     6, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     13, 
     'RSET'),
     ]

db_sample_accum= [     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     8, 
     'LSET')]

save_to_file_fanout = rf'z:\Data\Plotting\Plots\withfanout_combined_accumulation_fanoutyscale4.pdf'
json_pfile_paths_fanout = [r'z:\Data\Plotting\S13_D4_pfile_data.json']

accu_yrange = [0,2000]
accu_xrange = [0,1]
leak_xrange = [0,2]
leak_yrange = [-5, 20]


plot_multiple_accumulation_qcodes(db_table_pairs_fanout_qcodes,
                                  json_pfile_paths_fanout,
                                  save_to_file_fanout,
                                      accu_xrange=[0, 2],
                                    accu_yrange=accu_yrange,
                                    leak_xrange=[0, 2],
                                    leak_yrange= leak_yrange,
                                  )

#%% Sample Accum Plots

# S10 D4

db_path = r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db'
sample_name = extract_sample_name(db_path)
exp_id = 8
set_side = 'left'
plot_accumulation_v2(db_path,
                     exp_id,
                     set_side='left',
                     save_path= rf'z:\Data\Plotting\Plots\Accumulation\{sample_name}_accum_{exp_id}.pdf',
                     ylim_transport=[0,200],
                     ylim_leakage=[-60,0])



# S8_A3
# Left SET Accumulation
# lset_acc = False
# exp_id = 5
# meas = readout_database("data",
#                         r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db",exp_id - 1)
# title = meas["exp_name"]
# lset_acc_y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
# lset_acc_y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
# lset_acc_x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
# lset_acc_x2 = meas["data"]["lockin_up_R"]["keithley_left_volt"]
if lset_acc:
    fig, ax = plt.subplots()
    ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage current")
    ax.plot(lset_acc_x2, lset_acc_y2*1.e12, label="Transport current")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (pA)")
    ax.set_title("Left SET Accumulation")
    ax.legend()
    plt.show()

# Right SET Accumulation

rset_acc = True
exp_id = 3
meas = readout_database("data",
                        r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db", 
                        exp_id - 1)
title = meas["exp_name"]
rset_acc_y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
rset_acc_y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
rset_acc_x1 = meas["data"]["keithley_right_curr"]["keithley_right_volt"]
rset_acc_x2 = meas["data"]["lockin_down_R"]["keithley_right_volt"]



save_path = rf'z:\Data\Plotting\Plots\Sample Plots\S8_A3_accum_{exp_id}.pdf'
# Right SET Accumulation
if rset_acc:
    fig, ax = plt.subplots()
    ax.plot(rset_acc_x1, rset_acc_y1*1.e12, label="Leakage current")
    ax.plot(rset_acc_x2, rset_acc_y2*1.e12, label="Transport current")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (pA)")
    # ax.set_title(title)
    ax.legend()
    plt.ylim([-50, 200])
    plt.tight_layout()
    # plt.grid()
    ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
    # ax.minorticks_on()
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()

#  S12_B4
db_path = r'c:\Users\Suresh\Documents\Three Layer SET\Data\Three layer SET\09.11.2024\Batch26 - combined\GDSET_smile\S12_B4_20240927.db'
sample_name = extract_sample_name(db_path)

exp_id = 3
plot_accumulation_v2(db_path,
                     exp_id,
                     set_side='left',
                     save_path= rf'z:\Data\Plotting\Plots\Screening_gate\{sample_name}_accum_{exp_id}.pdf',
                     ylim_transport=[0,200],
                     ylim_leakage=[-0.1,2])

# Accumulation S12_B4 high leakage

db_path = r'c:\Users\Suresh\Documents\Three Layer SET\Data\Three layer SET\09.11.2024\Batch26 - combined\GDSET_smile\S12_B4_20240927.db'
sample_name = extract_sample_name(db_path)

exp_id = 3
plot_accumulation_v2(db_path,
                     exp_id,
                     set_side='left',
                     save_path= rf'z:\Data\Plotting\Plots\Screening_gate\{sample_name}_accum_high_leakage_{exp_id}.pdf',
                     ylim_transport=[0, 0.6],
                     ylim_leakage=[-0.1,5])

#%% Curve fitting using PANDAS given above

#%% Box plot for accumulation Vth thresholds
# z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_with_fanout\S8_A3_20240927.db
# === Load the JSON file ===
json_path = r"z:\Data\Plotting\accumulation_fits_batch26_better.json"  # Replace with your actual file path
with open(json_path, "r") as f:
    data = json.load(f)


# === Ensure data is a list of dictionaries ===
if isinstance(data, list):
    # === Extract Vth and Vth_err values ===
    vth_values = [entry.get('Vth') for entry in data if 'Vth' in entry]
    vth_errors = [entry.get('Vth_err') for entry in data if 'Vth_err' in entry]
else:
    print("JSON data is not in the expected list format.")

# === Prepare Data for Box Plot ===
batch_names = ["Batch 26"]
all_values = [vth_values]

# === Create Box Plot ===
fig, ax = plt.subplots(figsize=(6, 8))

# Box plot
bp = ax.boxplot(all_values, patch_artist=True, widths=0.5)
colors = ['skyblue']  # Custom color

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# === Plot Mean with Error Bars ===
mean_vth = round(np.mean(df['Vth']), 2)
std_vth = round(np.std(df['Vth']), 2)
ax.errorbar(1, mean_vth, 
            yerr=std_vth, 
            fmt='o', 
            color='red', 
            label=f'Mean ({mean_vth}) ± Std ({std_vth})',
            elinewidth=3,
            capsize=8)

# === Plot Individual Data Points ===
for i, vth, err in zip(range(len(vth_values)), vth_values, vth_errors):
    ax.scatter(1, vth, color='blue', edgecolor='black', zorder=3)

# === Formatting ===
ax.set_xticks([1])
ax.set_xticklabels(batch_names, rotation=45)
ax.set_ylabel("Activation Voltage (V)")
# ax.set_title("Activation Voltage Distribution - Batch 26")
ax.grid(True, which= 'both', linestyle=':', color='#CCCCCC')
ax.legend(loc='lower right')
save_path=rf"z:\Data\Plotting\Plots\withfanout_boxplot_activationvoltage.pdf"
plt.savefig(save_path, format='pdf', dpi=300)
plt.tight_layout()
plt.show()
#%% Box plot extension

# === Extract Vth and Vth_err values ===
vth_values = []
vth_errors = []
categories = []
screening_status = []

for entry in data:
    vth_values.append(entry['Vth'])
    vth_errors.append(entry['Vth_err'])
    categories.append(entry['SET'])
    screening_status.append(entry.get('needs_screening', False))

# === Convert to DataFrame for Easier Plotting ===
df = pd.DataFrame({
    'Vth': vth_values,
    'Vth_err': vth_errors,
    'Category': categories,
    'Screening': screening_status
})

# === Separate LSET and RSET Data ===
lset_data = df[df['Category'] == 'LSET']
rset_data = df[df['Category'] == 'RSET']

# === Create Box Plot ===
fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot(
    [lset_data['Vth'], rset_data['Vth']],
    labels=['LSET', 'RSET'],
    patch_artist=True,
    showfliers=False
)

# === Set Box Plot Colors ===
colors = ['lightblue', 'lightcoral']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# === Overlay Scatter Plot for Individual Points ===
for i, (data, color) in enumerate(zip([lset_data, rset_data], ['blue', 'red'])):
    x_positions = np.random.normal(i + 1, 0.05, size=len(data))
    for x, y, screening in zip(x_positions, data['Vth'], data['Screening']):
        if screening:
            ax.scatter(x, y, edgecolor='green', facecolor=color, s=100, linewidth=1.5)
        else:
            ax.scatter(x, y, color=color, s=50)

# === Add Mean and Median Values ===
mean_lset = np.mean(lset_data['Vth'])
median_lset = np.median(lset_data['Vth'])
mean_rset = np.mean(rset_data['Vth'])
median_rset = np.median(rset_data['Vth'])

ax.axhline(mean_lset, color='blue', linestyle='--', label=f'LSET Mean Vth = {mean_lset:.3f}V')
ax.axhline(median_lset, color='blue', linestyle='-', label=f'LSET Median Vth = {median_lset:.3f}V')
ax.axhline(mean_rset, color='red', linestyle='--', label=f'RSET Mean Vth = {mean_rset:.3f}V')
ax.axhline(median_rset, color='red', linestyle='-', label=f'RSET Median Vth = {median_rset:.3f}V')

# === Customize Plot ===
ax.set_title("Box Plot of Vth with Screening Highlights")
ax.set_xlabel("Category (LSET vs RSET)")
ax.set_ylabel("Turn-on Voltage (V)")
ax.grid(True, linestyle='--', color='grey', alpha=0.6)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()


#%% Sample accumulation plots in a quick way 
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A4_20240912.db'
# exp_id = 8
# set_side='lset'
# sample_name = extract_sample_name(db_path)
# meas = readout_database("data", db_path, exp_id - 1)
# lset_acc_y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
# lset_acc_y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
# lset_acc_x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
# lset_acc_x2 = meas["data"]["lockin_up_R"]["keithley_left_volt"]

# fig, ax = plt.subplots()
# ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage Current")
# ax.plot(lset_acc_x2, lset_acc_y2*1.e12, label="Transport Current")
# ax.set_xlabel("Voltage (V)")
# ax.set_ylabel("Current (pA)")
# # ax.set_title("Left SET Accumulation")
# ax.legend()
# ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
# ax.minorticks_on()
# ax.set_ylim([-100, 300])
# save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_accum_{set_side}_{exp_id}_{sample_name}.pdf"
# plt.savefig(save_path, format='pdf', dpi=300)
# plt.show()


# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch26 - qumada\GDSET_smile\S10_C2_20240730.db'
# exp_id = 4
# set_side='lset'
# sample_name = extract_sample_name(db_path)
# meas = readout_database("data",db_path, exp_id - 1)
# lset_acc_y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
# lset_acc_y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
# lset_acc_x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
# lset_acc_x2 = meas["data"]["lockin_up_R"]["keithley_left_volt"]

# fig, ax = plt.subplots()
# ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage Current")
# ax.plot(lset_acc_x2, lset_acc_y2*1.e12, label="Transport Current")
# ax.set_xlabel("Voltage (V)")
# ax.set_ylabel("Current (pA)")
# # ax.set_title("Left SET Accumulation")
# ax.legend()
# ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
# ax.minorticks_on()
# ax.set_ylim([-100, 300])
# save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_accum_{set_side}_{exp_id}_{sample_name}.pdf"
# plt.savefig(save_path, format='pdf', dpi=300)
# plt.show()

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch26 - qumada\GDSET_smile\S10_D4_20241022.db'
# exp_id = 3
# set_side='lset'
# sample_name = extract_sample_name(db_path)
# meas = readout_database("data",db_path, exp_id - 1)
# lset_acc_y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
# lset_acc_y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
# lset_acc_x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
# lset_acc_x2 = meas["data"]["lockin_up_R"]["keithley_left_volt"]

# fig, ax = plt.subplots()
# ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage Current")
# ax.plot(lset_acc_x2, lset_acc_y2*1.e12, label="Transport Current")
# ax.set_xlabel("Voltage (V)")
# ax.set_ylabel("Current (pA)")
# # ax.set_title("Left SET Accumulation")
# ax.legend()
# ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
# ax.minorticks_on()
# ax.set_ylim([-100, 300])
# save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_accum_{set_side}_{exp_id}_{sample_name}.pdf"
# plt.savefig(save_path, format='pdf', dpi=300)
# plt.show()


# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch26 - qumada\GDSET_smile\S12_B4_20241009.db'
# exp_id = 3
# set_side='lset'
# sample_name = extract_sample_name(db_path)
# meas = readout_database("data",db_path, exp_id - 1)
# lset_acc_y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
# lset_acc_y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
# lset_acc_x1 = meas["data"]["keithley_right_curr"]["keithley_right_volt"]
# lset_acc_x2 = meas["data"]["lockin_down_R"]["keithley_right_volt"]

# fig, ax = plt.subplots()
# ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage Current")
# ax.plot(lset_acc_x2, lset_acc_y2*1.e12, label="Transport Current")
# ax.set_xlabel("Voltage (V)")
# ax.set_ylabel("Current (pA)")
# # ax.set_title("Left SET Accumulation")
# ax.legend()
# ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
# ax.minorticks_on()
# ax.set_ylim([-100, 300])
# save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_accum_{set_side}_{exp_id}_{sample_name}.pdf"
# plt.savefig(save_path, format='pdf', dpi=300)
# plt.show()



#%% TIme Stability
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db'
# exp_id = 1
# set_side = 'lset'
# dev_name = 'S9_A3'
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability1_{set_side}_{exp_id}_{dev_name}.pdf")

db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch26 - qumada\GDSET_smile\S10_C2_20240730.db'
exp_id = 5
set_side='lset'
sample_name = extract_sample_name(db_path)
plot_time_stability_v2(db_path, 
                       exp_id,
                         set_side, 
                       save_path=rf"z:\Data\Plotting\Plots\withFanout\stability1_{set_side}_{exp_id}_{sample_name}.pdf")


# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db'
# exp_id = 24
# set_side = 'rset'
# dev_name = 'S8_A3'
# ylim = [-100,2000]
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                          ylim =ylim,
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db'
# exp_id = 12
# set_side = 'lset'
# dev_name = 'S12_B4'
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")
 
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db'
# exp_id = 8
# set_side = 'lset'
# dev_name = 'S12_A2'
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")
 
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db'
# exp_id = 27
# set_side = 'rset'
# dev_name = 'S11_B2'
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")
 
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db'
# exp_id = 10
# set_side = 'lset'
# dev_name = 'S10_D4'
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")
 

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db'
# exp_id = 10
# set_side = 'lset'
# dev_name = 'S10_D4'
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")


db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_C1_20240723.db'
exp_id = 7
set_side = 'rset'
dev_name = 'S8_C1'
plot_time_stability_v2(db_path, 
                       exp_id,
                         set_side, 
                       save_path=rf"z:\Data\Plotting\Plots\withFanout\stability_{set_side}_{exp_id}_{dev_name}.pdf")

#%% Combined Plot Time Stability from a list of db files
# Define the list of (db_path, exp_id, set_side)
db_table_list = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db', 
     1, 'lset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch26 - qumada\GDSET_smile\S10_C2_20240730.db',
      5,
      'lset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db',
      24, 'rset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db',
      12, 'lset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db',
      8, 'lset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db',
      27, 'rset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     10, 'lset'),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     14, 'rset'),
]

# Labels for devices
device_labels = ["Dev 12", "Dev 13", "Dev 14", "Dev 15", "Dev 16", "Dev 20", "Dev 22"]

# Output path
save_path = r"z:\Data\Plotting\Plots\multi_stability_plot_withFanout3.pdf"

# Call the function
plot_multi_time_stability_qcodes(db_table_list,
                                  device_labels=device_labels, 
                                  save_path=save_path)

#%% Sample time_stablity plots
db_table_pairs_fanout_qcodes_time = [
    # (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_C2_20240723.db', 
    #  2, 
    #  'RSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db', 
    #  2, 
    #  'LSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_B2_20241002.db', 
    #  'results-1-5', 
    #  'LSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     5, 
     'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db', 
     9, 
     'LSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
    #  3, 
    #  'LSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db', 
    #  33, 
    #  'RSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B3_20240912.db', 
    #  4, 
    #  'LSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B3_20240912.db', 
    #  12, 
    #  'RSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D2_20240823.db', 
    #  2, 
    #  'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     11, 
     'LSET'),
    #  (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_C1_20240723.db', 
    #  7, 
    #  'RSET'),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     15, 
     'RSET'),
     ]

# db_sample_accum= [ (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
#      11, 
#      'LSET')]
# plot_time_stability(db_sample_accum[0][0], db_sample_accum[0][1], db_sample_accum[0][2])
# plot_accumulation_v2(db_sample_accum[0][0], db_sample_accum[0][1], db_sample_accum[0][2])
plot_multi_time_stability_qcodes(db_table_pairs_fanout_qcodes_time, 
                                 device_labels=['Dev 3', 'Dev 4', 'Dev 10', 'Dev 12'],
                                 save_path = rf'z:\Data\Plotting\Plots\Stability\combined2_time.pdf')

#%%  1D Pinch Off measurements Plunger
# plunger_withfanout_sweeps = [
#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
#      6, 
#      "dac2_Slot2_Chan1_volt", 
#      "right",  
#      save_folder),

#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db'	, 
#      19, 
#      "dac4_Slot4_Chan2_volt", 
#      "left",  
#      save_folder),

#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db'	, 
#      22, 
#      "dac2_Slot2_Chan1_volt", 
#      "right",  
#      save_folder)]

# List of QCoDeS databases for plunger sweeps (with user-specified DAC keys)
plunger_withfanout_sweeps = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     6, 
     "dac2_Slot2_Chan1_volt", 
     'rset', 
     "Plunger"),
    
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
     19, 
     "dac4_Slot4_Chan2_volt", 
     'lset', 
     "Plunger"),
]

# JSON file with pre-saved plunger sweeps
json_file = r'z:\Data\Plotting\Plots\Plunger\json_plunger\plunger_data_withfanout.json'

# Save path
save_to_file = r'z:\Data\Plotting\Plots\Plunger\withfanout_Combined_Plunger_Sweeps_User_DAC_withleak3.pdf'

# Call the function
# plot_multiple_plunger_sweeps_v2(plunger_withfanout_sweeps, 
#                                 json_file=json_file, 
#                                 save_to_file=save_to_file)
device_labels = ['Dev 13', 'Dev 15', 'Dev 11', 'Dev 16']
ylim_transport = [-10,200]
ylim_leakage = [-100,100]
plot_multiple_plunger_sweeps_with_leakage_v4(plunger_withfanout_sweeps, 
                                json_file=json_file, 
                                save_to_file=save_to_file,
                                device_labels=device_labels,
                                ylim_transport=ylim_transport,
                                ylim_leakage=None)

#%%  Plunger second attempt

save_folder = "z:\Data\Plotting\Plots\Plunger"

plunger_withfanout_sweeps = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     9, 
     "dac2_Slot2_Chan1_volt", 
     "right",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     6, 
     "dac2_Slot2_Chan1_volt", 
     "right",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db'	, 
     14, 
     "dac4_Slot4_Chan3_volt", 
     "left",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db'	, 
     19, 
     "dac4_Slot4_Chan2_volt", 
     "left",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db'	, 
     32, 
     "dac2_Slot2_Chan0_volt", 
     "right",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db'	, 
     22, 
     "dac2_Slot2_Chan1_volt", 
     "right",  
     save_folder)]

plunger_withfanout_sweeps_2 = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db',  
     10,  
     "right",
       save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db',  
     7, 
       "right", 
       save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db', 
     15, 
     "left",  
     save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
     20, 
     "left",  
     save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db', 
     36, 
     "right", 
     save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db',
      17, 
      "right", 
      save_folder),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db',
      23, 
      "right", 
      save_folder),
]

plunger_withfanout_sweeps_combined = [
    # (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db',  
    #  10,  
    #  "right",
    #    save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db',  
     7, 
       "right", 
       save_folder),
    # (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db', 
    #  15, 
    #  "left",  
    #  save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
     20, 
     "left",  
     save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db', 
     36, 
     "right", 
     save_folder),
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db',
      17, 
      "right", 
      save_folder),
     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db',
      23, 
      "right", 
      save_folder),
]

for db_path, exp_id, side, save_folder in plunger_withfanout_sweeps_2:
    # meas = readout_database("data", db_path, exp_id - 1)
    plot_gate_with_sweep_direction(
        db_path=db_path,
        exp_id=exp_id,
        set_side=side,
        save_folder=save_folder,
        ylim=(-50, 150),
        xlabel="Plunger"
    )
sweeps = [(db, eid, side) for db, eid, side, _ in plunger_withfanout_sweeps_2]

plot_all_sweeps_in_single_figure(
    sweep_list=sweeps,
    save_path=r"z:\Data\Plotting\Plots\combined_plunger_fanout2.pdf",
    xlim=(0, 1.2),
    trans_ylim=(0, 700),
    leak_ylim=(-50, 50),
    xlabel="Plunger"
)


#%% Sample pinch off plots

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db'
# exp_id =13
# set_side ='lset'
# sample_name = extract_sample_name(db_path)
# save_to_file = rf'z:\Data\Plotting\Plots\Sample Plots\sample_pinchoffs_{exp_id}_{sample_name}.pdf'

# plot_gate_with_sweep_direction_v2(db_path=db_path,
#                                   exp_id=exp_id,
#                                   set_side=set_side,
#                                   save_path=save_to_file,
#                                   ylim_trans=[-10,250],
#                                   ylim_leakage=[-50,100],
#                                   xlabel='Plunger')

db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db'
exp_id =9
set_side ='rset'
sample_name = extract_sample_name(db_path)
save_to_file = rf'z:\Data\Plotting\Plots\Sample Plots\sample_pinchoffs_{exp_id}_{sample_name}.pdf'

plot_gate_with_sweep_direction_v2(db_path=db_path,
                                  exp_id=exp_id,
                                  set_side=set_side,
                                  save_path=save_to_file,
                                  ylim_trans=[-10,250],
                                  ylim_leakage=[-50,100],
                                  xlabel='Plunger')

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db'
# exp_id =19
# set_side ='lset'
# sample_name = extract_sample_name(db_path)
# save_to_file = rf'z:\Data\Plotting\Plots\Sample Plots\sample_pinchoffs_{exp_id}_{sample_name}.pdf'

# plot_gate_with_sweep_direction_v2(db_path=db_path,
#                                   exp_id=exp_id,
#                                   set_side=set_side,
#                                   save_path=save_to_file,
#                                   ylim_trans=[-10,500],
#                                   ylim_leakage=[-50,100],
#                                   xlabel='Plunger')

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db'
# exp_id =20
# set_side ='lset'
# sample_name = extract_sample_name(db_path)
# save_to_file = rf'z:\Data\Plotting\Plots\Sample Plots\sample_pinchoffs_{exp_id}_{sample_name}.pdf'

# plot_gate_with_sweep_direction_v2(db_path=db_path,
#                                   exp_id=exp_id,
#                                   set_side=set_side,
#                                   save_path=save_to_file,
#                                   ylim_trans=[-10,500],
#                                 #   ylim_leakage=[],
#                                   xlabel='Top Barrier')


# Plunger Pinch Off S8_A3
rset_acc = True
exp_id = 10


meas = readout_database("data",
                        r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db", 
                        exp_id - 1)
title = 'SG sweep to 0.25'
rset_acc_y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
rset_acc_y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
rset_sg_x1 = meas["data"]["keithley_right_curr"]["dac2_Slot2_Chan1_volt"]
rset_sg_x2 = meas["data"]["lockin_down_R"]["dac2_Slot2_Chan1_volt"]



save_path = rf'z:\Data\Plotting\Plots\Sample Plots\S8_A3_plunger_PO_{exp_id}3.pdf'
# Right SET Accumulation
if rset_acc:
    fig, ax = plt.subplots()
    ax.plot(rset_sg_x1, rset_acc_y1*1.e12, label="Leakage current")
    ax.plot(rset_sg_x2, rset_acc_y2*1.e12, label="Transport current")
    ax.set_xlabel(" Plunger Voltage (V)")
    ax.set_ylabel("Current (pA)")
    # ax.set_title(title)
    ax.legend()
    plt.ylim([-50, 2000])
    plt.tight_layout()
    # plt.grid()
    ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax.minorticks_on()
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()

# S12_B4 sample PinchOff
db_path = r'c:\Users\Suresh\Documents\Three Layer SET\Data\Three layer SET\09.11.2024\Batch26 - combined\GDSET_smile\S12_B4_20240927.db'
sample_name = extract_sample_name(db_path)

# plot_plunger_with_sweep_direction(
#     db_path=db_path,
#     exp_id=20,
#     gate_name="dac4_Slot4_Chan2_volt",
#     set_side='left',
#     save_path=rf'z:\Data\Plotting\Plots\Sample Plots\{sample_name}_plunger_PO_{exp_id}.pdf',
#     ylim=(-50, 500),
#     xlabel="Plunger"
# )

# plot_plunger_with_sweep_direction(
#     db_path=db_path,
#     exp_id = 21,
#     gate_name="dac4_Slot4_Chan0_volt",
#     set_side='left',
#     save_path=rf'z:\Data\Plotting\Plots\Sample Plots\{sample_name}_TB_PO_f{exp_id}_.pdf',
#     ylim=(-50, 2000),
#     xlabel = 'Top Barrier')


plot_plunger_with_sweep_direction(
    db_path=db_path,
    exp_id=19,
    gate_name="dac5_Slot0_Chan0_volt",
    set_side='left',
    save_path=rf'z:\Data\Plotting\Plots\Sample Plots\{sample_name}_BB_PO_f{exp_id}_2.pdf',
    ylim=(-50, 2000),
    xlabel = 'Bottom Barrier'
)

#%% 1D Top Barrier

topbarrier_withfanout_sweeps = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_C2_20240723.db', 
     3, 
     "dac2_Slot0_Chan3_volt", 
     "rset",  
     'Top Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db', 
     3, 
     "dac4_Slot4_Chan0_volt", 
     "lset", 
     'Top Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db', 
     11, 
     "dac4_Slot4_Chan0_volt", 
     "lset",  
     'Top Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240823.db', 
     3, 
     "dac2_Slot1_Chan2_volt", 
     "rset",  
     'Top Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D2_20240823.db', 
     3, 
     "dac2_Slot1_Chan2_volt", 
     "rset", 
     'Top Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_C1_20240723.db'	, 
     8, 
     "dac2_Slot0_Chan3_volt", 
     "rset",  
     'Top Barrier')]

json_file = r'z:\Data\Plotting\Plots\Plunger\S13_D4_rtb_pinch_off_data.json'

# Save path
save_to_file = r'z:\Data\Plotting\Plots\Plunger\withfanout_Combined_TopBarrier_Sweeps_withleakscaled.pdf'

device_labels = ['Dev 11', 'Dev 12', 'Dev 15', 'Dev 16',  'Dev 19', 'Dev 21', 'Dev 24']

ylim_transport = [-10,500]
ylim_leakage = [-0.2,10]

plot_multiple_topbarrier_sweeps_with_leakage_v4(topbarrier_withfanout_sweeps, 
                                json_file=json_file, 
                                save_to_file=save_to_file,
                                device_labels=device_labels,
                                ylim_transport=ylim_transport,
                                ylim_leakage=ylim_leakage,
                                xlim_trans=None,
                                xlim_leak=None)


#%% 1D Bottom Barrier

bottom_gate_withfanout_sweeps = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_C2_20240723.db', 
     3, 
     "dac2_Slot0_Chan3_volt", 
     "rset",  
     'Bottom Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_A3_20240816.db', 
     3, 
     "dac4_Slot4_Chan0_volt", 
     "lset",  
     'Bottom Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     8, 
     "dac2_Slot1_Chan3_volt", 
     "rset",  
     'Bottom Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_B4_20241009.db'	, 
     13, 
     "dac5_Slot0_Chan1_volt", 
     "lset",  
     'Bottom Barrier'),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db'	, 
     12, 
     "dac5_Slot0_Chan0_volt", 
     "lset",  
     'Bottom Barrier'),]

save_to_file = r'z:\Data\Plotting\Plots\Plunger\withfanout_BottomBarrier_Sweeps_withleakscaled.pdf'

device_labels = ['Dev 11', 'Dev 12', 'Dev 13', 'Dev 14',  'Dev 15']

ylim_transport = [-10,230]
# ylim_leakage = [-100,100]

plot_multiple_bottombarrier_sweeps_with_leakage_v4(bottom_gate_withfanout_sweeps, 
                                json_file=json_file, 
                                save_to_file=save_to_file,
                                device_labels=device_labels,
                                ylim_transport=ylim_transport,
                                ylim_leakage=ylim_leakage)

#%% SG sweep

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db'
# exp_id =31
# set_side ='rset'
# sample_name = extract_sample_name(db_path)
# save_to_file = rf'z:\Data\Plotting\Plots\Sample Plots\sample_pinchoffs_{exp_id}_{sample_name}.pdf'

# plot_gate_with_sweep_direction_v2(db_path=db_path,
#                                   exp_id=exp_id,
#                                   set_side=set_side,
#                                   save_path=save_to_file,
#                                   ylim_trans=[-10,200],
#                                   ylim_leakage=[-50,100],
#                                   xlabel='SG Voltage')

db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db'
exp_id =20
set_side ='rset'
sample_name = extract_sample_name(db_path)
save_to_file = rf'z:\Data\Plotting\Plots\Sample Plots\sample_pinchoffs_{exp_id}_{sample_name}.pdf'

plot_gate_with_sweep_direction_v2(db_path=db_path,
                                  exp_id=exp_id,
                                  set_side=set_side,
                                  save_path=save_to_file,
                                  ylim_trans=[-10,200],
                                  ylim_leakage=[-50,100],
                                  xlabel='SG Voltage')


#%% 2D sweep of plunger-SG 


# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db'
# exp_id = 23
# set_side = 'rset'
# plot_2d_barrier_barrier(db_path, 
#                         exp_id, 
#                         set_side,
#                         BB_gate='dac2_Slot3_Chan1_volt',
#                         TB_gate='dac2_Slot2_Chan1_volt',
#                         # xlim = [0.45,None],
#                         # ylim=[0.95, None],
#                         xlabel= 'SG',
#                         ylabel='Plunger',
#                         save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_PSGsweep_{set_side}_{exp_id}.pdf")


db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db'
exp_id = 17
set_side = 'rset'
sample_name = extract_sample_name(db_path)
plot_2d_barrier_barrier(db_path, 
                        exp_id, 
                        set_side,
                        BB_gate='dac2_Slot3_Chan1_volt',
                        TB_gate='dac2_Slot2_Chan1_volt',
                        xlim = [-0.45,None],
                        # ylim=[0.95, None],
                        xlabel= 'SG',
                        ylabel='Plunger',
                        save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_{sample_name}_PSGsweep_{set_side}_{exp_id}.pdf")



#%% Plotting 1D slice from 2D
# ]


# db_path = r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S9_C2_20240723.db"
# exp_id = 25
# set_side = 'rset'
# fixed_voltage_x = 0.5
# fixed_voltage_y = 0.81
# TB_gate='dac2_Slot0_Chan2_volt'
# BB_gate='dac2_Slot0_Chan3_volt'

# save_json_path = r"z:\Data\Plotting\Plots\Plunger\json_plunger\plunger_data_withfanout.json"

# plot_2d_barrier_barrier(db_path,exp_id,set_side, 
#                         TB_gate, # Plunger
#                         BB_gate,
#                         xlabel='Bottom',
#                         ylabel='Plunger')

# # plot_2d_barrier_barrier_v3(db_path,exp_id,set_side)
# plot_1d_slices_from_2d_v2(db_path,
#                        exp_id,
#                        set_side, 
#                        TB_gate,
#                        BB_gate,
#                        fixed_voltage_x, 
#                        fixed_voltage_y,
#                        save_json_path=save_json_path)

db_path = r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240823.db"
exp_id = 6
set_side = 'rset'
fixed_voltage_x = 0.66
fixed_voltage_y = 0.65
TB_gate='dac2_Slot2_Chan0_volt'
BB_gate='dac2_Slot1_Chan2_volt'

save_json_path = r"z:\Data\Plotting\Plots\Plunger\json_plunger\plunger_data_withfanout.json"

plot_2d_barrier_barrier(db_path,exp_id,
                        set_side, 
                        TB_gate, #plunger
                        BB_gate,
                        xlabel='Bottom',
                        ylabel='Plunger')

# plot_2d_barrier_barrier_v3(db_path,exp_id,set_side)
plot_1d_slices_from_2d_v2(db_path,
                       exp_id,
                       set_side, 
                       TB_gate,
                       BB_gate,
                       fixed_voltage_x, 
                       fixed_voltage_y,
                       save_json_path=save_json_path)


db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240730.db'

exp_id = 9
set_side = 'rset'

# json_path = r'z:\Data\Plotting\Plots\Plunger\plunger_data_withfanout.json'
# plot_1d_slice_from_json(json_path)


# db_path = r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S12_A2_20240927.db"
# exp_id = 26
# set_side = 'lset'
# fixed_voltage_x = 0.6
# fixed_voltage_y = 0.7

# plot_2d_barrier_barrier_v3(db_path, exp_id, set_side)
# plot_1d_slices_from_2d(db_path,exp_id,set_side, fixed_voltage_x, fixed_voltage_y)
















#%%   Screening Gate

rset_acc = True
exp_id = 22
meas = readout_database("data",
                        r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db", 
                        exp_id - 1)
title = 'SG sweep to 0.25'
rset_acc_y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
rset_acc_y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
rset_sg_x1 = meas["data"]["keithley_right_curr"]["dac2_Slot3_Chan1_volt"]
rset_sg_x2 = meas["data"]["lockin_down_R"]["dac2_Slot3_Chan1_volt"]



save_path = rf'z:\Data\Plotting\Plots\Sample Plots\S8_A3_sg_to_0.2_{exp_id}.pdf'
# Right SET Accumulation
if rset_acc:
    fig, ax = plt.subplots()
    ax.plot(rset_sg_x1, rset_acc_y1*1.e12, label="Leakage current")
    ax.plot(rset_sg_x2, rset_acc_y2*1.e12, label="Transport current")
    ax.set_xlabel(" SG Voltage (V)")
    ax.set_ylabel("Current (pA)")
    # ax.set_title(title)
    ax.legend()
    plt.ylim([-50, 1500])
    plt.tight_layout()
    # plt.grid()
    ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax.minorticks_on()
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()

#%% Screening Gate combined

save_folder = "z:\Data\Plotting\Plots\Screening_gate"
sg_sweeps = [
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S11_B2_20240827_Qumada_Test.db', 
     32, 
     "dac2_Slot3_Chan0_volt", 
     "right", "SG sweep to 0.25", 
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     19, 
     "dac2_Slot3_Chan1_volt", 
     "right",  
     "SG sweep to 0.4",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     20, 
     "dac2_Slot3_Chan1_volt", 
     "right",  
     "SG sweep to 0.4",  
     save_folder),

    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     21, 
     "dac2_Slot3_Chan1_volt", 
     "right",  
     "SG sweep to 0.4",  
     save_folder),
     
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db', 
     22, "dac2_Slot3_Chan1_volt", 
     "right", 
     "SG sweep to 0.25", 
     save_folder),
     
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     19,
     "dac2_Slot3_Chan1_volt", 
     "right", 
     "SG sweep to 0.25", 
     save_folder),
     
    (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S10_D4_20241027.db', 
     21,
     "dac2_Slot3_Chan1_volt", 
     "right", 
     "SG sweep to 0.25", 
     save_folder),

    ]


for db_path, exp_id, gate, side, title, save_folder in sg_sweeps:
    plot_gate_with_sweep_direction(
        db_path=db_path,
        exp_id=exp_id,
        gate_name=gate,
        set_side=side,
        title=title,
        save_folder=save_folder,
        ylim=(-50, 1500),
        xlabel="SG"
    )









#%% Plunger sample
rset_acc = True
exp_id = 9
# gate_name = "dac2_Slot1_Chan3_volt"

meas = readout_database("data",
                        r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db',
                        exp_id - 1)

for key in meas["data"]["lockin_down_R"]:
    if "volt" in key:
        gate_name = key
        break
# title = 'SG sweep to 0.25'
# rset_acc_y1 = meas["data"]["keithley_right_curr"]["keithley_right_curr"]
rset_acc_y2 = meas["data"]["lockin_down_R"]["lockin_down_R"]
# rset_sg_x1 = meas["data"]["keithley_right_curr"][gate_name]
rset_sg_x2 = meas["data"]["lockin_down_R"][gate_name]




save_path = rf'z:\Data\Plotting\Plots\Plunger\S9_C3_Plunger_PO_{exp_id}.pdf'
# Right SET Accumulation
if rset_acc:
    fig, ax = plt.subplots()
    # ax.plot(rset_sg_x1, rset_acc_y1*1.e12, label="Leakage current")
    ax.plot(rset_sg_x2, rset_acc_y2*1.e12, color= 'blue',label="Transport current")
    ax.set_xlabel(" Plunger Voltage (V)")
    ax.set_ylabel("Current (pA)")
    # ax.set_title(title)
    ax.legend()
    plt.ylim([-50, 200])
    plt.tight_layout()
    # plt.grid()
    ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
    ax.minorticks_on()
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()
#%% Plunger pinchoff color code

# plot_plunger_with_sweep_direction(
#     db_path=r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db",
#     exp_id=10,
#     gate_name="dac2_Slot2_Chan1_volt",
#     set_side='right',
#     save_path=r'z:\Data\Plotting\Plots\Sample Plots\S8_A3_plunger_PO_10_2.pdf',
#     trans_ylim = ([-50, 2000]),
#     xlim=[0,2],
#     leak_ylim = ([-50, 50]),
#     xlabel='Plunger'
# )


#%% S8_B4 sample Pinchoff


# exp_id = 10
# plot_plunger_with_sweep_direction(
#     db_path=r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db",
#     exp_id=10,
#     gate_name="dac2_Slot2_Chan1_volt",
#     set_side='right',
#     save_path=r'z:\Data\Plotting\Plots\Sample Plots\S8_A3_plunger_PO_10.pdf',
#     ylim=(-50, 2000),
#     xlabel="Plunger"
# )
# exp_id = 21
# plot_plunger_with_sweep_direction(
#     db_path = r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db",
#     exp_id = exp_id,
#     gate_name="dac2_Slot3_Chan1_volt",
#     set_side='right',
#     save_path=rf'z:\Data\Plotting\Plots\Sample Plots\S8_A3_SG_PO_f{exp_id}2.pdf',
#     ylim=(-50, 2000),
#     xlabel = 'SG')


# plot_plunger_with_sweep_direction(
#     db_path=r"Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_combined\Batch26_QuMADA\GDSET_smile\S8_A3_20240927.db",
#     exp_id=9,
#     gate_name="dac2_Slot1_Chan3_volt",
#     set_side='right',
#     save_path=rf'z:\Data\Plotting\Plots\Sample Plots\S8_A3_BB_PO_f{exp_id}_2.pdf',
#     ylim=(-50, 2000),
#     xlabel = 'Bottom Barrier'
# )




#%%  S12_B4 BB sweep

db_path = r'c:\Users\Suresh\Documents\Three Layer SET\Data\Three layer SET\09.11.2024\Batch26 - combined\GDSET_smile\S12_B4_20240927.db'
sample_name = extract_sample_name(db_path)
# TB_gate = 'dac4_Slot4_Chan0_volt'
# BB_gate = 'dac5_Slot0_Chan0_volt'
exp_id = 36

plot_2d_barrier_barrier(db_path,
                        exp_id=exp_id,
                        set_side='lset',
                        TB_gate = 'dac4_Slot4_Chan0_volt',
                        BB_gate = 'dac5_Slot0_Chan0_volt',
                        save_path = rf'z:\Data\Plotting\Plots\Sample Plots\{sample_name}_2D_BB_PO_{exp_id}.pdf'
                        )

#%% ############################ with mesa ###################################

# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\QuBus_with_mesa\S14_B4_20241004.db'
# exp_id = 14
# set_side = 'lset'
# # dev_name = 'S9_A3'
# sample_name =extract_sample_name(db_path)
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                         #  ylim=[-10,350],
#                        save_path=rf"z:\Data\Plotting\Plots\with_mesa\{sample_name}_stability_{set_side}_{exp_id}.pdf")

# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\QuBus_with_mesa\S14_B4_20250116.db'
# exp_id = 9
# set_side = 'lset'
# # dev_name = 'S9_A3'
# sample_name =extract_sample_name(db_path)
# plot_time_stability_v2(db_path, 
#                        exp_id,
#                          set_side, 
#                         #  ylim=[-10,350],
#                        save_path=rf"z:\Data\Plotting\Plots\with_mesa\{sample_name}_stability_{set_side}_{exp_id}_{db_path}.pdf")


db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\QuBus_with_mesa\S14_B4_20250117.db'
exp_id = 6
set_side = 'lset'
# dev_name = 'S9_A3'
sample_name =extract_sample_name(db_path)
plot_time_stability_v2(db_path, 
                       exp_id,
                         set_side, 
                        #  ylim=[-10,350],
                       save_path=rf"z:\Data\Plotting\Plots\with_mesa\{sample_name}_stability_{set_side}_{exp_id}_0117.pdf")



#%% ################## Independent Fanout FET #############

# z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250321.db 
# This is also not good


# r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A2_QuBus_IndFanoutFET_20250324.db' 
# Not good full noisy even though a small turning on at 1V





# Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250321.db


# Accumulation LSET



db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250324.db'
exp_id = 10
set_side='lset'
sample_name = extract_sample_name(db_path)
meas = readout_database("data", db_path, exp_id - 1)
lset_acc_y2 = meas["data"]["mfli1_current"]["mfli1_current"]
# lset_acc_x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
lset_acc_x2 = meas["data"]["mfli1_current"]["keithley_left_volt"]

db_path2 =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
exp_id2 = 23
set_side2='rset'
sample_name = extract_sample_name(db_path2)
meas2 = readout_database("data", db_path2, exp_id2 - 1)
# lset_acc_y11 = meas1["data"]["keithley_right_curr"]["keithley_right_curr"]
lset_acc_y22 = meas2["data"]["mfli2_current"]["mfli2_current"]
# lset_acc_x11 = meas1["data"]["keithley_right_curr"]["keithley_right_volt"]
lset_acc_x22 = meas2["data"]["mfli2_current"]["dac2_Slot4_Chan2_volt"]

db_path1 =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A2_QuBus_IndFanoutFET_20250403.db'
exp_id1 = 2
set_side1='rset'
sample_name = extract_sample_name(db_path1)
meas1 = readout_database("data", db_path1, exp_id1 - 1)
# lset_acc_y11 = meas1["data"]["keithley_right_curr"]["keithley_right_curr"]
lset_acc_y21 = meas1["data"]["mfli2_current"]["mfli2_current"]
# lset_acc_x11 = meas1["data"]["keithley_right_curr"]["keithley_right_volt"]
lset_acc_x21 = meas1["data"]["mfli2_current"]["dac2_Slot4_Chan3_volt"]


fig, ax = plt.subplots()
# ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage Current")
ax.plot(lset_acc_x2, lset_acc_y2*1.e12,  label="Dev 27")
ax.plot(lset_acc_x22, lset_acc_y22*1.e12, label="Dev 28")
ax.plot(lset_acc_x21, lset_acc_y21*1.e12, label="Dev 29")

ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Transport Current (pA)")
# ax.set_title("Left SET Accumulation")
ax.legend()
ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
ax.minorticks_on()
ax.set_ylim([-10, 300])
ax.set_xlim([0, 1.2])
save_path=rf"z:\Data\Plotting\Plots\Sample Plots\indfanout_combined_{set_side}_{exp_id}_{sample_name}.pdf"
plt.savefig(save_path, format='pdf', dpi=300)
plt.show()

# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A2_QuBus_IndFanoutFET_20250324.db'
# exp_id = 8
# set_side='lset'
# sample_name = extract_sample_name(db_path)
# meas = readout_database("data", db_path, exp_id - 1)
# lset_acc_y1 = meas["data"]["keithley_left_curr"]["keithley_left_curr"]
# lset_acc_y2 = meas["data"]["lockin_up_R"]["lockin_up_R"]
# lset_acc_x1 = meas["data"]["keithley_left_curr"]["keithley_left_volt"]
# lset_acc_x2 = meas["data"]["lockin_up_R"]["keithley_left_volt"]

# fig, ax = plt.subplots()
# ax.plot(lset_acc_x1, lset_acc_y1*1.e12, label="Leakage Current")
# ax.plot(lset_acc_x2, lset_acc_y2*1.e12, label="Transport Current")
# ax.set_xlabel("Voltage (V)")
# ax.set_ylabel("Current (pA)")
# # ax.set_title("Left SET Accumulation")
# ax.legend()
# ax.grid(True, which='both', linestyle=':', color='#CCCCCC')
# ax.minorticks_on()
# ax.set_ylim([-100, 300])
# save_path=rf"z:\Data\Plotting\Plots\Sample Plots\withfanout_accum_{set_side}_{exp_id}_{sample_name}.pdf"
# plt.savefig(save_path, format='pdf', dpi=300)
# plt.show()

#%% Accumulation S15_A1 first sample

# db_table_pairs_ind_fanout_qcodes = [
#     (r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250324.db', 
#      10, 
#      'LSET'),
#      (r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db', 
#      40, 
#      'RSET'),
#      ]

# save_to_file_ind_fanout = rf'z:\Data\Plotting\Plots\Independent FET\combined_accumulation_ind_fanoutFET.pdf'
# # json_pfile_paths_fanout = [r'z:\Data\Plotting\S13_D4_pfile_data.json']

# accu_yrange = [0,200]
# accu_xrange = [0,1]
# leak_xrange = [0,2]
# leak_yrange = [-100,50]


# plot_multiple_accumulation_qcodes_with_mfli(db_table_pairs_ind_fanout_qcodes,
#                                 #   json_pfile_paths_fanout,
#                                   save_to_file_ind_fanout,
#                                       accu_xrange=[0, 2],
#                                     accu_yrange=[0, 200],
#                                     leak_xrange=[0, 2],
#                                     leak_yrange=[-100, 50],
                                  
# stability

# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250324.db'
# exp_id = 11
# set_side = 'lset'
# plot_time_stability_v2(db_path, exp_id, set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\stability_{set_side}_{exp_id}.pdf")


# Accumulation RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 40
# set_side='rset'
# plot_accumulation_v3(db_path, exp_id, set_side=set_side, 
#                      save_path=rf"z:\Data\Plotting\Plots\Independent FET\accum_{set_side}_{exp_id}.pdf")


# stability RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 24
# set_side = 'rset'
# plot_time_stability_v2(db_path, exp_id, set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\stability_{set_side}_{exp_id}.pdf")




# stability RSET
db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250410.db'
exp_id = 12
set_side = 'rset'
plot_time_stability_v2(db_path, exp_id, set_side, 
                       save_path=rf"z:\Data\Plotting\Plots\Independent FET\long_flowrate_stability_{set_side}_{exp_id}.pdf")





# RBB sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 42
# set_side = 'rset'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\gatesweep_{set_side}_{exp_id}.pdf",
#                        xlabel='RBB')




# # RP sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 44
# set_side = 'rset'
# xlabel = 'RP'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)



# RTB sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 44
# set_side = 'rset'
# xlabel = 'RTB'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)


# z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_26_full\TLSET_without_fanout\S9_C3_20240729.db

# RTA sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 46
# set_side = 'rset'
# xlabel = 'RTA'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)




# RBA sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 47
# set_side = 'rset'
# xlabel = 'RBA'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)




# RS -ve sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 55
# xlabel = 'RS'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)

# RS -ve sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 53
# xlabel = 'RS'
# set_side ='rset'
# sample_name = extract_sample_name(db_path)
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{sample_name}_{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel,
#                        ylim_trans= [-10,300])

# RS +ve sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 54 
# # exp_id = 62
# xlabel = 'RS'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)

# Ind FET -ve sweep RSET
# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 59
# xlabel = 'iFET'
# sample_name = extract_sample_name(db_path)
# plot_gate_with_sweep_direction_v2(db_path,
#                                    exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{sample_name}_{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel,
#                        ylim_trans=[-10,300])

# # Ind FET +ve sweep RSET
# db_path =r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db'
# exp_id = 58
# xlabel = 'Ind FET'
# plot_gate_with_sweep_direction_v2(db_path, exp_id, 
#                                   set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel)




#%% S15_A2 Second sample

# Ind FET -ve sweep RSET
# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A2_QuBus_IndFanoutFET_20250403.db'
# exp_id = 20
# xlabel = 'iFET'
# set_side = 'rset'
# sample_name = extract_sample_name(db_path)
# plot_gate_with_sweep_direction_v2(db_path=db_path, exp_id=exp_id, 
#                                   set_side = set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{sample_name}_{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel,
#                        ylim_trans=[0,300])

# iFET sweep
# db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A2_QuBus_IndFanoutFET_20250403.db'
# exp_id = 21
# xlabel = 'iFET'
# set_side = 'rset'
# sample_name = extract_sample_name(db_path)
# plot_gate_with_sweep_direction_v2(db_path=db_path, exp_id=exp_id, 
#                                   set_side = set_side, 
#                        save_path=rf"z:\Data\Plotting\Plots\Independent FET\{sample_name}_{xlabel}_sweep_{set_side}_{exp_id}.pdf",
#                        xlabel=xlabel,
#                        ylim_trans=[0,300])
# # SG sweep

db_path =r'z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A2_QuBus_IndFanoutFET_20250403.db'
exp_id = 31
xlabel = 'SG'
set_side = 'rset'
sample_name = extract_sample_name(db_path)
plot_gate_with_sweep_direction_v2(db_path=db_path, exp_id=exp_id, 
                                  set_side = set_side, 
                       save_path=rf"z:\Data\Plotting\Plots\Independent FET\{sample_name}_{xlabel}_sweep_{set_side}_{exp_id}.pdf",
                       xlabel=xlabel,
                       ylim_trans=[0,300])


# combined_sweeps = [
#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db', 43, 'rset', 'RBB'),
#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db', 44, 'rset', 'RP'),
#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db', 45, 'rset', 'RTB'),
#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db', 46, 'rset', 'RTA'),
#     (r'Z:\Data\Three layer SET\QCoDeS+QuMADA\Batch_29\S15_A1_QuBus_IndFanoutFET_20250403.db', 47, 'rset', 'RBA'),
# ]

# plot_combined_gate_sweeps_v2(combined_sweeps,
#                           save_path=r"z:\Data\Plotting\Plots\Independent FET\all_gatesweep_transport_combined.pdf")


#%% #################### FLow rate plot using csv #######################

# === Load and clean the data ===
file_path = r"z:\Data\Three layer SET\Flowrate_Hermit\Flow-data-2025-05-05 16_55_42.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Clean and extract columns
time_col = df.columns[0]
q0_col = df.columns[1]

# Remove 'L/min' and convert Q0 to float
df[q0_col] = df[q0_col].astype(str).str.replace('L/min', '', regex=False).str.strip()
df[q0_col] = pd.to_numeric(df[q0_col], errors='coerce')

# Filter rows where Q0 is valid
df_valid = df.dropna(subset=[q0_col]).copy()

if df_valid.empty:
    print(" Still no valid Q0 values found after cleaning.")
else:
    # Generate synthetic time axis spaced 10 seconds apart
    start_time = datetime(2025, 4, 10, 12, 26, 0)
    df_valid["Time"] = [start_time + timedelta(seconds=10 * i) for i in range(len(df_valid))]

    # Plot
    
    plt.figure(figsize=(5, 4))
    plt.axhline(y=5, linestyle = '--', label='Set Flow Rate')
    plt.plot(np.arange(0, 290, 10), df_valid[q0_col], color='green', label='Actual Flow Rate', marker='o')

    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (L/min)")
    # plt.title("Flow Rate vs Time (Every 10s, Total 300s)")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', color='#CCCCCC')
    plt.minorticks_on()
    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig(r"z:\Data\Plotting\Plots\Independent FET\flow_rate_plot_from_grafana.pdf", format='pdf', dpi=300)
    plt.tight_layout()
    plt.show()





