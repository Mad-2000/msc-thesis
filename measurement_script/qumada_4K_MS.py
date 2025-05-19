""

#%% Import

#%% Imports
import json
import yaml
import time
import numpy as np


from os.path import expanduser  
import numpy as np
from tqdm import tqdm
import qcodes as qc
import matplotlib.pyplot as plt
from datetime import datetime

from qcodes.station import Station
from qumada.instrument.custom_drivers.Harvard.Decadac import Decadac
from qumada.instrument.mapping.Harvard.Decadac import DecadacMapping
from qcodes_contrib_drivers.drivers.QDevil.QDAC1 import QDac, Mode
from qumada.instrument.buffered_instruments import BufferedSR830 as SR830
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400 as Keithley_2400
from qumada.instrument.mapping.QDevil.qdac import QDacMapping
from qumada.instrument.mapping.base import save_mapped_terminal_parameters, load_mapped_terminal_parameters
# from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.dataset import (
    Measurement,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment
)

from qumada.instrument.mapping import (
    add_mapping_to_instrument,
    QDAC_MAPPING,
    #MFLI_MAPPING,
    KEITHLEY_2400_MAPPING,
    SR830_MAPPING,
    DECADAC_MAPPING)
from qumada.instrument.mapping import map_gates_to_instruments, map_terminals_gui
from qumada.measurement.scripts import (
    Generic_1D_Sweep,
    Generic_nD_Sweep,
    Generic_1D_parallel_Sweep,
    Generic_1D_parallel_asymm_Sweep,
    Generic_2D_Sweep_buffered,
    Generic_1D_Sweep_buffered,
    # Generic_2D_Sweep,
    Generic_1D_Sweep,
    Timetrace,
    Timetrace_buffered,
    )
from qumada.utils.load_from_sqlite_db import load_db
from qumada.utils.generate_sweeps import generate_sweep, replace_parameter_settings
from qumada.utils.ramp_parameter import *
from qumada.utils.GUI import open_web_gui
from qumada.instrument.buffers.buffer import map_buffers, map_triggers
from qumada.measurement.device_object import *
# from gate_configs import MAPPINGS_DICT, print_dac_connections, get_special_gates

#  -------------------------------------------------
#  --------- START OF USER DEFINED CONFIG ----------
#  -------------------------------------------------


# Connections:


dac1_connection = 'ASRL12::INSTR'
dac2_connection = 'ASRL21::INSTR'
dac3_connection = 'ASRL5::INSTR'
dac4_connection = 'ASRL20::INSTR'
dac5_connection = 'ASRL3::INSTR'
lockin_up_connection = 'GPIB0::8::INSTR'
lockin_down_connection = 'GPIB0::13::INSTR'
keithley_right_connection = 'GPIB0::25::INSTR'
keithley_left_connection = 'GPIB0::19::INSTR'


# Storage options
device = 'S11_B2'
TYPE = "GDSET_Smile"
BONDING_TYPE = "IMOspcb16tbv2"
HERMIT_SLOT = "left"
# HERMIT_SLOT = "right"

date = datetime.now().strftime("%Y%m%d")

database_path = f"{expanduser('~')}/Documents/Measurements/Marcogliese/Batch26/{TYPE}/{device}_{date}_Qumada_Test.db"
#

# print_dac_connections(TYPE,BONDING_TYPE,HERMIT_SLOT)
# t_o_gates = get_special_gates(TYPE,BONDING_TYPE,HERMIT_SLOT,validate_not_in_dac_map=True)



#%%

# sample_config = {
#     "sample_name": device,
#     "gate_mapping": MAPPINGS_DICT['gate_mapping'][TYPE][BONDING_TYPE],
#     "dac_mapping": {
#         'dac1': MAPPINGS_DICT['dac_mapping'][HERMIT_SLOT]["dac1"],
#         'dac2': MAPPINGS_DICT['dac_mapping'][HERMIT_SLOT]["dac2"],
#         'dac3': MAPPINGS_DICT['dac_mapping'][HERMIT_SLOT]["dac3"],
#         'dac4': MAPPINGS_DICT['dac_mapping'][HERMIT_SLOT]["dac4"],
#         'dac5': MAPPINGS_DICT['dac_mapping'][HERMIT_SLOT]["dac5"],
#         },
    
#     "lockin_amplitude": {
#         "lockin_amplitude_down": 1,
#         "lockin_amplitude_up": 1

#         }
#     }


#%%

try:
    assert 'station' not in locals().keys()

except AssertionError:
    print('Setup already initialized, skipping initialization')

else:   
    # Initialize Instruments
    qc.Instrument.close_all()
    # Clock
    #clock = Clock('clock')


    # DECADAC
    dac1 = Decadac(
                    'dac1',
                    dac1_connection,
                    min_val=-10,
                    max_val=10, 
                    terminator='\n'
                    )
    dac2 = Decadac(
                    'dac2',
                    dac2_connection,
                    min_val=-10,
                    max_val=10,
                    terminator='\n'
                    )
    dac3 = Decadac(
                    'dac3',
                    dac3_connection,
                    min_val=-10,
                    max_val=10,
                    terminator='\n'
                    )
    dac4 = Decadac(
                    'dac4',
                    dac4_connection,
                    min_val=-10,
                    max_val=10,
                    terminator='\n'
                    )
    dac5 = Decadac(
                    'dac5',
                    dac5_connection,
                    min_val=-10,
                    max_val=10,
                    terminator='\n'
                    ) 
        
    # Lockins
    lockin_down = SR830("lockin_down", lockin_down_connection)
    lockin_up = SR830("lockin_up", lockin_up_connection)


    # Keithleys
    keithley_left = Keithley_2400('keithley_left', keithley_left_connection)
    keithley_right = Keithley_2400('keithley_right', keithley_right_connection)


    time.sleep(5)

    keithley_left.output.set(1)
    keithley_right.output.set(1)


    # Aquire voltage once as it seems to stabelize the connection
    _ = keithley_left.volt()
    _ = keithley_right.volt()

    keithley_left.rangev(5)
    keithley_left.rangei(1e-06)
    keithley_left.compliancei.set(1e-06)
    
    keithley_right.rangev(5)
    keithley_right.rangei(1e-06)
    keithley_right.compliancei.set(1e-06)

    # Aquire voltage once as it seems to stabelize the connection
    _ = keithley_left.volt()
    _ = keithley_right.volt()

    
    # Add to Station
    station = qc.Station(
                            dac1,
                            dac2,
                            dac3,
                            dac4, 
                            dac5,
                            lockin_up,
                            lockin_down,
                            keithley_right,
                            keithley_left
                            )
    for i in range(1,6):
        add_mapping_to_instrument(station.components[f"dac{i}"] , mapping = DecadacMapping())
        station.components[f"dac{i}"].channels.update_period.set(50)
    add_mapping_to_instrument(lockin_up, mapping = SR830_MAPPING)
    add_mapping_to_instrument(lockin_down, mapping = SR830_MAPPING)
    add_mapping_to_instrument(keithley_left, mapping = KEITHLEY_2400_MAPPING)
    add_mapping_to_instrument(keithley_right, mapping = KEITHLEY_2400_MAPPING)

    print("Look up :)")
    if input("Reset Dac Voltages (Y/N)").lower() == "y":
        print("Resetting Dacs")
        for i in range(1, 6):
            station.components[f"dac{i}"].channels.volt(0)
    else:
        print("Keeping Voltages")
# Initialize database
# %% Initiaize
qc.initialise_or_create_database_at(database_path)
qc.load_or_create_experiment(experiment_name="LEFT SET",
                            sample_name='S11_B2'
                            )
# %%
num_points = 200
backsweep = False



parameters = {
    "Source_Drain_Left": {
        "amplitude": { 
            "type": "static",
            "value": 1
        },
        "frequency": {
            "type": "static",
            "value": 117
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"],
        },
        "phase": {
            "type": ""
        },
    },
    "Left Top Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"]
        }
    },
    # Only one top gate is required as both are connected to same channel. Keep the second constant or unused
    "Left Bottom Accumulation": {
        "voltage": {
            "type": "gettable",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "gettable",
        #     "break_conditions": ["val > 1e-9"]
        # },
    },
    "Left Top Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Bottom Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Plunger": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Screening Gate": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 0.3, num_points, backsweep = backsweep),
            "value": 0,
        },    
    },
    "Source_Drain_Right": {
        "amplitude": { 
            "type": "static",
            "value": 1
        },
        "frequency": {
            "type": "static",
            "value": 173
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"],
        },
        "phase": {
            "type": "gettable"
        },
    },
    "Right Top Accumulation": {
        "voltage": {
            "type": "static gettable",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "static"
        }
    },
    # Only one top gate is required as both are connected to same channel. Keep the second constant or unused
    "Right Bottom Accumulation": {
        "voltage": {
            "type": "gettable",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "static"
        # }
    },
    "Right Top Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Bottom Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Plunger": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Screening Gate": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },    
    },
}
#%%
device = QumadaDevice.create_from_dict(parameters, station =station, make_terminals_global=True, namespace = globals())
load_mapped_terminal_parameters(station = station, path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping_w_current.json" ,terminal_parameters = device.instrument_parameters,)

device.mapping()
mapping = device.instrument_parameters
#%%

# %% Accumulation
script = Generic_1D_parallel_asymm_Sweep()
script.setup(
    parameters, 
    measurement_name = "Accumulation",
    metadata = None,
)
#load_mapped_terminal_parameters(station = station, path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping1.json" ,terminal_parameters = script.gate_parameters,)

map_terminals_gui(station.components, script.gate_parameters, mapping)

#%%
script.run(insert_metadata_into_db=False)
# %% Due Timetrace

device.timetrace(duration  = 18)

# %%  hysteresis
left_SET_gates = [device.Left_Bottom_Barrier, device.Left_Plunger, device.Left_Top_Barrier, device.Left_Top_Accumulation]
Source_Drain_Left.current.break_conditions = []
Right_Top_Accumulation.current.type = '' 
for gate in left_SET_gates:
    current_voltage = gate()
    gate.voltage.measured_ramp(0, num_points= 100, backsweep = True)

#%% 2D Barrier Barrier Scan
Left_Top_Barrier(0.3)
Left_Bottom_Barrier.voltage.ramp(0.2)
device.sweep_2D(slow_param = Left_Top_Barrier.voltage, fast_param =Left_Bottom_Barrier.voltage, slow_param_range=0.4, fast_param_range= 0.4, name ="our measurement")
    


















# %%
num_points = 200
backsweep = False



parameters = {
    "Source_Drain_Left": {
        "amplitude": { 
            "type": "static",
            "value": 1
        },
        "frequency": {
            "type": "static",
            "value": 111
        },
        "current": {
            "type": "",
            "break_conditions": ["val > 100e-12"],
        },
        "phase": {
            "type": ""
        },
    },
    "Left Top Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": []
        },
    },
    "Left Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": []
        },
    },
    "Left Top Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Bottom Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Plunger": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Screening Gate": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },    
    },
    "Source_Drain_Right": {
        "amplitude": { 
            "type": "static",
            "value": 1
        },
        "frequency": {
            "type": "static",
            "value": 173
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 100e-12"],
        },
        "phase": {
            "type": "gettable"
        },
    },
    "Right Top Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"]
        },
    },
    "Right Bottom Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"]
        },
    },
    "Right Top Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Bottom Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Plunger": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Screening Gate": {
        "voltage": {
            "type": "static gettable",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },    
    },
}

# %% Update Parameters
qc.load_or_create_experiment(experiment_name="Right SET Till Test",
                            sample_name='S11_B2'
                            )
# %%
device.load_from_dict(parameters)
# %% Accumulation
script = Generic_1D_parallel_asymm_Sweep()
script.setup(
    parameters, 
    measurement_name = "Test Accumulation",
    metadata = None,
)
#load_mapped_terminal_parameters(station = station, path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping1.json" ,terminal_parameters = script.gate_parameters,)

map_terminals_gui(station.components, script.gate_parameters, mapping)

#%%
script.run(insert_metadata_into_db=False)



# %%
device.timetrace(180)
# %%



# %%  hysteresis

right_SET_gates = [device.Right_Bottom_Barrier, device.Right_Plunger, device.Right_Top_Barrier, device.Right_Top_Accumulation]
Source_Drain_Right.current.break_conditions = [] # removing break conditions

# Right_Top_Accumulation.current.type = '' 
for gate in right_SET_gates:
    current_voltage = gate()
    gate.voltage.measured_ramp(-0.2, num_points= 100, backsweep = True)




#%% 2D Barrier Barrier Scan
Left_Top_Barrier(0.3)
Left_Bottom_Barrier.voltage.ramp(0.2)
device.sweep_2D(slow_param = Left_Top_Barrier.voltage, fast_param =Left_Bottom_Barrier.voltage, slow_param_range=0.4, fast_param_range= 0.4, name ="our measurement")
    