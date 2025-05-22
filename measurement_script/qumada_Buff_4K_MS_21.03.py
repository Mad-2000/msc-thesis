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

from datetime import datetime
from qcodes.dataset import (
    Measurement,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
)
from qcodes.station import Station
from qcodes.dataset import (
    Measurement,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
)
from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400 as Keithley_2400
from qumada.instrument.buffered_instruments import BufferedDummyDMM as DummyDmm
from qumada.instrument.buffers.buffer import (
    load_trigger_mapping,
    map_triggers,
    save_trigger_mapping,
)
from qumada.instrument.custom_drivers.Harvard.Decadac import Decadac
from qumada.instrument.mapping.Harvard.Decadac import DecadacMapping
from qumada.instrument.buffered_instruments import BufferedMFLI as mfli
from qumada.instrument.custom_drivers.Dummies.dummy_dac import DummyDac
from qumada.instrument.mapping import (
    DUMMY_DMM_MAPPING,
    MFLI_MAPPING,
    KEITHLEY_2400_MAPPING,
    add_mapping_to_instrument,
    map_terminals_gui,
)
from qumada.instrument.mapping.Dummies.DummyDac import DummyDacMapping
from qumada.measurement.device_object import *
from qumada.measurement.scripts import (
    Generic_1D_parallel_Sweep,
    Generic_1D_Sweep,
    Generic_1D_Sweep_buffered,
    Generic_nD_Sweep,
    Timetrace,
    Timetrace_with_Sweeps_buffered,
)
from qumada.utils.generate_sweeps import generate_sweep
from qumada.utils.load_from_sqlite_db import load_db

#%%
#  -------------------------------------------------
#  --------- START OF USER DEFINED CONFIG ----------
#  -------------------------------------------------


# Connections:


# dac1_connection = 'ASRL12::INSTR'
# dac2_connection = 'ASRL21::INSTR'
# dac3_connection = 'ASRL5::INSTR'
# dac4_connection = 'ASRL20::INSTR'
# dac5_connection = 'ASRL3::INSTR'
# lockin_up_connection = 'GPIB0::8::INSTR'
# lockin_down_connection = 'GPIB0::13::INSTR'
# keithley_right_connection = 'GPIB0::25::INSTR'
# keithley_left_connection = 'GPIB0::19::INSTR'


# Storage options
folder = 'Batch 29'
device_name = 'S15_A1'
TYPE = "QuBus_IndFanoutFET"
BONDING_TYPE = "IMOspcb16tbv2"
HERMIT_SLOT = "left"
# HERMIT_SLOT = "right"

date = datetime.now().strftime("%Y%m%d")

database_path = rf"{expanduser('~')}/Documents/Measurements\Sasikala\Batch 29\{device_name}_{TYPE}_{date}.db"
#

print(device_name, TYPE, BONDING_TYPE, HERMIT_SLOT)

#%%
# %% Only required to simulate buffered instruments
# As we have only dummy instruments that are not connected, we have to use a global
# trigger event for triggering.


# Setup qcodes station and load instruments
try:
    assert 'station' not in locals().keys()

except AssertionError:
    print('Setup already initialized, skipping initialization')

else:   
    # Initialize Instruments
    qc.Instrument.close_all()
    station = Station()

    # The dummy instruments have a trigger_event attribute as replacement for
    # the trigger inputs of real instruments./
    mfli1 = mfli("mfli1", device = "DEV30413", 
                 serverhost="169.254.180.92", 
                 allow_version_mismatch=True)


    mfli2 = mfli("mfli2", device = "DEV30281", 
                 serverhost="169.254.180.91", 
                 allow_version_mismatch=True)




    dac1 = Decadac("dac1", "ASRL6::INSTR",
                        min_val=-10,
                        max_val=10,
                        terminator='\n')


    dac2 = Decadac("dac2", "ASRL3::INSTR", 
                        min_val=-10,
                        max_val=10,
                        terminator='\n')



    dac3 = Decadac("dac3", "ASRL4::INSTR",
                        min_val=-10,
                        max_val=10,
                        terminator='\n')



    dac4 = Decadac("dac4", "ASRL5::INSTR", 
                        min_val=-10,
                        max_val=10,
                        terminator='\n')



    dac5 = Decadac("dac5", "ASRL7::INSTR", 
                        min_val=-10,
                        max_val=10,
                        terminator='\n')
  
    # Keithleys
    keithley_right_connection = 'GPIB0::25::INSTR'
    keithley_left_connection = 'GPIB0::19::INSTR'

    keithley_left = Keithley_2400('keithley_left', keithley_left_connection)
    keithley_right = Keithley_2400('keithley_right', keithley_right_connection)

    # time.sleep(5)

    keithley_left.output.set(1)
    keithley_right.output.set(1)


    # Aquire voltage once as it seems to stabilize the connection
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
                            mfli1,
                            mfli2,
                            keithley_right,
                            keithley_left
                            )
    for i in range(1,6):
        add_mapping_to_instrument(station.components[f"dac{i}"] , mapping = DecadacMapping())
        station.components[f"dac{i}"].channels.update_period.set(50)
    add_mapping_to_instrument(mfli1, mapping=MFLI_MAPPING)
    add_mapping_to_instrument(mfli2, mapping=MFLI_MAPPING)
    add_mapping_to_instrument(keithley_left, mapping = KEITHLEY_2400_MAPPING)
    add_mapping_to_instrument(keithley_right, mapping = KEITHLEY_2400_MAPPING)

    print("Look up :)")
    if input("Reset Dac Voltages (Y/N)").lower() == "y":
        print("Resetting Dacs")
        for i in range(1, 6):
            station.components[f"dac{i}"].channels.volt(0)
        print("Resetting mflis") 
        mfli1.amplitude.set(0.004)
        mfli2.amplitude.set(0.004)
        print("Resetting Keithleys") 
        keithley_left.volt(0.0)
        keithley_right.volt(0.0)
    else:
        print("Keeping Voltages")


#%%

#%%  Set all DACs to 0 V


for i in range(1, 6):
            station.components[f"dac{i}"].channels.volt(1)

#%%  Set DAC1 to 1 V

station.components["dac1"].channels.volt(1)

#%%  Set DAC2 to 1 V

station.components["dac2"].channels.volt(1)
#%%  Set DAC3 to 1 V

station.components["dac3"].channels.volt(0.1)
#%%  Set DAC4 to 1 V

station.components["dac4"].channels.volt(0.1)
#%%  Set DAC5 to 1 V

station.components["dac5"].channels.volt(0.1)

# %%  Set all DACs to 1 V


for i in range(1, 6):
            station.components[f"dac{i}"].channels.volt(1)


# %% Load database for data storage. This will open a window.
# Alternatively, you can use initialise_or_create_database_at from QCoDeS
# initialise_or_create_database_at(r"C:\Users\lab2\Documents\Measurements\Huckemann\JARA\Batch 33\C1 Device 32\C1_Device_32_2.db")

# We need to create an experiment in the QCoDeS database
# load_or_create_experiment("4K Test", "5QBit_5_V5")
# %% Setup buffers (only need for buffered measurements).

def trigger_set():
    mfli1.instr.auxouts[0].offset(7)
    
def trigger_reset():
    mfli1.instr.auxouts[0].offset(0)
# Those buffer settings specify how the triggers are setup and how the data is recorded.
buffer_settings = {
    # We don't have to specify threshold and mode for our dummy instruments
    "trigger_threshold": 2,
    "trigger_mode": "edge",
    "sampling_rate": 100,
    "num_points": 300,
    "delay": 0,
}

# This tells a measurement script how to start a buffered measurement.
# "Hardware" means that you want to use a hardware trigger. To start a measurement,
# the method provided as "trigger_start" is called. The "trigger_reset" method is called
# at the end of each buffered line, in our case resetting the trigger flag.
# For real instruments, you might have to define a method that sets the output of your instrument
# to a desired value as "trigger_start". For details on other ways to setup your triggers,
# check the documentation.

buffer_script_settings = {
    "trigger_type": "hardware",
    "trigger_start": trigger_set,
    "trigger_reset": trigger_reset,
}

 
 

# %% Initiaize database

qc.initialise_or_create_database_at(database_path)
qc.load_or_create_experiment(experiment_name="LEFT SET",
                            sample_name=device_name
                            )
# %%
num_points = 200
backsweep = True
sweep_min = 0
sweep_max = 2
# We want to include the lever arm effects of barrier gates. 
# As they are on the top most layer, it's influence on the 2DEG is smaller
# compared to PGs and AGs. Using the layer separation distances,  we want to add 
# 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
leverarm_scaling = 1


parameters_LSET = {
    "Source_Drain_Left": {
        "amplitude": { 
            "type": "static",
            "value": 0.0001,
        },
        "frequency": {
            "type": "static",
            "value": 77,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 0.5e-9"],
        },
        "phase": {
            "type": "gettable",
        },
    },
    "Left Top Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"],
        }
    },
    # As we connected the LBA to Dac, we don't need to define current.
    "Left Bottom Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "gettable",
        #     "break_conditions": ["val > 1e-9"]
        # },
    },
    # We want to include the lever arm effects of barrier gates. 
    # As they are on the top most layer, it's influence on the 2DEG is smaller
    # compared to PGs and AGs. Using the layer separation distances,  we want to add 
    # 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
    "Left Top Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Bottom Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Plunger": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Screening Gate": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
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
            "value": 111
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 0.2e-9"],
        },
        "phase": {
            "type": "gettable"
        },
    },
    "Right Top Accumulation": {
        "voltage": {
            "type": "static gettable",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable"
        },
    },
    # CONNECTED TO DAC SO CURRENT CAN'T BE MEASURED
    "Right Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
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
    "BS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "B3": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "B2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "B1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "P2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "P1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "TS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    
}
#%% Device Object Creation and mapping GUI


device = QumadaDevice.create_from_dict(parameters_LSET, station =station,
                                        make_terminals_global=True, 
                                        namespace = globals())
load_mapped_terminal_parameters(station = station, 
    path=rf"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping_{HERMIT_SLOT}slot.json" ,
    terminal_parameters = device.instrument_parameters,)

device.mapping()
mapping = device.instrument_parameters

#%%
# When readjusting the gate parameters, use this argument for updating device object.

device.load_from_dict(parameters_LSET)



# %% Accumulation
script = Generic_1D_parallel_asymm_Sweep()

script.setup(
    parameters_LSET, 
    measurement_name = "Accumulation, SG = 0, SD_T = 0.5nA",
    metadata = None,
)
#load_mapped_terminal_parameters(station = station, path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping1.json" ,terminal_parameters = script.gate_parameters,)

map_terminals_gui(station.components, 
                  script.gate_parameters, 
                  mapping, 
                  skip_gui_if_mapped=True )

#%%
script.run(insert_metadata_into_db=False)


#%% Ramp down gates slowly

left_SET_gates = [device.Left_Bottom_Barrier, 
    device.Left_Plunger, 
    device.Left_Top_Barrier, 
    device.Left_Top_Accumulation, 
    device.Left_Bottom_Accumulation,]
    # device.Left_Screening_Gate]

for gate in left_SET_gates:
    current_voltage = gate()
    gate.voltage.ramp(0.0)
                    
# %% Due Timetrace

device.timetrace(duration  = 30, timestep=0.01)

# %%  hysteresis
left_SET_gates = [device.Left_Bottom_Barrier, 
    device.Left_Plunger, 
    device.Left_Top_Barrier,]
device.Source_Drain_Left.current.break_conditions = []
device.Left_Top_Accumulation.current.break_conditions = ["val > 30e-9"]

for gate in left_SET_gates:
    current_voltage = gate()
    gate.voltage.measured_ramp(0, 
                               num_points= 100, 
                               backsweep = True)

# %% Due Timetrace

device.timetrace(duration  = 30, timestep=0.01)


#%% 2D Barrier Barrier Scan

device.Left_Top_Barrier.voltage.ramp(1.5)
device.Left_Bottom_Barrier.voltage.ramp(2.0)
device.Left_Plunger.voltage.ramp(1.4)
device.Left_Screening_Gate.voltage.ramp(0.2)

device.sweep_2D(slow_param = Left_Top_Barrier.voltage, 
    fast_param = Left_Bottom_Barrier.voltage, 
    slow_param_range=0.3, 
    fast_param_range= 0.25, 
    slow_num_points=50, 
    fast_num_points=100,
    name ="BB sweep P=2.0 V")
    















# %% Initiaize database

qc.initialise_or_create_database_at(database_path)
qc.load_or_create_experiment(experiment_name="RIGHT SET",
                            sample_name=device_name
                            )


# %% RSET Measurement and Parameteres updation

num_points = 200
backsweep = False
sweep_min = 0.0
sweep_max = 1.5
# We want to include the lever arm effects of barrier gates. 
# As they are on the top most layer, it's influence on the 2DEG is smaller
# compared to PGs and AGs. Using the layer separation distances,  we want to add 
# 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
leverarm_scaling = 1.2



parameters_RSET = {
    "Source_Drain_Left": {
        "amplitude": { 
            "type": "static",
            "value": 1,
        },
        "frequency": {
            "type": "static",
            "value": 77,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 0.5e-9"],
        },
        "phase": {
            "type": "gettable",
        },
    },
    "Left Top Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
        }
    },
    # LBA IS CONNECTED TO DAC SO NO CURRENT
    "Left Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "gettable",
        #     "break_conditions": ["val > 1e-9"]
        # },
    },
    "Left Top Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Bottom Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Plunger": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Screening Gate": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 1, 
                                        num_points, 
                                        backsweep = backsweep),
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
            "value": 111
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 2e-9"],
        },
        "phase": {
            "type": "gettable"
        },
    },
    "Right Top Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 10e-9"],
        },
    },
    # LBA IS CONNECTED TO DAC SO NO CURRENT
    "Right Bottom Accumulation": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "static"
        # }
    },
    "Right Top Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Bottom Barrier": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Plunger": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Right Screening Gate": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        }, 
    },
    "BS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "B3": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "B2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "B1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "P2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "P1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "TS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    
}

#%%  GO TO NEXT CELL!! Optional only when we measure RSET first

device = QumadaDevice.create_from_dict(parameters_RSET, 
                                       station =station,
                                       make_terminals_global=True, 
                                        namespace = globals())
load_mapped_terminal_parameters(station = station, 
                                path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping_rightslot.json" ,
                                terminal_parameters = device.instrument_parameters,)

device.mapping()
mapping = device.instrument_parameters

# %% otherwise continue here

device.load_from_dict(parameters_RSET)


# %% Accumulation

script = Generic_1D_parallel_asymm_Sweep()

script.setup(
    parameters_RSET, 
    measurement_name = "Accumulation, SG = 0, SD_T = 2nA",
    metadata = None,
)

map_terminals_gui(station.components, 
                  script.gate_parameters, 
                  mapping, 
                  skip_gui_if_mapped=True)

#%% Accumulation
script.run(insert_metadata_into_db=False)



# %% Time Stability

device.timetrace(30,timestep=0.01)

# %% RAMP DOWN GATES SLOWLY

Right_SET_gates = [device.Right_Bottom_Barrier, 
    device.Right_Plunger, 
    device.Right_Top_Barrier, 
    device.Right_Top_Accumulation, 
    device.Right_Bottom_Accumulation]

for gate in Right_SET_gates:
    current_voltage = gate()
    gate.voltage.ramp(0.0)


# %%  Hysteresis

right_SET_gates = [device.Right_Bottom_Barrier, 
                   device.Right_Plunger, 
                   device.Right_Top_Barrier]
Source_Drain_Right.current.break_conditions = [] # removing break conditions

# Right_Top_Accumulation.current.type = '' 
for gate in right_SET_gates:
    current_voltage = gate()
    gate.voltage.measured_ramp(0, 
                               num_points= 100, 
                               backsweep = True)

# %%1D Screening Gate Sweep(Hysterisis)

device.Right_Screening_Gate.voltage.measured_ramp(0.2, 
                                                  num_points = 100, 
                                                  backsweep = True)



#%% 2D Barrier Barrier Scan

device.sweep_2D(slow_param =Right_Plunger.voltage, 
                fast_param =Right_Bottom_Barrier.voltage, 
                slow_param_range=1.0, 
                fast_param_range= 2.0,
                slow_num_points=50,
                fast_num_points=100, 
                name ="BP sweep", )
    
#%% 2D Plunger-SG Scan

device.sweep_2D(slow_param =Right_Plunger.voltage, 
                fast_param =Right_Screening_Gate.voltage, 
                slow_param_range=1.0, 
                fast_param_range= 0.4,
                slow_num_points=50,
                fast_num_points=100, 
                name ="PSG sweep", )
# %% Accumulation hysteresis


script = Generic_1D_parallel_asymm_Sweep()

script.setup(
    parameters_RSET, 
    measurement_name = "Accumulation hysteresis",
    metadata = None,
)

map_terminals_gui(station.components, 
                  script.gate_parameters, 
                  mapping, 
                  skip_gui_if_mapped=False)

# %%
script.run(insert_metadata_into_db=False)








#%% QUBUS Measurement!!!

# %% Initiaize database

qc.initialise_or_create_database_at(database_path)
qc.load_or_create_experiment(experiment_name="LEFT SET",
                            sample_name=device_name
                            )



#%%

num_points = 200
backsweep = False
sweep_min = 0
sweep_max = 1.5

# We want to include the lever arm effects of barrier gates. 
# As they are on the top most layer, it's influence on the 2DEG is smaller
# compared to PGs and AGs. Using the layer separation distances,  we want to add 
# 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
leverarm_scaling = 1


parameters_QB_LSET = {
    "Source_Drain_Left": {
        "amplitude": { 
            "type": "static",
            "value": 0.0001,
        },
        "frequency": {
            "type": "static",
            "value": 257,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 0.2e-9"],
        },
        "phase": {
            "type": "gettable",
        },
    },
    "Left Top Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"],
        }
    },
    # As we connected the LBA to Dac, we don't need to define current.
    "Left Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "gettable",
        #     "break_conditions": ["val > 1e-9"]
        # },
    },
    # We want to include the lever arm effects of barrier gates. 
    # As they are on the top most layer, it's influence on the 2DEG is smaller
    # compared to PGs and AGs. Using the layer separation distances,  we want to add 
    # 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
    "Left Top Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Bottom Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Plunger": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Screening Gate": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0.0,
        },    
    },
    "Source_Drain_Right": {
        "amplitude": { 
            "type": "static",
            "value": 0.0001
        },
        "frequency": {
            "type": "static",
            "value": 134
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
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable"
        },
    },
    # CONNECTED TO DAC SO CURRENT CAN'T BE MEASURED
    "Right Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
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
    "BS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "LP1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "LB1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "LP2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "LB2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "LP3": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },  
    "RP1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RB1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RP2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "TS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RB2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RP3": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "Independent FET": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
}
#%% CHANGE TO RIGHT SLOT WHEN MEASURING RIGHT SAMPLE IN HERMIT!!!(MAPPING_RIGHTSLOT.JSON)

device = QumadaDevice.create_from_dict(parameters_QB_LSET, station=station, namespace=globals())
device.map_terminals(path = rf"C:\Users\lab2\Documents\Userfiles\Sasikala\Scripts\qumada\mapping_Qubus_IndFET_{HERMIT_SLOT}_2.json")
device.buffer_script_setup = buffer_script_settings
device.buffer_settings = buffer_settings
device.map_triggers()

device.set_stored_values()

# save_mapped_terminal_parameters(station = station, 
#     path=rf"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping_Qubus_IndFET_{HERMIT_SLOT}.json" ,
#     terminal_parameters = device.terminal_parameters,)

device.map_terminals()


1
#%%
# When readjusting the gate parameters, use this argument for updating device object.

device.load_from_dict(parameters_QB_LSET)



# %% Accumulation


# left_screening_gates = [LS, B1L, ScrLT1, ScrLB]
# for gate in left_screening_gates:
#     gate.voltage(0.4)
#     gate.voltage.type = "static gettable"

Source_Drain_Left.current.type ="gettable"
Source_Drain_Left.current.break_conditions = ["val > 0.2e-9"]
Source_Drain_Right.current.type = ""
Left_Top_Accumulation.current.break_conditions = ["val > 1e-9"]


accu_left_gates = [device.Left_Bottom_Barrier, 
    device.Left_Plunger, 
    device.Left_Top_Barrier, 
    device.Left_Top_Accumulation, 
    device.Left_Bottom_Accumulation,]
    # device.Left_Screening_Gate]#, LS, B1L, ScrLT1, ScrLB]

accu_left_setpoints = [1.5, 1.5, 1.5, 1.5, 1.5]#, 0.55, 0.4, 0.45, 0.45]
# accu_left_setpoints = [2, 2, 2, 2, 2]



# device.Left_Screening_Gate.voltage.ramp(0.2)
device.sweep_parallel([g.voltage for g in accu_left_gates], 
                      target_values = accu_left_setpoints, 
                      num_points = 200, 
                      name="Accumulation Left SET, V_sg = 0,V_InFF = 0"
                      )


#%% Ramp down gates slowly

left_SET_gates = [device.Left_Bottom_Barrier, 
    device.Left_Plunger, 
    device.Left_Top_Barrier, 
    device.Left_Top_Accumulation, 
    device.Left_Bottom_Accumulation]

for gate in left_SET_gates:
    current_voltage = gate()
    gate.voltage.ramp(0)

print(Source_Drain_Left.current())
print(Left_Top_Accumulation.current())
print(device.voltages())
                    
#%%
Source_Drain_Left.current.break_conditions = ["val > 0.5e-9"]

device.Independent_FET.voltage.measured_ramp(-0.5, start=-0.1, name = "Independent FET sweep 0 to -0.5V", 
                                             buffered = False,
                                              num_points = 100, 
                                              backsweep =True)
# device.save_state("accumulated_left")
# %% Due Timetrace

device.timetrace(duration  = 30, timestep=0.01)

# %%  hysteresis
left_SET_gates = [device.Left_Bottom_Barrier, 
    device.Left_Plunger, 
    device.Left_Top_Barrier]

for gate in left_SET_gates:
    current_voltage = gate()
    gate.voltage.measured_ramp(-0.3,
                               num_points= 100, 
                               backsweep = True)

# %% Due Timetrace

device.timetrace(duration  = 30, timestep=0.01)


#%% 2D Barrier Barrier Scan

device.Left_Top_Barrier.voltage.ramp()
device.Left_Bottom_Barrier.voltage.ramp()
# device.Left_Plunger.voltage.ramp(0.6)
# device.Left_Screening_Gate.voltage.ramp(0.2)



device.sweep_2D(slow_param = Left_Bottom_Barrier.voltage, 
    fast_param = Left_Top_Barrier.voltage, 
    slow_param_range=0.6, 
    fast_param_range= 0.6, 
    slow_num_points=20, 
    fast_num_points=30,
    name ="BB sweep P = 0.8V")
#%% Loop for BB with varying plunger

for i in np.lispace(0.6, 0.8, 10):
    device.Left_Plunger.voltage.ramp(i)

    device.sweep_2D(slow_param = Left_Bottom_Barrier.voltage, 
    fast_param = Left_Top_Barrier.voltage, 
    slow_param_range=0.6, 
    fast_param_range= 0.6, 
    slow_num_points=20, 
    fast_num_points=30,
    name ="BB sweep P = {i}V")

#%% 2D Barrier Plunger Scan

# device.Left_Top_Barrier.voltage.ramp(0.8)
# device.Left_Bottom_Barrier.voltage.ramp(0.7)
# device.Left_Plunger.voltage.ramp(0.8)
# device.Left_Screening_Gate.voltage.ramp(0.2)

device.sweep_2D(slow_param = Left_Top_Barrier.voltage, 
    fast_param = Left_Plunger.voltage, 
    slow_param_range=0.6, 
    fast_param_range= 0.6, 
    slow_num_points=30, 
    fast_num_points=50,
    name ="BB sweep P=0.95 V")
    















# %% Initiaize database

qc.initialise_or_create_database_at(database_path)
qc.load_or_create_experiment(experiment_name="RIGHT SET",
                            sample_name=device_name
                            )


# %% RSET Measurement and Parameteres updation

num_points = 200
backsweep = False
sweep_min = 0
sweep_max = 1.5

# We want to include the lever arm effects of barrier gates. 
# As they are on the top most layer, it's influence on the 2DEG is smaller
# compared to PGs and AGs. Using the layer separation distances,  we want to add 
# 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
leverarm_scaling = 1


parameters_QB_RSET = {
    "Source_Drain_Left": {
        "amplitude": { 
            "type": "static",
            "value": 0.0001,
        },
        "frequency": {
            "type": "static",
            "value": 257,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 0.5e-9"],
        },
        "phase": {
            "type": "gettable",
        },
    },
    "Left Top Accumulation": {
        "voltage": {
            "type": "static gettable",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "static gettable",
            "break_conditions": ["val > 10e-9"],
        }
    },
    # As we connected the LBA to Dac, we don't need to define current.
    "Left Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        # "current": {
        #     "type": "gettable",
        #     "break_conditions": ["val > 1e-9"]
        # },
    },
    # We want to include the lever arm effects of barrier gates. 
    # As they are on the top most layer, it's influence on the 2DEG is smaller
    # compared to PGs and AGs. Using the layer separation distances,  we want to add 
    # 1.2 times increase to the Barrier voltages. 2 x 1.2 = 2.4.
    "Left Top Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Bottom Barrier": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        leverarm_scaling*sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Plunger": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
    },
    "Left Screening Gate": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0.0,
        },    
    },
    "Source_Drain_Right": {
        "amplitude": { 
            "type": "static",
            "value": 0.0001
        },
        "frequency": {
            "type": "static",
            "value": 134
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
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable"
        },
    },
    # CONNECTED TO DAC SO CURRENT CAN'T BE MEASURED
    "Right Bottom Accumulation": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
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
    "BS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },   
    },
    "LP1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "LB1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "LP2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "LB2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "LP3": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },  
    "RP1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RB1": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RP2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    },
    "TS": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RB2": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "RP3": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
    "Independent FET": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(0, 2, num_points, backsweep = backsweep),
            "value": 0,
        },
    }, 
}

#%%  GO TO NEXT CELL!! Optional only when we measure RSET first


# %% otherwise continue here

device.load_from_dict(parameters_QB_RSET)

#%% Accumulation RSET

Source_Drain_Right.current.type ="gettable"
Source_Drain_Right.current.break_conditions = ["val > 0.2e-9"]
Source_Drain_Left.current.type = ""
Right_Top_Accumulation.current.break_conditions = ["val > 1e-9"]


accu_right_gates = [device.Right_Bottom_Barrier, 
    device.Right_Plunger, 
    device.Right_Top_Barrier, 
    device.Right_Top_Accumulation, 
    device.Right_Bottom_Accumulation,]
    # device.Left_Screening_Gate]#, LS, B1L, ScrLT1, ScrLB]

accu_right_setpoints = [1.5, 1.5, 1.5, 1.5, 1.5]#, 0.55, 0.4, 0.45, 0.45]
# accu_left_setpoints = [2, 2, 2, 2, 2]



# device.Left_Screening_Gate.voltage.ramp(0.2)
device.sweep_parallel([g.voltage for g in accu_right_gates], 
                      target_values = accu_right_setpoints, 
                      num_points = 200, 
                      name="Accumulation Right SET, V_sg = 0,V_InFF = 0"
                      )
# %% Time Stability
Source_Drain_Right.current.type ="static"

device.timetrace(30, buffered=True, timestep=0.01)



# %%

# Source_Drain_Left.current.break_conditions = ["val > 0.5e-9"]
Right_Top_Accumulation.current

device.Independent_FET.voltage.measured_ramp(0.5, name = "Independent FET sweep", buffered = True,
                                              num_points = 100, 
                                              backsweep =True)


# %% RAMP DOWN GATES SLOWLY

Right_SET_gates = [device.Right_Bottom_Barrier, 
    device.Right_Plunger, 
    device.Right_Top_Barrier, 
    device.Right_Top_Accumulation, 
    device.Right_Bottom_Accumulation]

for gate in Right_SET_gates:
    current_voltage = gate()
    gate.voltage.ramp(0.0)


# %%  Hysteresis

right_SET_gates = [device.Right_Bottom_Barrier, 
                   device.Right_Plunger, 
                   device.Right_Top_Barrier]
Source_Drain_Right.current.break_conditions = [] # removing break conditions

# Right_Top_Accumulation.current.type = '' 
for gate in right_SET_gates:
    current_voltage = gate()
    gate.voltage.measured_ramp(-0.5, 
                               num_points= 100, 
                               backsweep = True)

# %%1D Screening Gate Sweep(Hysterisis)

device.Right_Screening_Gate.voltage.measured_ramp(0.2, 
                                                  num_points = 100, 
                                                  backsweep = True)



#%% 2D Barrier Barrier Scan

device.sweep_2D(slow_param =Right_Plunger.voltage, 
                fast_param =Right_Bottom_Barrier.voltage, 
                slow_param_range=1.0, 
                fast_param_range= 2.0,
                slow_num_points=50,
                fast_num_points=100, 
                name ="BP sweep", )
    
#%% 2D Plunger-SG Scan

device.sweep_2D(slow_param =Right_Plunger.voltage, 
                fast_param =Right_Screening_Gate.voltage, 
                slow_param_range=1.0, 
                fast_param_range= 0.4,
                slow_num_points=50,
                fast_num_points=100, 
                name ="PSG sweep", )
# %% Accumulation hysteresis
# %% Accumulation

script = Generic_1D_parallel_asymm_Sweep()

script.setup(
    parameters_QB_RSET, 
    measurement_name = "Accumulation, SG = 0, SDmin = 0.5nA",
    metadata = None,
)

map_terminals_gui(station.components, 
                  script.gate_parameters, 
                  mapping, 
                  skip_gui_if_mapped=True)



#%%  TEST STRUCTURES


# %% Initiaize database

experiment_name="BOT FET"
qc.initialise_or_create_database_at(database_path)
qc.load_or_create_experiment(experiment_name=experiment_name,
                            sample_name=device_name
                            )

# %%
num_points = 200
backsweep = True
sweep_min = 0
sweep_max = 0.5



parameters_FET = {
    "Source_Drain_on": {
        "amplitude": { 
            "type": "static",
            "value": 1,
        },
        "frequency": {
            "type": "static",
            "value": 77,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 100e-9"],
        },
        "phase": {
            "type": "gettable",
        },
    },
    "Accumulation_on": {
        "voltage": {
            "type": "dynamic",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1000e-9"],
        }
    },

    "Source_Drain_standby": {
        "amplitude": { 
            "type": "static",
            "value": 0.004,
        },
        "frequency": {
            "type": "static",
            "value": 111,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 1e-9"],
        },
        "phase": {
            "type": "gettable",
        },
    },
    "Accumulation_standby": {
        "voltage": {
            "type": "static",
            "setpoints": generate_sweep(sweep_min, 
                                        sweep_max, 
                                        num_points, 
                                        backsweep = backsweep),
            "value": 0,
        },
        "current": {
            "type": "gettable",
            "break_conditions": ["val > 10e-9"],
        }
    },
}

#%% CHANGE TO RIGHT SLOT WHEN MEASURING RIGHT SAMPLE IN HERMIT!!!(MAPPING_RIGHTSLOT.JSON)


device = QumadaDevice.create_from_dict(parameters_FET, station =station,
                                        make_terminals_global=True, 
                                        namespace = globals())
# load_mapped_terminal_parameters(station = station, 
#     path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping_rightslot.json" ,
#     terminal_parameters = device.instrument_parameters,)

device.mapping()
mapping = device.instrument_parameters

#%%
# When readjusting the gate parameters, use this argument for updating device object.

device.load_from_dict(parameters_FET)



# %% Accumulation
script = Generic_1D_parallel_asymm_Sweep()

script.setup(
    parameters_FET, 
    measurement_name = f"Accumulation_{experiment_name}",
    metadata = None,
)
#load_mapped_terminal_parameters(station = station, path=r"C:\Users\lab2\Documents\Userfiles\Marcogliese\qumada_scripts\mapping1.json" ,terminal_parameters = script.gate_parameters,)

map_terminals_gui(station.components, 
                  script.gate_parameters, 
                  mapping, 
                  skip_gui_if_mapped=False )

#%%
script.run(insert_metadata_into_db=False)


# %% Due Timetrace

device.timetrace(duration  = 30, timestep=0.01)
