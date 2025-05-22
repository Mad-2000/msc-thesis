#%%

import poisson_schr_new as ps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#%% Matplotlib Parameteres

plt.rcParams.update({'font.size': 20})
RWTHblue = (0,103/255,166/255)
RWTHred = (161/255,16/255,53/255)
RWTHgreen = (87/255,171/255,39/255)
RWTHlightblue = (119/255,158/255,201/255)

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 18
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import convolve


def lin(x,a,b):
    return a*x+b
colorarr = [RWTHblue,RWTHred,RWTHgreen, 
            'c', 'm', 'y', 'k',
            'orange','darkgreen','navy','grey','purple','gold',
            'saddlebrown','tomato']#'b', 'g', 'r', ,'maroon'

plt.rcParams['figure.figsize'] = (5,4)


#%% Mesa Part Defining the HS

def mesa_layerstack(qreg: int, 
                    T: float, 
                    t_Ox=20, 
                    t_Si_cap=2, 
                    t_SiGe_spacer=30, 
                    t_Si_QW=10, 
                    t_SiGe_buffer=1000):
    """    
    
    qreg: Layer in which we want to solve the Poisson-Schroedinger equations (silicon quantum well)
    T: target temperature (K)

    """

    # Initialize layerstack
    hs = ps.poiss_schr()

    # Background doping concentration
    n = 1e16


    Ec_Si = 1.1/2

    # Gate dielectric
    hs.addlayer(t_Ox, Ec=2.1, epsr=7.)

    # Si cap (to protect again SiGe oxidation)
    hs.addlayer(t_Si_cap, Ec=Ec_Si, epsr=11.7, n=n)  

    # SiGe spacer 
    hs.addlayer(t_SiGe_spacer, Ec=Ec_Si+0.15, epsr=13.05, n=n) 

    # Si quantum well 
    hs.addlayer(t_Si_QW, Ec=Ec_Si,  epsr=11.7, n=n)  # solve here

    # SiGe buffer 
    hs.addlayer(t_SiGe_buffer, Ec=Ec_Si+0.15, epsr=13.05, n=n) 

    # Dopant energy offset from conduction band edge
    Edx = -.0206

    # Effective mass of electron in silicon
    meff = 0.19

    # Step size 
    h = 0.05

    # Number of eigenvalues
    nev = 3

    hs.setup(qreg, Edx=Edx, meff=meff, h=h, nev=nev, T=T)

    return hs



#%% Outside Mesa Part Defining the HS

def no_mesa_layerstack(qreg: int, 
                    T: float, 
                    t_Ox=20, 
                    t_SiGe_top=20, 
                    t_SiGe_buffer=1000):
    """    
    
    qreg: Layer in which we want to solve the Poisson-Schroedinger equations (silicon quantum well)
    T: target temperature (K)

    """

    # Initialize layerstack
    hs = ps.poiss_schr()

    # Background doping concentration
    n = 1e16

    Ec_Si = 1.1/2

    # Gate dielectric
    hs.addlayer(t_Ox, Ec=2.1, epsr=7.)

    # SiGe top
    hs.addlayer(t_SiGe_top, Ec=Ec_Si+0.15, epsr=13.05, n=n) 

    # SiGe buffer 
    hs.addlayer(t_SiGe_buffer, Ec=Ec_Si+0.15, epsr=13.05, n=n) 

    # Dopant energy offset from conduction band edge
    Edx = -.0206

    # Effective mass of electron in silicon
    meff = 0.19

    # Step size 
    h = 0.05

    # Number of eigenvalues
    nev = 3

    hs.setup(qreg, Edx=Edx, meff=meff, h=h, nev=nev, T=T)

    return hs


#%% Accumulation Together
T=4.2

heterostructure_mesa = mesa_layerstack(qreg=3, T=T, t_Ox=20, 
                    t_Si_cap=2, 
                    t_SiGe_spacer=30, 
                    t_Si_QW=10, 
                    t_SiGe_buffer=200)
heterostructure_mesa_cap = mesa_layerstack(qreg=1, T=T, t_Ox=20, 
                    t_Si_cap=2, 
                    t_SiGe_spacer=30, 
                    t_Si_QW=10, 
                    t_SiGe_buffer=200)
heterostructure_no_mesa = no_mesa_layerstack(qreg=1,
                                              T=T,
                                              t_Ox=20, 
                    t_SiGe_top=30, 
                    t_SiGe_buffer=200)

heterostructure_mesa_FET= mesa_layerstack(qreg = 3, T=T,
                                           t_Ox=10, 
                    t_Si_cap=2, 
                    t_SiGe_spacer=30, 
                    t_Si_QW=10, 
                    t_SiGe_buffer=200)

V_gate_min = 0.0
V_gate_max = 1.5

heterostructure_mesa.voltage_sweep(V_gate_min, V_gate_max, T, num_pts=50)
heterostructure_mesa_cap.voltage_sweep(V_gate_min, V_gate_max, T, num_pts=50)
heterostructure_no_mesa.voltage_sweep(V_gate_min, V_gate_max, T, num_pts=50)
#%% Plotting ###########
plt.figure(figsize=(6,5))

plt.plot(heterostructure_mesa.V_gate_arr, heterostructure_mesa.sheet_densities)
plt.plot(heterostructure_mesa_cap.V_gate_arr, heterostructure_mesa_cap.sheet_densities)
plt.plot(heterostructure_no_mesa.V_gate_arr, heterostructure_no_mesa.sheet_densities)


# plt.yscale('log')
plt.xlabel('Gate voltage (V)')
plt.ylabel('2D charge density (cm$^{-2}$)')
plt.legend(['Si/SiGe quantum well', 
            'Si cap', 
            'SiGe outside mesa'], loc='upper left')
plt.grid(True, which='both', linestyle=':', color='#CCCCCC')
plt.minorticks_on()

plt.savefig(r"z:\Data\Plotting\Plots\poisson_schrodinger\mesa_vs_nomesa_vs_FET_accumulation_n=1e16_h=0.05_SiGe=200_noFET.pdf",
             format='pdf',
               dpi=300)

