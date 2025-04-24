#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import lammpsio

#load trajectory in pandas dataframe

input_file ="final_frame_MODEL.xyz"
df=pd.read_csv(input_file, header=7,delimiter=r"\s+",engine='c', iterator=True)
df2=df.get_chunk()
df_modified = df2[df2.columns[:-2]]
df_modified.columns = df2.columns[2:]
df_modified


#translate from df to numpy

x_inp=df_modified["x"].to_numpy()
y_inp=df_modified["y"].to_numpy()
z_inp=df_modified["z"].to_numpy()


#create numpy array and put in correct shape to fit to MDAnalysis
pos_array=np.array((x_inp,y_inp,z_inp))
pos_array_swap=np.swapaxes(pos_array, 0,1)
pos_array_swap

#write new lammps data file with swapped coordinates
def write_new_data(pos_array_swap,MACE=False):
    #swapping from classical to MACE  
    if MACE:
       datafile = lammpsio.DataFile('/home/drosenbe/run_EC_LiPF6/swap_potentials/lammps_settings/mixture_MACE.data', atom_style='atomic')
       snapshot= datafile.read()
       snapshot.position=pos_array_swap
       lammpsio.DataFile.create("mixture_MACE_XXX.data",snapshot)
    else: 
       #assumes trajectory comes from MACE
       datafile = lammpsio.DataFile('/home/drosenbe/run_EC_LiPF6/swap_potentials/lammps_settings/mixture_classical.data', atom_style='full')
       snapshot= datafile.read()
       snapshot.position=pos_array_swap
       lammpsio.DataFile.create('mixture_classical_XXX.data',snapshot)


write_new_data(pos_array_swap,MACE=False)
