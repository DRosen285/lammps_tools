#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import lammpsio

#load trajectory in pandas dataframe

input_file ="/home/drosenbe/run_EC_LiPF6/reactive_system/classical/prod_run/frame_8781.xyz"
df=pd.read_csv(input_file, header=8,delimiter=r"\s+",engine='c', iterator=True)
df2=df.get_chunk()
df_modified = df2[df2.columns[:-2]]
df_modified.columns = df2.columns[2:]

#read in box coordinates

# Read the file line by line
box_lines = []
with open(input_file, 'r') as f:
    capture = False
    for line in f:
        line = line.strip()
        if line.startswith("ITEM: BOX BOUNDS"):
            capture = True
            continue  # skip header line
        if capture:
            if line.startswith("ITEM"):  # stop if next section starts
                break
            box_lines.append(line)

# Convert to floats
box_data = [list(map(float, line.split())) for line in box_lines]
# Create DataFrame
df_box = pd.DataFrame(box_data, columns=['min', 'max'], index=['x', 'y', 'z'])

#translate from df to numpy
x_inp=df_modified["x"].to_numpy()
y_inp=df_modified["y"].to_numpy()
z_inp=df_modified["z"].to_numpy()
min_box=df_box["min"].to_numpy()
max_box=df_box["max"].to_numpy()

print(max_box)

#create numpy array and put in correct shape to fit to MDAnalysis
pos_array=np.array((x_inp,y_inp,z_inp))
pos_array_swap=np.swapaxes(pos_array, 0,1)
pos_array_swap

#write new lammps data file with swapped coordinates
def write_new_data(pos_array_swap,MACE=False):
    #swapping from classical to MACE  
    if MACE:
       datafile = lammpsio.DataFile('/home/drosenbe/run_EC_LiPF6/reactive_system/MACE_input/mixture_MACE.data', atom_style='atomic')
       snapshot= datafile.read()
       snapshot.position=pos_array_swap
       new_snapshot= lammpsio.Snapshot(
                     snapshot.N,
                     box=lammpsio.Box(min_box, max_box)
                     )
       new_snapshot.id=snapshot.id
       new_snapshot.mass=snapshot.mass
       new_snapshot.molecule=snapshot.molecule
       new_snapshot.num_types=snapshot.num_types
       new_snapshot.position=snapshot.position
       new_snapshot.typeid=snapshot.typeid
       new_snapshot.type_label=snapshot.type_label
       lammpsio.DataFile.create("mixture_MACE_frame_8781.data",new_snapshot,atom_style='atomic')

    else: 
       #assumes trajectory comes from MACE
       datafile = lammpsio.DataFile('mixture.data', atom_style='full')
       snapshot= datafile.read()
       snapshot.position=pos_array_swap
       new_snapshot= lammpsio.Snapshot(
                     snapshot.N,
                     box=lammpsio.Box(min_box, max_box)
                     )
       new_snapshot.angles=snapshot.angles
       new_snapshot.bonds=snapshot.bonds
       new_snapshot.charge=snapshot.charge
       new_snapshot.dihedrals=snapshot.dihedrals
       new_snapshot.id=snapshot.id
       new_snapshot.impropers=snapshot.impropers
       new_snapshot.mass=snapshot.mass
       new_snapshot.molecule=snapshot.molecule
       new_snapshot.num_types=snapshot.num_types
       new_snapshot.position=snapshot.position
       new_snapshot.typeid=snapshot.typeid
       new_snapshot.type_label=snapshot.type_label
       lammpsio.DataFile.create('mixture_classical_after_500K_NPT.data',new_snapshot,atom_style='full')


write_new_data(pos_array_swap,MACE=True)
