#!/usr/bin/env python

import os
import numpy as np
import lammps_reader
import MSD_utils
import calc_COM

#anything molecule specfic e.g. number of atoms per molecule, atom type ID in molecule etc. requires same order as in lammps data file
molecular_system = {
   "n_frames":1, #number of frames in trajectory: will be updated after trajectory is read
   "n_part": 8004, #total number of particles
   "n_molecule_types": 3,#number of different molecules in trajectory: EC,PF6-, Li+
   "n_atoms_mol": [10,7,1], #number of atoms per molecule: 10 (EC), 7(PF6-), 1 (Li+)
   "n_mol":[750,63,63], #number of molecules for each type 
   "atom_types_mol": [[1,2,3],[4,5],[6]],#atom type IDs for atoms in molecule 
   "md_time_step": 1, #time step for MD integrator  #unit ps
   "md_out_steps": 1 # frequency at which MD trajectory is written 
}


#stride trajectory to make reading memory efficient
stride=10
#read_trajectory: single
#frames,ts,box_bounds=lammps_reader.read_lammps_trajectory_fast("../dump_EC_LiPF6_MACE_1.lammpstrj",stride=stride)

#read_trajectory: multiple
filenames =["../dump_EC_LiPF6_MACE_1.lammpstrj", "../dump_EC_LiPF6_classical_1.lammpstrj", "../dump_EC_LiPF6_MACE_2.lammpstrj","../dump_EC_LiPF6_classical_2.lammpstrj"]
fields = ['id','type','mass','xu','yu','zu']
frames,timesteps,box_bounds = lammps_reader.read_multiple_lammps_trajectories(filenames,fields_to_extract=fields,stride=stride)
molecular_system["n_frames"]=len(frames)

#convert read information to numpy arrays
t_array=np.zeros(int(molecular_system["n_frames"])) # store frame number
box_array=np.zeros((int(molecular_system["n_frames"]),3)) #contains side length of cubic box (assumes that in lammps box dimension starts at 0)
x_clean=np.zeros((int(molecular_system["n_frames"]),molecular_system["n_part"],3)) # store per frame particle#, particle type and x-coordinate
y_clean=np.zeros((int(molecular_system["n_frames"]),molecular_system["n_part"],3)) # store per frame particle#, particle type and y-coordinate
z_clean=np.zeros((int(molecular_system["n_frames"]),molecular_system["n_part"],3)) # store per frame particle#, particle type and z-coordinate
m_clean=np.zeros((int(molecular_system["n_frames"]),molecular_system["n_part"],3)) # store per frame particle#, particle type and mass

traj_size=int(molecular_system["n_frames"])
print(traj_size)
num_part=int(molecular_system["n_part"])
x_clean,y_clean,z_clean,m_clean,box_array,t_array=lammps_reader.convert_traj(frames,box_bounds,num_part,traj_size,x_clean,y_clean,z_clean,m_clean,box_array,t_array,unwrapped=True)

#compute center of mass coordinates for all molecules
n_frames=int(molecular_system["n_frames"])
#generate dictionary of lists depending on number of different molecule types
com_x = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
com_y = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
com_z = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
cnt=0#loop over number of molecule types
n_frames=int(molecular_system["n_frames"])
keys = list(com_x.keys()) # Create a list of keys to iterate over
for key in keys:
    tmp = int(key[-1])
    com_x[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_y[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_z[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_x[tmp],com_y[tmp],com_z[tmp]=calc_COM.com_molecule(molecular_system,x_clean,y_clean,z_clean,m_clean,box_array,cnt,tmp,n_frames,com_x,com_y,com_z)
    cnt+=1




#Computation of Mean Squared Displacement (MSD)
#compute MSD (see e.g. Frenkel and Smit Understanding Molecular Simulations)
n_type=molecular_system["n_molecule_types"]
n_frames=int(molecular_system["n_frames"])
md_time_step=molecular_system["md_time_step"]
n_steps=molecular_system["md_out_steps"]

dx=np.zeros((n_type,n_frames))
dy=np.zeros((n_type,n_frames))
dz=np.zeros((n_type,n_frames))
cnt=np.zeros((n_type,n_frames))
msd_total=np.zeros((n_type,n_frames))
time=np.zeros((n_type,n_frames))

for i in range(0,n_type):
    tmp=int(list(com_x)[i][-1])
    print(tmp)
    mol_type=i
    n_part=molecular_system["n_mol"][i] #as we consider center of mass coordinates: one molecule = one particle
    cnt[i],dx[i],dy[i],dz[i]=MSD_utils.calc_msd(mol_type,n_frames,n_part,com_x[tmp],com_y[tmp],com_z[tmp],cnt,dx,dy,dz)
    #compute msd: normalization based on time origins
    msd_total[i],time[i]= MSD_utils.calc_msd_norm(mol_type,n_frames,n_part,cnt[i],md_time_step,n_steps,stride,dx[i],dy[i],dz[i],msd_total,time) 

for i in range(0,n_type):
    tmp=list(com_x)[i]
    myfile = 'msd_MACE_merged_%s' % tmp
    myfile_t = 't_MACE_merged_%s' % tmp
    np.save(myfile, msd_total[i])
    np.save(myfile_t, time[i])


