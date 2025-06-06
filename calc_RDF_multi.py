#!/usr/bin/env python

import os
import numpy as np
import lammps_reader
import calc_COM
import RDF_utils
import math

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

t_array=np.zeros(molecular_system["n_frames"]) # store frame number
box_array=np.zeros((molecular_system["n_frames"],3)) #contains side length of cubic box (assumes that in lammps box dimension starts at 0)
x_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and x-coordinate
y_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and y-coordinate
z_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and z-coordinate
m_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and mass

traj_size=molecular_system["n_frames"]
print(traj_size)
num_part=int(molecular_system["n_part"])
x_clean,y_clean,z_clean,m_clean,box_array,t_array=lammps_reader.convert_traj(frames,box_bounds,num_part,traj_size,x_clean,y_clean,z_clean,m_clean,box_array,t_array,unwrapped=True)

#compute center of mass coordinates for all molecules
n_frames=molecular_system["n_frames"]
#generate dictionary of lists depending on number of different molecule types
com_x = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
com_y = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
com_z = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
cnt=0#loop over number of molecule types
n_frames=molecular_system["n_frames"]
keys = list(com_x.keys()) # Create a list of keys to iterate over
for key in keys:
    tmp = int(key[-1])
    com_x[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_y[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_z[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_x[tmp],com_y[tmp],com_z[tmp]=calc_COM.com_molecule(molecular_system,x_clean,y_clean,z_clean,m_clean,box_array,cnt,tmp,n_frames,com_x,com_y,com_z)
    cnt+=1


#Compute Radial distribution function
n_frames=molecular_system["n_frames"]
#RDFs between molecules of the same type
self_rdf = {f'rdf{i}': [] for i in range(molecular_system["n_molecule_types"])}
n_type=molecular_system["n_molecule_types"]
print(n_type,n_frames)
nbins=100
for i in range (0,n_type):
    tmp2=list(self_rdf)[i]#convert dictionary key to index
    print(i,tmp,tmp2)
    dmin, dmax = 0.0,  box_array[0][0]/2
    rdf, edges = np.histogram([0], bins=nbins, range=(dmin, dmax))
    rdf = np.array([0]*nbins)
    dist = np.zeros(int(molecular_system["n_mol"][i]* (molecular_system["n_mol"][i] - 1) / 2,))
    self_rdf[tmp2],radii = RDF_utils.self_rdf(molecular_system,com_x,com_y,com_z,box_array,n_frames,i,rdf,edges,nbins,dmin,dmax,dist)

for i in range(0,n_type):
    print(i)
    tmp2=list(self_rdf)[i]
    myfile = 'self_rdf_MACE_merged_%s' % i
    myfile_r = 'r_MACE_merged_%s' % i
    np.save(myfile, self_rdf[tmp2])
    np.save(myfile_r, radii)


#RDFs between molecules of different type
n_frames=molecular_system["n_frames"]
n_type=molecular_system["n_molecule_types"]
n_pairs = math.comb(molecular_system["n_molecule_types"], 2)
cross_rdf = {f'rdf{i}': [] for i in range(n_pairs)}
pair_cnt=0 #index to specifc pair
nbins=100
for i in range (0,n_type-1):
    #tmp=int(list(com_x)[i][-1])#convert dictionary key to index
    for j in range (i+1,n_type):    
        print(i,j,pair_cnt)
        tmp2=list(cross_rdf)[pair_cnt]#convert dictionary key to index
    #    tmp3=int(list(com_x)[j][-1])#convert dictionary key to index
        dmin, dmax = 0.0,  box_array[0][0]/2
        rdf, edges = np.histogram([0], bins=nbins, range=(dmin, dmax))
        rdf = np.array([0]*nbins)
        cross_rdf[tmp2],radii = RDF_utils.cross_rdf(molecular_system,com_x,com_y,com_z,box_array,n_frames,i,j,rdf,edges,nbins,dmin,dmax)
        pair_cnt+=1

for i in range(0,n_pairs):
    myfile = 'cross_rdf_MACE_merged_%s' % i
    myfile_r = 'r_MACE_merged_%s' % i
    tmp2=list(cross_rdf)[i]
    np.save(myfile, cross_rdf[tmp2])
    np.save(myfile_r, radii)


