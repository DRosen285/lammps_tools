#!/usr/bin/env python

import os
import numpy as np
import lammps_reader
import calc_COM
import RDF_utils
import math

#anything molecule specfic e.g. number of atoms per molecule, atom type ID in molecule etc. requires same order as in lammps data file
molecular_system = {
  "n_frames":1, #number of frames in trajectory
   "n_part": 5264, #total number of particles
  "n_molecule_types": 3,#number of different molecules in trajectory: EC,PF6-, Li+
  "n_atoms_mol": [10,7,1], #number of atoms per molecule: 10 (EC), 7(PF6-), 1 (Li+)
   "n_mol":[500,33,33], #number of molecules for each type 
  "atom_types_mol": [[1,2,3],[4,5],[6]],#atom type IDs for atoms in molecule 
   "md_time_step": 0.001, #time step for MD integrator  #unit ps
   "md_out_steps": 1000 #  frequency output is written to trajectory file 
}


#stride trajectory to make reading memory efficient
stride=1
#read_trajectory: single
frames,ts,box_bounds=lammps_reader.read_lammps_trajectory_fast("dump_EC_w_mass.lammpstrj",stride=stride)

#assign number of frames based on actual trajectory size
molecular_system["n_frames"]=len(frames)

t_array=np.zeros(molecular_system["n_frames"]) # store frame number
box_array=np.zeros((molecular_system["n_frames"],3)) #contains side length of cubic box (assumes that in lammps box dimension starts at 0)
x_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and x-coordinate
y_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and y-coordinate
z_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and z-coordinate
m_clean=np.zeros((molecular_system["n_frames"],molecular_system["n_part"],3)) # store per frame particle#, particle type and mass


traj_size=molecular_system["n_frames"]
print(traj_size)
num_part=molecular_system["n_part"]
x_clean,y_clean,z_clean,m_clean,box_array,t_array=lammps_reader.convert_traj(frames,box_bounds,num_part,traj_size,x_clean,y_clean,z_clean,m_clean,box_array,t_array)

n_frames=molecular_system["n_frames"]

#generate dictionary of lists depending on number of different atom types
# Atom types you want RDFs for
atom_types_interest = [4, 5, 6]

# --- Create empty dictionaries with the same naming pattern as your COM arrays
rdf_x = {f'arr{i}': [] for i in range(len(atom_types_interest))}
rdf_y = {f'arr{i}': [] for i in range(len(atom_types_interest))}
rdf_z = {f'arr{i}': [] for i in range(len(atom_types_interest))}

# --- Fill the dictionaries with position arrays per atom type
cnt = 0
keys = list(rdf_x.keys())

for key in keys:
    tmp = int(key[-1])  # extract numeric suffix (0, 1, 2)
    atom_type = atom_types_interest[tmp]

    # number of atoms of this type (assume same across all frames)
    n_atoms_type = np.sum(x_clean[0, :, 1] == atom_type)
    print(n_atoms_type)
    # initialize arrays like your COM arrays
    rdf_x[tmp] = np.zeros((n_frames, n_atoms_type, 3))
    rdf_y[tmp] = np.zeros((n_frames, n_atoms_type, 3))
    rdf_z[tmp] = np.zeros((n_frames, n_atoms_type, 3))

    # fill arrays frame by frame
    for i in range(n_frames):
        mask = x_clean[i, :, 1] == atom_type
        rdf_x[tmp][i, :, :] = x_clean[i, mask, :]  # [id, type, x]
        rdf_y[tmp][i, :, :] = y_clean[i, mask, :]  # [id, type, y]
        rdf_z[tmp][i, :, :] = z_clean[i, mask, :]  # [id, type, z]

    cnt += 1

#Compute Radial distribution function
n_frames=molecular_system["n_frames"]
#RDFs between molecules of the same type
self_rdf = {f'rdf{i}': [] for i in range(molecular_system["n_molecule_types"])}
n_type=len(atom_types_interest)
print(n_type,n_frames)
nbins=100
for i in range (0,n_type):
    tmp2=list(self_rdf)[i]#convert dictionary key to index
    print(i,tmp2)
    dmin, dmax = 0.0,  box_array[0][0]/2
    rdf, edges = np.histogram([0], bins=nbins, range=(dmin, dmax))
    rdf = np.array([0]*nbins)
    atom_type = atom_types_interest[i]
    # number of atoms of this type (assume same across all frames)
    n_atoms_type = np.sum(x_clean[0, :, 1] == atom_type)
    dist = np.zeros(int(n_atoms_type* (n_atoms_type - 1) / 2,))
    self_rdf[tmp2],radii = RDF_utils.self_rdf(molecular_system,rdf_x,
                                              rdf_y,
                                              rdf_z,
                                              box_array,
                                              n_frames,
                                              i,
                                              rdf,
                                              edges,
                                              nbins,
                                              dmin,
                                              dmax,
                                              dist,
                                              atom=True,
                                               **({"atom_types_interest": atom_types_interest} if True else {}))

for i in range(0,n_type):
    print(i)
    tmp2=list(self_rdf)[i]
    myfile = 'self_rdf_MACE_AT_%s' % i
    myfile_r = 'r_MACE_AT_%s' % i
    np.save(myfile, self_rdf[tmp2])
    np.save(myfile_r, radii)


#RDFs between molecules of different type
n_frames=molecular_system["n_frames"]
n_type=len(atom_types_interest)
n_pairs = math.comb(n_type, 2)
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
        cross_rdf[tmp2],radii = RDF_utils.cross_rdf(molecular_system,rdf_x,rdf_y,rdf_z,box_array,n_frames,i,j,rdf,edges,nbins,dmin,dmax,atom=True,
                                                    **({"atom_types_interest": atom_types_interest} if True else {}))
        pair_cnt+=1

for i in range(0,n_pairs):
    myfile = 'cross_rdf_MACE_AT_%s' % i
    myfile_r = 'r_MACE_AT_%s' % i
    tmp2=list(cross_rdf)[i]
    np.save(myfile, cross_rdf[tmp2])
    np.save(myfile_r, radii)


