#!/usr/bin/env python

import os
import numpy as np
import lammps_reader
import MSD_utils

#anything molecule specfic e.g. number of atoms per molecule, atom type ID in molecule etc. requires same order as in lammps data file
molecular_system = {
  "n_frames":50000, #number of frames in trajectory
   "n_part": 8004, #total number of particles
  "n_molecule_types": 3,#number of different molecules in trajectory: EC,PF6-, Li+
  "n_atoms_mol": [10,7,1], #number of atoms per molecule: 10 (EC), 7(PF6-), 1 (Li+)
   "n_mol":[750,63,63], #number of molecules for each type 
  "atom_types_mol": [[1,2,3],[4,5],[6]],#atom type IDs for atoms in molecule 
   "md_time_step": 0.001, #time step for MD integrator  #unit ps
   "md_out_steps": 10 # every 10th step is written out 
}


#stride trajectory to make reading memory efficient
stride=10
#read_trajectory
frames,ts,box_bounds=lammps_reader.read_lammps_trajectory_fast("../dump_EC_LiPF6_MACE_merged.lammpstrj",stride=stride)


t_array=np.zeros(int(molecular_system["n_frames"]/stride)) # store frame number
box_array=np.zeros((int(molecular_system["n_frames"]/stride),3)) #contains side length of cubic box (assumes that in lammps box dimension starts at 0)
x_clean=np.zeros((int(molecular_system["n_frames"]/stride),molecular_system["n_part"],3)) # store per frame particle#, particle type and x-coordinate
y_clean=np.zeros((int(molecular_system["n_frames"]/stride),molecular_system["n_part"],3)) # store per frame particle#, particle type and y-coordinate
z_clean=np.zeros((int(molecular_system["n_frames"]/stride),molecular_system["n_part"],3)) # store per frame particle#, particle type and z-coordinate
m_clean=np.zeros((int(molecular_system["n_frames"]/stride),molecular_system["n_part"],3)) # store per frame particle#, particle type and mass

for i in range(0,int(molecular_system["n_frames"]/stride)):
        for j in range(0,molecular_system["n_part"]):
        #print(p_x[z][0],p_x[z][1],p_x[z][2],p_y[z][2],p_z[z][2])
             x_clean[i][j][0]=frames[i][j]["id"]#assign particle #        
             x_clean[i][j][1]=frames[i][j]["type"]#assign particle type
             x_clean[i][j][2]=frames[i][j]["x"]#assign coordinate  

             y_clean[i][j][0]=frames[i][j]["id"]#assign particle #        
             y_clean[i][j][1]=frames[i][j]["type"]#assign particle type
             y_clean[i][j][2]=frames[i][j]["y"]#assign coordinate    

             z_clean[i][j][0]=frames[i][j]["id"]#assign particle #        
             z_clean[i][j][1]=frames[i][j]["type"]#assign particle type
             z_clean[i][j][2]=frames[i][j]["z"]#assign coordinate   

             m_clean[i][j][0]=frames[i][j]["id"]#assign particle #      
             m_clean[i][j][1]=frames[i][j]["type"]#assign particle type
             m_clean[i][j][2]=frames[i][j]["mass"]#assign mass

        box_array[i][0]= np.sqrt((box_bounds[i][0][1]-box_bounds[0][0][0])**2)
        box_array[i][1]= np.sqrt((box_bounds[i][1][1]-box_bounds[0][1][0])**2)
        box_array[i][2]= np.sqrt((box_bounds[i][2][1]-box_bounds[0][2][0])**2)

        t_array[i]=i

#cpmpute center of mass coordinates for all molecules
#generate dictionary of lists depending on number of different molecule types
com_x = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
com_y = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}
com_z = {f'arr{i}': [] for i in range(molecular_system["n_molecule_types"])}

cnt=0#loop over number of molecule types
n_frames=int(molecular_system["n_frames"]/stride)

for i in com_x:
    tmp=list(com_x)[cnt]
    com_x[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_y[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))
    com_z[tmp]=np.zeros((n_frames,molecular_system["n_mol"][cnt],3))

    for j in range(0,n_frames):

        #cnt2:assign starting index of first atom for a given molecule type per frame (cnt==molecule_type ID)
        cnt2=0#initial index of first atom of molecule 1 
        if cnt!=0 and cnt <2:
            temp=((molecular_system["n_mol"][m]*molecular_system["n_atoms_mol"][m]) for m in range (0,cnt))#total number of particles in molecules 0
            cnt2=np.sum(list(temp))
        if cnt ==2:
           temp=((molecular_system["n_mol"][m]*molecular_system["n_atoms_mol"][m]) for m in range (0,cnt-1))#total number of particles in molecules 0,1
           cnt2=np.sum(list(temp))+molecular_system["n_atoms_mol"][cnt-1]#compute index of first Li ion per frame by adding number of atoms/anion

        for k in range (0, molecular_system["n_mol"][cnt]):#loop over number of molecules for a given type 
            com_x_temp=0 
            com_y_temp=0 
            com_z_temp=0 
            tot_mass_temp=0 
            for l in range (0,molecular_system["n_atoms_mol"][cnt]):#loop over number of atoms of one molecule of a given type
                if l >0:
                #check_for pbc: molecules can sit in different images, but com should be split over multiple images
                   x_clean[j][cnt2][2],y_clean[j][cnt2][2],z_clean[j][cnt2][2]=lammps_reader.check_pbc(x_clean[j][cnt2-1][2],y_clean[j][cnt2-1][2],z_clean[j][cnt2-1][2],x_clean[j][cnt2][2],y_clean[j][cnt2][2],z_clean[j][cnt2][2],box_array[j])                
                com_x_temp+=m_clean[j][cnt2][2]*x_clean[j][cnt2][2] 
                com_y_temp+=m_clean[j][cnt2][2]*y_clean[j][cnt2][2] 
                com_z_temp+=m_clean[j][cnt2][2]*z_clean[j][cnt2][2] 
                tot_mass_temp+=m_clean[j][cnt2][2]  
                if cnt ==2:#for PF6- (molecule type 3 (in python: 2) we have to jump over the Li+ cation before the next anion molecule starts
                    #print(j,cnt2,m_clean[j][cnt2][2],x_clean[j][cnt2][2])
                    if l == molecular_system["n_atoms_mol"][cnt]-1:
                        cnt2+=1#jump over Li+ ion between PF6-
                    cnt2+=1    
                elif cnt==3:#for Li+ (molecule type 4 (in python: 3) we have to jump over the Pf6-(7atoms) anion before the next cation molecule starts
                    #print(j,cnt2,m_clean[j][cnt2][2],x_clean[j][cnt2][2])
                    cnt2+=molecular_system["n_atoms_mol"][cnt-1]+1#jump over PF6-
                else:
                    #if cnt==0:
                    #   print(j,m_clean[j][cnt2][2],x_clean[j][cnt2][2])
                    cnt2+=1 

            #print(cnt,j,k,com_x_temp/tot_mass_temp)    
            com_x[tmp][j][k][0]=j#store frame #
            com_x[tmp][j][k][1]=k #store molecule # in dim=1    
            com_x[tmp][j][k][2]=com_x_temp/tot_mass_temp #store com x coordinate in dim=2 

            com_y[tmp][j][k][0]=j#store frame #
            com_y[tmp][j][k][1]=k #store molecule # in dim=1    
            com_y[tmp][j][k][2]=com_y_temp/tot_mass_temp #store com y coordinate in dim=2 

            com_z[tmp][j][k][0]=j#store frame #
            com_z[tmp][j][k][1]=k #store molecule # in dim=1    
            com_z[tmp][j][k][2]=com_z_temp/tot_mass_temp #store com z coordinate in dim=2 

    cnt+=1



#Computation of Mean Squared Displacement (MSD)
#compute MSD (see e.g. Frenkel and Smit Understanding Molecular Simulations)
n_type=molecular_system["n_molecule_types"]
n_frames=int(molecular_system["n_frames"]/stride)
md_time_step=molecular_system["md_time_step"]
n_steps=molecular_system["md_out_steps"]

dx=np.zeros((n_type,n_frames))
dy=np.zeros((n_type,n_frames))
dz=np.zeros((n_type,n_frames))
cnt=np.zeros((n_type,n_frames))
msd_total=np.zeros((n_type,n_frames))
time=np.zeros((n_type,n_frames))

for i in range(0,n_type):
    tmp=list(com_x)[i]
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


