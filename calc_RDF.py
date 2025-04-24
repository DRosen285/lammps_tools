#!/usr/bin/env python

import os
import numpy as np
import lammps_reader
from MDAnalysis.analysis.distances import *
import math

#anything molecule specfic e.g. number of atoms per molecule, atom type ID in molecule etc. requires same order as in lammps data file
molecular_system = {
  "n_frames":50000, #number of frames in trajectory
   "n_part": 8004, #total number of particles
  "n_molecule_types": 3,#number of different molecules in trajectory: EC,PF6-, Li+
  "n_atoms_mol": [10,7,1], #number of atoms per molecule: 10 (EC), 7(PF6-), 1 (Li+)
   "n_mol":[750,63,63], #number of molecules for each type 
  "atom_types_mol": [[1,2,3],[4,5],[6]],#atom type IDs for atoms in molecule 
   "md_time_step": 0.001, #time step for MD integrator  #unit ps
   "md_out_steps": 10 # every 250th step is written out 
}

#stride trajectory for memory efficient reading
stride=10
#read trajectory
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
                if cnt ==2:#for PF6- (molecule type 3 (in python: 2) we take P as reference we  have to jump over the 6F- and Li+ cation before the next anion molecule starts
                    #print(j,cnt2,m_clean[j][cnt2][2],x_clean[j][cnt2][2])
                    if l == molecular_system["n_atoms_mol"][cnt]-1:
                        cnt2+=7#jump over F6 and Li+ ion between P
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

#Compute Radial distribution function

#RDFs between molecules of the same type
self_rdf = {f'rdf{i}': [] for i in range(molecular_system["n_molecule_types"])}
n_frames=int(molecular_system["n_frames"]/stride)
n_type=molecular_system["n_molecule_types"]
print(n_type)
nbins=100
for i in range (0,n_type):
    tmp=list(com_x)[i]#convert dictionary key to index
    tmp2=list(self_rdf)[i]#convert dictionary key to index
    print(i,tmp,tmp2)
    dmin, dmax = 0.0,  box_array[0][0]/2
    rdf, edges = np.histogram([0], bins=nbins, range=(dmin, dmax))
    rdf = np.array([0]*nbins)
    dist = np.zeros(int(molecular_system["n_mol"][i]* (molecular_system["n_mol"][i] - 1) / 2,))
    boxvolume=0
    for j in range(0,n_frames):
        com=np.zeros((molecular_system["n_mol"][i],3))
        for k in range (0,molecular_system["n_mol"][i]):
            com[k][0]=com_x[tmp][j][k][2]
            com[k][1]=com_y[tmp][j][k][2]
            com[k][2]=com_z[tmp][j][k][2]     
        box_temp=[box_array[j][0],box_array[j][1],box_array[j][2], 90, 90 ,90]#mdanalysis requires unit cell angle        
        self_distance_array(com, box_temp,result=dist)
        new_rdf, edges = np.histogram(dist, bins=nbins, range=(dmin, dmax))#histogram of current frame j
        rdf += new_rdf #update total histogram
        boxvolume+=(box_array[j][0]*box_array[1][0]*box_array[j][2])
    self_rdf[tmp2]=rdf   
    boxvolume /= n_frames  # average volume

# Normalize RDF
#print edges[1:], edges[:-1]
    radii = 0.5 * (edges[1:] + edges[:-1])
    vol = (4. / 3.) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
# normalization to the average density n/boxvolume in the simulation
    density = molecular_system["n_mol"][i]/ boxvolume
    norm = density  * n_frames
#n/2 because pairs  2-1and 1-2... are the same; only half the particles have to be considered
    self_rdf[tmp2] = self_rdf[tmp2]/(norm*vol*molecular_system["n_mol"][i]/2)

for i in range(0,n_type):
    print(i)
    tmp2=list(self_rdf)[i]
    myfile = 'self_rdf_MACE_merged_%s' % i
    myfile_r = 'r_MACE_merged_%s' % i
    np.save(myfile, self_rdf[tmp2])
    np.save(myfile_r, radii)


#RDFs between molecules of different type
n_frames=int(molecular_system["n_frames"]/stride)
n_type=molecular_system["n_molecule_types"]
n_pairs = math.comb(molecular_system["n_molecule_types"], 2)
cross_rdf = {f'rdf{i}': [] for i in range(n_pairs)}
pair_cnt=0 #index to specifc pair
nbins=100
for i in range (0,n_type-1):
    tmp=list(com_x)[i]#convert dictionary key to index
    for j in range (i+1,n_type):    
        print(i,j,pair_cnt)
        tmp2=list(cross_rdf)[pair_cnt]#convert dictionary key to index
        tmp3=list(com_x)[j]#convert dictionary key to index
        dmin, dmax = 0.0,  box_array[0][0]/2
        rdf, edges = np.histogram([0], bins=nbins, range=(dmin, dmax))
        rdf = np.array([0]*nbins)
        boxvolume=0
        for k in range(0,n_frames):
            com_i=np.zeros((molecular_system["n_mol"][i],3))
            com_j=np.zeros((molecular_system["n_mol"][j],3))
            for l in range (0,molecular_system["n_mol"][i]):
                com_i[l][0]=com_x[tmp][k][l][2]
                com_i[l][1]=com_y[tmp][k][l][2]
                com_i[l][2]=com_z[tmp][k][l][2]   
            for m in range (0,molecular_system["n_mol"][j]):
                com_j[m][0]=com_x[tmp3][j][m][2]
                com_j[m][1]=com_y[tmp3][j][m][2]
                com_j[m][2]=com_z[tmp3][j][m][2] 

            box_temp=[box_array[j][0],box_array[j][1],box_array[j][2], 90, 90 ,90]#mdanalysis requires unit cell angle        
            dist=distance_array(com_i,com_j, box_temp)
            new_rdf, edges = np.histogram(dist, bins=nbins, range=(dmin, dmax))#histogram of current frame k
            rdf += new_rdf #update total histogram
            boxvolume+=(box_array[j][0]*box_array[1][0]*box_array[j][2])
        #print(rdf)    
        cross_rdf[tmp2]=rdf
        boxvolume /= n_frames  # average volume

     # Normalize RDF
        #print edges[1:], edges[:-1]
        radii = 0.5 * (edges[1:] + edges[:-1])
        vol = (4. / 3.) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
# normalization to the average density n/boxvolume in the simulation
        density = (molecular_system["n_mol"][j])/ boxvolume
        norm = density  * n_frames
#n/2 because pairs  2-1and 1-2... are the same; only half the particles have to be considered
        cross_rdf[tmp2] = cross_rdf[tmp2]/(norm*vol*molecular_system["n_mol"][i])
        pair_cnt+=1



for i in range(0,n_pairs):
    myfile = 'cross_rdf_MACE_merged_%s' % i
    myfile_r = 'r_MACE_merged_%s' % i
    tmp2=list(cross_rdf)[i]
    np.save(myfile, cross_rdf[tmp2])
    np.save(myfile_r, radii)


