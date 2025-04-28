import numpy as np
from MDAnalysis.analysis.distances import *

#RDF between particles of the same type i

def self_rdf(molecular_system,x,y,z,box_array,n_frames,i,rdf,edges,nbins,dmin,dmax,dist):

#expects x,y,z coordinates in the following format
#x=np.array((n_frames,num_particles,3))
#y=np.array((n_frames,num_particles,3))
#z=np.array((n_frames,num_particles,3))
#index i: loop over particle type i as in calc_RDF.py

    boxvolume=0

    for j in range(0,n_frames):
        coords=np.zeros((molecular_system["n_mol"][i],3))
        for k in range (0,molecular_system["n_mol"][i]):
            coords[k][0]=x[i][j][k][2]
            coords[k][1]=y[i][j][k][2]
            coords[k][2]=z[i][j][k][2]
        box_temp=[box_array[j][0],box_array[j][1],box_array[j][2], 90, 90 ,90]#mdanalysis requires unit cell angle        
        self_distance_array(coords, box_temp,result=dist)
        new_rdf, edges = np.histogram(dist, bins=nbins, range=(dmin, dmax))#histogram of current frame j
        rdf += new_rdf #update total histogram
        boxvolume+=(box_array[j][0]*box_array[1][0]*box_array[j][2])
    boxvolume /= n_frames  # average volume

# Normalize RDF
    radii = 0.5 * (edges[1:] + edges[:-1])
    vol = (4. / 3.) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
# normalization to the average density n/boxvolume in the simulation
    density = molecular_system["n_mol"][i]/ boxvolume
    norm = density  * n_frames
#n/2 because pairs  2-1and 1-2... are the same; only half the particles have to be considered
    rdf = rdf/(norm*vol*molecular_system["n_mol"][i]/2)
    
    return rdf,radii


#RDF between particles of different type i and j

def cross_rdf(molecular_system,x,y,z,box_array,n_frames,i,j,rdf,edges,nbins,dmin,dmax):

#expects x,y,z coordinates in the following format
#x=np.array((n_frames,num_particles,3))
#y=np.array((n_frames,num_particles,3))
#z=np.array((n_frames,num_particles,3))
#index i: loop over particle type i in calc_RDF.py
#index j: loop over 2nd particle type j in calc_RDF.py
    
    boxvolume=0
    
    for k in range(0,n_frames):
        coords_i=np.zeros((molecular_system["n_mol"][i],3))
        coords_j=np.zeros((molecular_system["n_mol"][j],3))
        for l in range (0,molecular_system["n_mol"][i]):
            coords_i[l][0]=x[i][k][l][2]
            coords_i[l][1]=y[i][k][l][2]
            coords_i[l][2]=z[i][k][l][2]
        for m in range (0,molecular_system["n_mol"][j]):
            coords_j[m][0]=x[j][k][m][2]
            coords_j[m][1]=y[j][k][m][2]
            coords_j[m][2]=z[j][k][m][2]

        box_temp=[box_array[j][0],box_array[j][1],box_array[j][2], 90, 90 ,90]#mdanalysis requires unit cell angle
        dist=distance_array(coords_i,coords_j, box_temp)
        new_rdf, edges = np.histogram(dist, bins=nbins, range=(dmin, dmax))#histogram of current frame k
        rdf += new_rdf #update total histogram
        boxvolume+=(box_array[j][0]*box_array[1][0]*box_array[j][2])
    boxvolume /= n_frames  # average volume

 # Normalize RDF
    radii = 0.5 * (edges[1:] + edges[:-1])
    vol = (4. / 3.) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
# normalization to the average density n/boxvolume in the simulation
    density = (molecular_system["n_mol"][j])/ boxvolume
    norm = density  * n_frames
    rdf = rdf/(norm*vol*molecular_system["n_mol"][i])

    return rdf, radii 
