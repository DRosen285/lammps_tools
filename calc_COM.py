#compute center of mass of molecules
import numpy as np
import lammps_reader

def com_molecule(molecular_system, x_clean,y_clean,z_clean,m_clean,box_array,cnt,tmp,n_frames,com_x,com_y,com_z): 
    
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
                    x_clean[j][cnt2][2],y_clean[j][cnt2][2],z_clean[j][cnt2][2]=lammps_reader.check_pbc(x_clean[j][cnt2-l][2],y_clean[j][cnt2-l][2],z_clean[j][cnt2-l][2],x_clean[j][cnt2][2],y_clean[j][cnt2][2],z_clean[j][cnt2][2],box_array[j])
                com_x_temp+=m_clean[j][cnt2][2]*x_clean[j][cnt2][2]
                com_y_temp+=m_clean[j][cnt2][2]*y_clean[j][cnt2][2]
                com_z_temp+=m_clean[j][cnt2][2]*z_clean[j][cnt2][2]
                tot_mass_temp+=m_clean[j][cnt2][2]

                if cnt ==1:#for PF6- (molecule type 3 (in python: 2) we take P as reference we  have to jump over the 6F- and Li+ cation before the next anion molecule starts
                    #print(j,cnt2,m_clean[j][cnt2][2],x_clean[j][cnt2][2])
                    if l == molecular_system["n_atoms_mol"][cnt]-1:
                        cnt2+=1#jump over Li+ ion between PF6-
                    cnt2+=1
                elif cnt==2:#for Li+ (molecule type 4 (in python: 3) we have to jump over the Pf6-(7atoms) anion before the next cation molecule starts
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
 
    return (com_x[tmp],com_y[tmp],com_z[tmp])
