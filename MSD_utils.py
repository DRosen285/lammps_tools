import numba
from numba import jit

#function to compute_MSD
@jit(nopython=True)
def calc_msd(type_ID,n_fr,npar,x,y,z,cnt,dx,dy,dz):
    #outer loop over frames
    for t in range (0,n_fr):
    #middle loop over time origins: when starting in 1st frame, corrleation is calculated over whole trajectory
    #when starting in frame n-1: corrleation is performed only over 1 frame; problems for normalization
        dt=1
        while(dt+t<n_fr):
        #count frequency of time origins with interval dt
            cnt[type_ID][dt]+=1
        #inner loop over particles    
            for j in range (0,npar):
                dx[type_ID][dt] += (x[t+dt][j][2]-x[t][j][2])**2;
                dy[type_ID][dt] += (y[t+dt][j][2]-y[t][j][2])**2;
                dz[type_ID][dt] += (z[t+dt][j][2]-z[t][j][2])**2;
            dt+=1
    return (cnt[type_ID],dx[type_ID],dy[type_ID],dz[type_ID])
    #return (cnt,dy[j]) 



#compute MSD normalization for textbook method
@jit(nopython=True)
def calc_msd_norm(type_ID,n_fr,npar,cnt,md_time_step,n_steps,stride,dx,dy,dz,msd_total,time):
    for t in range(1,n_fr):
        msd_total[type_ID][t]=(dx[t]+dy[t]+dz[t])/(cnt[t]*npar)
        time[type_ID][t]=t*md_time_step*n_steps*stride #factor 20 corresponds to frequency with which trajector is written 
    return(msd_total[type_ID],time[type_ID])

