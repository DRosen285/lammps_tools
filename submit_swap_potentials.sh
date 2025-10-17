#!/bin/bash

#SBATCH -J test_MACE_classical_swap
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-gpu=1        # cpu-cores per gpu (>1 if multi-threaded tasks)
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

#export OMP_NUM_THREADS=2
eval "$(micromamba shell hook --shell bash)"
unset LD_LIBRARY_PATH
micromamba activate mace_lammps_MLIAP

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/drosenbe/micromamba/envs/mace_lammps_MLIAP/lib:/home/drosenbe/programs/lammps_MLIAP_mace_installed/lib:/home/drosenbe/programs/lammps_mace_installed/lib:/home/drosenbe/programs/ACEsuit_lammps/libtorch/lib

export OMPI_MCA_opal_cuda_support=true


cd prod_run
   for i in $(seq 1 1);
   do	   
    if [ $i -eq 1 ]; then 
    cp ../energy_min_equib_first_classical/final_frame_classical.xyz .
    fi 
    #swap coordinates in MACE data File
    sed 's/XXX/'$i'/' ../swap_coordinates_lammpsio.py > tmp.py
    sed 's/MODEL/classical/' tmp.py >tmp2.py
    sed -E 's/(write_new_data\s*\([^)]*MACE=)False/\1True/' tmp2.py > tmp3.py
    python tmp3.py
    sed 's/XXX/'$i'/' ../lammps_settings/mixture_MACE.in > mixture_MACE_$i.in
    mpirun -np 1 /home/drosenbe/programs/lammps_MLIAP_mace_installed/bin/lmp  -k on g 1 -sf kk -pk kokkos newton on neigh half -in mixture_MACE_$i.in
    #extract final frame from MACE trajectory
    tac dump_EC_LiPF6_MACE_$i.lammpstrj | awk '/ITEM: TIMESTEP/ {exit} 1'| tac > final_frame_MACE.xyz
    #
    #swap coordinates in classical data_file
    sed 's/XXX/'$i'/' ../swap_coordinates_lammpsio.py > tmp.py
    sed 's/MODEL/MACE/' tmp.py > tmp2.py
    python tmp2.py
    sed 's/XXX/'$i'/' ../lammps_settings/mixture_classical.in > mixture_classical_$i.in
    mpirun -np 1 /home/drosenbe/programs/lammps_MLIAP_mace_installed/bin/lmp -k on g 1 -sf kk -in mixture_classical_$i.in
    tac dump_EC_LiPF6_classical_$i.lammpstrj | awk '/ITEM: TIMESTEP/ {exit} 1'| tac > final_frame_classical.xyz
    done
rm tmp*
cd ..   

