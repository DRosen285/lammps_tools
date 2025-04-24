#!/bin/bash

#SBATCH -J test_MACE_classical_swap
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-gpu=1        # cpu-cores per gpu (>1 if multi-threaded tasks)
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

#export OMP_NUM_THREADS=2
eval "$(micromamba shell hook --shell bash)"
micromamba activate mace_lammps
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/drosenbe/programs/lammps_mace_installed/lib:/home/drosenbe/programs/ACEsuit_lammps/libtorch/lib

cd prod_run
   for i in $(seq 1 2);
   do	   
    if [ $i -eq 1 ]; then 
    cp ../energy_min_equib_first_classical/final_frame_classical.xyz .
    fi 
    #swap coordinates in MACE data File
    sed 's/XXX/'$i'/' ../swap_coordinates_lammpsio.py > tmp.py
    sed 's/MODEL/classical/' tmp.py >tmp2.py
    sed 's/False/True/' tmp2.py > tmp3.py
    python tmp3.py
    sed 's/XXX/'$i'/' ../lammps_settings/mixture_MACE_run.in > mixture_MACE_$i.in
    mpirun -np 1 ~/programs/ACEsuit_lammps/lammps/build/lmp -k on g 1 -sf kk -in mixture_MACE_$i.in
    #extract final frame from MACE trajectory
    tac dump_EC_LiPF6_MACE_$i.lammpstrj | awk '/ITEM: TIMESTEP/ {exit} 1'| tac > final_frame_MACE.xyz
    #
    #swap coordinates in classical data_file
    sed 's/XXX/'$i'/' ../swap_coordinates_lammpsio.py > tmp.py
    sed 's/MODEL/MACE/' tmp.py > tmp2.py
    python tmp2.py
    sed 's/XXX/'$i'/' ../lammps_settings/mixture_classical_run.in > mixture_classical_$i.in
    mpirun -np 1 ~/programs/ACEsuit_lammps/lammps/build/lmp -k on g 1 -sf kk -in mixture_classical_$i.in
    tac dump_EC_LiPF6_classical_$i.lammpstrj | awk '/ITEM: TIMESTEP/ {exit} 1'| tac > final_frame_classical.xyz
    done

