#!/bin/bash -x
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=168:00:00
#PBS -P gca50115
#PBS -N TiO2_SiO2_H2

cd $PBS_O_WORKDIR

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gennip1.2.2

# source /etc/profile.d/modules.sh
# module load hpcx/2.20

export OMP_NUM_THREADS=1

lmp -in lammps.in &> run_md.sh.log
# mpirun -np 1 -bind-to none lmp -in lammps.in &> run_md.sh.log
