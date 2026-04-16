#!/bin/bash 
#SBATCH -t 00:15:00
#SBATCH --qos=shared
#SBATCH -A m4237
#SBATCH -C cpu
#SBATCH --job-name=apply_footprint
#SBATCH --output=logs/apply_footprint-%A_%a.out
#SBATCH --mem=20G
#SBATCH --array=1-10

module load python
conda activate basic_311
source /global/common/software/desi/desi_environment.sh main

# Format the array number to two digits
array_number=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

srun python /global/homes/a/amsellem/amjsmith_shared_code/abacus_mocks/apply_desi_footprint_hdf5.py /global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/v0.6/iron/BGS_PV_AbacusSummit_base_c000_ph000_r0${array_number}_z0.11.dat.hdf5

