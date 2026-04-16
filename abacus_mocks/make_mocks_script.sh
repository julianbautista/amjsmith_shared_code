#!/bin/bash
#SBATCH -t 00:40:00
#SBATCH --qos=shared
#SBATCH -A m4237
#SBATCH -C cpu
#SBATCH --job-name=make_base_mocks
#SBATCH --output=logs/make_abacus_base_mocks.out
#SBATCH --mem=70G
#SBATCH --array=2-10

module load python
conda activate abacus_env

srun python /global/homes/a/amsellem/amjsmith_shared_code/abacus_mocks/make_mocks_script.py ${SLURM_ARRAY_TASK_ID} 0
