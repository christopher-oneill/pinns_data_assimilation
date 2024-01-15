#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=0-11:30
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/dense_small_S32_m002-%A.out

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Run your simulation step here...
cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python /home/coneill/sync/code/pinns_data_assimilation/training_scripts/mazi_fixed_grid/mfg_dense_mean_small.py 2 32 11


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

sbatch /home/coneill/sync/code/pinns_data_assimilation/slurm/mazi_fixed_grid/dense/d_sm_S32_m002.sh
