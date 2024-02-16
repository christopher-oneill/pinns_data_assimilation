#!/bin/bash
#SBATCH --account=def-martinuz
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0-23:30
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/mfd8_1_S16-%A.out

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Run your simulation step here...
cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python /home/coneill/sync/code/pinns_data_assimilation/training_scripts/mazi_fixed/mfg_dense008.py 1 16 23


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

sbatch /home/coneill/sync/code/pinns_data_assimilation/slurm/mazi_fixed_grid/dense/d_lg_S1_m001.sh
