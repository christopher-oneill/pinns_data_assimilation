#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --cpus-per-task=16
#SBATCH --mem=144G
#SBATCH --time=4-16:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/mfg_S32_m003-%A.out

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Run your simulation step here...
cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python /home/coneill/sync/code/pinns_galerkin_viv/training_scripts/mazi_fixed_grid/mfg_new_mean.py 3 32


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------


