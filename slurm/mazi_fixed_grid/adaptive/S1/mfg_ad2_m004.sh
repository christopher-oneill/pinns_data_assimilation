#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=2-23:30
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/mfg_ad2_S1_m004-%A.out

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Run your simulation step here...
cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python /home/coneill/sync/code/pinns_galerkin_viv/training_scripts/mazi_fixed_grid/mfg_adaptive2_mean.py 4 1 10 50


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------


