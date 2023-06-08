#!/bin/bash
#SBATCH --account=def-martinuz
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/mfgwS16f21_001-%A.out

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Run your simulation step here...
cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python /home/coneill/sync/code/pinns_galerkin_viv/training_scripts/mfgwS16/fourier21/mfgwS16_fourier21_001.py


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------


