#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/slurm-%A.out

cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python /home/coneill/sync/code/pinns_galerkin_viv/training_scripts/riches_uStar7_50/riches_uStar7_50_mean002.py

