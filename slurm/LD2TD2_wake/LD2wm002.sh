#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/slurm-LD2wm002-%A.out

cd $SLURM_TMPDIR

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python  /home/coneill/sync/code/pinns_galerkin_viv/training_scripts/LD2TD2_wake/LD2TD2_wake_mean002.py

