#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/slurm-%A.out

cd $SLURM_TMPDIR

mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python $SLURM_TMPDIR/code/pinns_galerkin_viv/training_scripts/riches_uStar7_50/riches_uStar7_50_mean003.py

