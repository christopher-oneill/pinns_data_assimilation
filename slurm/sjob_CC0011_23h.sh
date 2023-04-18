#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/slurm-%A.out

CASENAME="CC0011"

cd $SLURM_TMPDIR

mkdir $SLURM_TMPDIR/data/
cp /home/coneill/sync/data/mazi_fixed_modes.tar $SLURM_TMPDIR/data/
tar -xf $SLURM_TMPDIR/data/mazi_fixed_modes.tar -C $SLURM_TMPDIR/data/


mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

mkdir $SLURM_TMPDIR/output/
mkdir $SLURM_TMPDIR/output/"$CASENAME"_output

python $SLURM_TMPDIR/code/pinns_galerkin_viv/training_scripts/mazi_fixed/cylinder_pinn_"$CASENAME"_23h.py


DATE_STR=$(date '+%Y%m%d_%H%M%S')
cd $SLURM_TMPDIR/output/
tar -cf ./"$CASENAME"_output.tar ./"$CASENAME"_output/
tar -cf ./"$DATE_STR"_"$CASENAME"_output.tar ./"$CASENAME"_output/
cd $SLURM_TMPDIR
cp $SLURM_TMPDIR/output/"$CASENAME"_output.tar /home/coneill/sync/output/
cp $SLURM_TMPDIR/output/"$DATE_STR"_"$CASENAME"_output.tar /home/coneill/sync/output/

