#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000M
#SBATCH --time=0-00:25
#SBATCH --mail-user=christopher.oneill@ucalgary.ca
#SBATCH --mail-type=ALL



CASENAME="RS3_CC0001_23h"

cd $SLURM_TMPDIR

mkdir $SLURM_TMPDIR/data/
cp /home/coneill/sync/data/mazi_fixed.tar $SLURM_TMPDIR/data/
tar -xf $SLURM_TMPDIR/data/mazi_fixed.tar -C $SLURM_TMPDIR/data/


mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/
module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

mkdir $SLURM_TMPDIR/output/
cp /home/coneill/sync/output/"$CASENAME"_output.tar $SLURM_TMPDIR/output/
tar -xf $SLURM_TMPDIR/output/"$CASENAME"_output.tar -C $SLURM_TMPDIR/output/


python $SLURM_TMPDIR/code/pinns_galerkin_viv/example_problems/cylinder_pinn_"$CASENAME".py


cd $SLURM_TMPDIR/output/
tar -cf ./"$CASENAME"_output.tar ./"$CASENAME"_output/
cd $SLURM_TMPDIR
cp $SLURM_TMPDIR/output/"$CASENAME"_output.tar /home/coneill/sync/output/

