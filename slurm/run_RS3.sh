
# salloc --account=def-martinuz --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00

CASENAME="CC0001_23h"

cd $SLURM_TMPDIR

# source data
mkdir ./data/
cp /home/coneill/sync/data/mazi_fixed.tar $SLURM_TMPDIR/data/
tar -xf $SLURM_TMPDIR/data/mazi_fixed.tar -C ./data/

# environment
mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate


# checkpoints/output
cp /home/coneill/sync/output/"$CASENAME"_output.tar $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/"$CASENAME"_output.tar 
rm $SLURM_TMPDIR/"$CASENAME"_output.tar 

# run
python ./code/pinns_galerkin_viv/example_problems/cylinder_pinn_RS3_"$CASENAME".py

# export
tar -cf "$CASENAME"_output.tar "$CASENAME"_output
cp "$CASENAME"_output.tar /home/coneill/sync/output/

