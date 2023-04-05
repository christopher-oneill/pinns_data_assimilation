
# salloc --account=def-martinuz --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00

CASENAME="RS3_CC0001_23h"

cd $SLURM_TMPDIR

mkdir ./data/
cp /home/coneill/sync/data/mazi_fixed.tar $SLURM_TMPDIR/data/
tar -xf $SLURM_TMPDIR/data/mazi_fixed.tar -C ./data/

mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

mkdir ./output
mkdir ./output/"$CASENAME"_output

python ./code/pinns_galerkin_viv/example_problems/cylinder_pinn_"$CASENAME".py

tar -cf ./output/"$CASENAME"_output.tar ./output/"$CASENAME"_output
cp ./output/"$CASENAME"_output.tar /home/coneill/sync/output/

