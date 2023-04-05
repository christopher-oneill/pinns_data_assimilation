
# salloc --account=def-martinuz --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00

cd $SLURM_TMPDIR

mkdir ./data/
cp /home/coneill/sync/data/mazi_fixed.tar $SLURM_TMPDIR/data/
tar -xf $SLURM_TMPDIR/data/mazi_fixed.tar -C ./data/

mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/

source /home/coneill/sync/venv/tf1/bin/activate

mkdir ./output

python ./code/pinns_galerkin_viv/example_problems/cylinder_pinn_RS3_CC0001_23h.py

tar -cf output.tar output
cp output.tar /home/coneill/sync/output/

