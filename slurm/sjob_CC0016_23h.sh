#!/bin/bash
#SBATCH --account=def-martinuz    
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16000M
#SBATCH --time=0-23:00
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/coneill/sync/logs/slurm-%A.out
#SBATCH --array=1-3%1   # Run a 10-job array, one job at a time.
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run your simulation step here...
CASENAME="CC0016"

cd $SLURM_TMPDIR

mkdir $SLURM_TMPDIR/code/
cp /home/coneill/sync/code/pinns_galerkin_viv.tar $SLURM_TMPDIR/code/
tar -xf $SLURM_TMPDIR/code/pinns_galerkin_viv.tar -C $SLURM_TMPDIR/code/

module load python/3.10
source /home/coneill/sync/venv/tf1/bin/activate

python $SLURM_TMPDIR/code/pinns_galerkin_viv/training_scripts/mazi_fixed/cylinder_pinn_"$CASENAME"_23h.py


# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
