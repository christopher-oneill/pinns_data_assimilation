

import sys


SLURM_TMPDIR='F:/projects/pinns_narval/sync/'
HOMEDIR = 'F:/projects/pinns_narval/sync/'
PROJECTDIR = HOMEDIR
sys.path.append('F:/projects/pinns_local/code/')

run_script_folder = 'F:/projects/pinns_local/code/pinns_data_assimilation/slurm/mazi_fixed_grid/test/'
case_scripts_folder = run_script_folder + 't10_f2/'

cluster_case_file_loc = "/home/coneill/sync/code/pinns_data_assimilation/slurm/mazi_fixed_grid/test/t10_f2/"

frequencies = [0,1,2,3,4,5]
supersample_factors = [0,2,4,8,16,32]

run_file = open(run_script_folder+'run_t10_f2.sh','w')
run_line1 = "#!/bin/bash\n\n"
run_file.write(run_line1)

case_line1 = """#!/bin/bash
#SBATCH --account=def-martinuz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=0-23:30
#SBATCH --mail-user=christopher.mark.oneill@gmail.com
#SBATCH --mail-type=ALL\n"""

case_line2 =  """# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Run your simulation step here...

module load python/3.10
source /home/coneill/projects/def-martinuz/coneill/venv/tf1/bin/activate\n\n"""

case_line3 = """# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------"""

for i in range(len(frequencies)):
    for j in range(len(supersample_factors)):
        case_file_name = 'mfg_t10f2_j1_f'+str(frequencies[i])+'_S'+str(supersample_factors[j])+'.sh'
        case_file = open(case_scripts_folder+case_file_name,'w')
        case_file.write(case_line1)
        case_file.write("#SBATCH --output=/home/coneill/sync/logs/mfg_t10f2_j1_f"+str(frequencies[i])+"_S"+str(supersample_factors[j])+"-%A.out")
        case_file.write("\n")
        case_file.write(case_line2)
        case_file.write("python /home/coneill/sync/code/pinns_data_assimilation/training_scripts/mazi_fixed_grid/test/mfg_t010_f002.py 1 "+str(frequencies[i])+" "+str(supersample_factors[j])+" 23\n")
        case_file.write(case_line3)
        case_file.close()

        
        run_file.write("sbatch "+cluster_case_file_loc+case_file_name+"\n")

run_file.close()