#!/bin/bash

cd /home/coneill/sync/code/pinns_data_assimilation/slurm/mazi_fixed_grid/femb_dft/

sbatch S1/dft0_S1_j001.sh
sbatch S1/dft0_S1_j002.sh
sbatch S1/dft0_S1_j003.sh
sbatch S1/dft0_S1_j004.sh
sbatch S1/dft1_S1_j001.sh
sbatch S1/dft1_S1_j002.sh
sbatch S1/dft1_S1_j003.sh
sbatch S1/dft1_S1_j004.sh

sbatch S4/dft0_S4_j001.sh
sbatch S4/dft0_S4_j002.sh
sbatch S4/dft0_S4_j003.sh
sbatch S4/dft0_S4_j004.sh
sbatch S4/dft1_S4_j001.sh
sbatch S4/dft1_S4_j002.sh
sbatch S4/dft1_S4_j003.sh
sbatch S4/dft1_S4_j004.sh

sbatch S8/dft0_S8_j001.sh
sbatch S8/dft0_S8_j002.sh
sbatch S8/dft0_S8_j003.sh
sbatch S8/dft0_S8_j004.sh
sbatch S8/dft1_S8_j001.sh
sbatch S8/dft1_S8_j002.sh
sbatch S8/dft1_S8_j003.sh
sbatch S8/dft1_S8_j004.sh


