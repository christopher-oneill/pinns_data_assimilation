#!/bin/bash

cd /home/coneill/sync/code/pinns_data_assimilation/slurm/mazi_fixed_grid/qres/

sbatch S1_m001.sh
sbatch S1_m002.sh
sbatch S1_m003.sh
sbatch S16_m001.sh
sbatch S16_m002.sh
sbatch S16_m003.sh
sbatch S32_m001.sh
sbatch S32_m002.sh
sbatch S32_m003.sh

cd /home/coneill/