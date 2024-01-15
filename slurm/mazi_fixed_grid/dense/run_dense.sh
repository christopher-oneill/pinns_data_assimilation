#!/bin/bash

cd /home/coneill/sync/code/pinns_data_assimilation/slurm/mazi_fixed_grid/dense/

sbatch d_lg_S1_m001.sh
sbatch d_lg_S1_m002.sh
sbatch d_lg_S1_m003.sh
sbatch d_lg_S16_m001.sh
sbatch d_lg_S16_m002.sh
sbatch d_lg_S16_m003.sh
sbatch d_lg_S32_m001.sh
sbatch d_lg_S32_m002.sh
sbatch d_lg_S32_m003.sh

sbatch d_sm_S1_m001.sh
sbatch d_sm_S1_m002.sh
sbatch d_sm_S1_m003.sh
sbatch d_sm_S16_m001.sh
sbatch d_sm_S16_m002.sh
sbatch d_sm_S16_m003.sh
sbatch d_sm_S32_m001.sh
sbatch d_sm_S32_m002.sh
sbatch d_sm_S32_m003.sh

cd /home/coneill/