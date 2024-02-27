#!/bin/bash

cd /home/coneill/sync/

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/adaptive/S4/mfg_ad2_m001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/adaptive/S4/mfg_ad2_m002.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/adaptive/S4/mfg_ad2_m003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/adaptive/S4/mfg_ad2_m004.sh