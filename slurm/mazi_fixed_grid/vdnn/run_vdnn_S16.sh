#!/bin/bash

cd /home/coneill/sync/


sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m005_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m006_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m007_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m008_L10N100.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m005_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m006_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m007_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S16/mfg_vdnn_S16_m008_L10N200.sh