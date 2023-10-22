#!/bin/bash

cd /home/coneill/sync/


sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m005_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m006_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m007_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m008_L10N100.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m005_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m006_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m007_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S8/mfg_vdnn_S8_m008_L10N200.sh