#!/bin/bash

cd /home/coneill/sync/


sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m001_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m002_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m003_L10N100.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m004_L10N100.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m001_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m002_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m003_L10N200.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/vdnn/S32/mfg_vdnn_S32_m004_L10N200.sh