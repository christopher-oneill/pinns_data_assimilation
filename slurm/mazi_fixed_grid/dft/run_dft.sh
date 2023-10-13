#!/bin/bash

cd /home/coneill/sync/

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft0_S1_j003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft0_S1_j004.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft1_S1_j003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft1_S1_j004.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft2_S1_j003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft2_S1_j004.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft3_S1_j003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft3_S1_j004.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft4_S1_j003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft4_S1_j004.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft5_S1_j003.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/dft/mfg_dft5_S1_j004.sh
