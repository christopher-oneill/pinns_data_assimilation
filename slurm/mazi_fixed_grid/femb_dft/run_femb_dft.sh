#!/bin/bash

cd /home/coneill/sync/

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft0_S1_j001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft0_S1_j002.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft1_S1_j001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft1_S1_j002.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft2_S1_j001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft2_S1_j002.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft3_S1_j001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft3_S1_j002.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft4_S1_j001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft4_S1_j002.sh

sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft5_S1_j001.sh
sbatch code/pinns_galerkin_viv/slurm/mazi_fixed_grid/femb_dft/mfg_femb_dft5_S1_j002.sh
