#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=8G
#SBATCH --time=0:5:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-adcockb

# set up environment

module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch numpy Pillow --no-index

cd $SLURM_TMPDIR

git clone https://github.com/mneyrane/NESTA-net.git

cd NESTA-net/pytorch

python demo_NESTA_TV_Haar_stability.py

tar -cf stability_results.tar *.npy *.png

mv stability_results.tar $HOME/projects/def-adcockb/mneyrane
