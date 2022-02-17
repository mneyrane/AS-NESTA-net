#!/bin/bash

#SBATCH --nodes=1 		        # number of nodes to use
#SBATCH --ntasks-per-node=4     # number of tasks
#SBATCH --exclusive             # run on whole node
#SBATCH --cpus-per-task=6       # There are 24 CPU cores on Cedar p100 GPU nodes
#SBATCH --gres=gpu:p100:4       # number of GPUs to use
#SBATCH --mem=0     	    	# memory per node (0 = use all of it)
#SBATCH --time=00:30:00         # time (DD-HH:MM)
#SBATCH --account=def-adcockb

module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch numpy scipy Pillow --no-index

cd $SLURM_TMPDIR
git clone https://github.com/mneyrane/NESTA-net.git

cd NESTA-net/pytorch

PROJECT_DIR="$HOME/projects/def-adcockb/mneyrane"
ARCHIVE_NAME="CC_stability_results.tar"

NUM_TRIALS=1
IM_PATH="../demo_images/brain_512.png"
MASK_PATH="../sampling_masks/tv_haar_mask_25.png"
ETA_VALUES=(1 0.1 0.01 0.001)
ETA_MULTS=(1 10 100 1000)

for ((t=1;t<=NUM_TRIALS;t++)); do

    for i in ${!ETA_VALUES[@]}; do

        for j in ${!ETA_MULTS[@]}; do
            CUDA_VISIBLE_DEVICES=$j python CC_stability.py --im-path $IM_PATH --mask-path $MASK_PATH --eta ${ETA_VALUES[$i]} --eta-p-mult ${ETA_MULTS[$j]} &
        done

        wait # for tasks to finish

    done

done

shopt -s extglob
tar -cf $ARCHIVE_NAME +([0-9])-+([0-9])
shopt -u extglob

mv $ARCHIVE_NAME $PROJECT_DIR
