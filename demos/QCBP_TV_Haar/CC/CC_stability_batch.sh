#!/bin/bash

#SBATCH --array=1-100%10 	# job array (limit up to 10)
#SBATCH --nodes=1 		# number of nodes to use
#SBATCH --ntasks-per-node=4     # number of tasks
#SBATCH --cpus-per-task=2       # number of CPU cores per task
#SBATCH --gres=gpu:4            # number of GPUs to use
#SBATCH --mem=16G     	    	# memory per node (0 = use all of it)
#SBATCH --time=02:00:00         # time (DD-HH:MM)
#SBATCH --account=def-adcockb

module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch numpy scipy Pillow --no-index

cd $SLURM_TMPDIR
git clone https://github.com/mneyrane/NESTA-net.git
cd NESTA-net
git checkout package
pip install -e .
cd demos/QCBP_TV_Haar/CC

PROJECT_DIR="$HOME/projects/def-adcockb/mneyrane"
ARCHIVE_NAME="CC_res_$SLURM_ARRAY_TASK_ID.tar"

NUM_TRIALS=4
IMAGE_PATH="$SLURM_TMPDIR/NESTA-net/demos/images/GPLU_phantom_512.png"
MASK_PATH="$SLURM_TMPDIR/NESTA-net/demos/QCBP_TV_Haar/CC/mask.png"
ETA="0.01"
ETA_PERT_VALUES=("0.01" "0.1" "1.0" "10.0")

for ((t=1;t<=NUM_TRIALS;t++)); do
    for i in ${!ETA_PERT_VALUES[@]}; do
        CUDA_VISIBLE_DEVICES=$i python CC_stability.py --save-dir "RES-$SLURM_ARRAY_TASK_ID-$t-$i" --image-path $IMAGE_PATH --mask-path $MASK_PATH --eta $ETA --eta-pert ${ETA_PERT_VALUES[$i]} &
    done

    wait # for tasks to finish

done

tar -cf $ARCHIVE_NAME RES-*

mv $ARCHIVE_NAME $PROJECT_DIR
