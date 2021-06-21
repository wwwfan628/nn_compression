#!/bin/bash

#SBATCH --mail-type=ALL                   # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err                  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

# exit on errors
set -o errexit
# execute python file
python train.py --dataset_name=CIFAR10 --model_name=VGG --train_index --ste --granularity_kernel
# python train.py --dataset_name=CIFAR10 --model_name=VGG --train_index --ste --granularity_channel
# python train.py --dataset_name=CIFAR10 --model_name=VGG --train_index --ste
# python train3.py --dataset_name=CIFAR10 --model_name=VGG --granularity_kernel
# python train3.py --dataset_name=CIFAR10 --model_name=VGG --granularity_channel
# python train3.py --dataset_name=CIFAR10 --model_name=VGG
echo finished at: `date`
exit 0;
