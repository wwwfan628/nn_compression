#!/bin/bash

#SBATCH --mail-type=ALL                   # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err                  # where to store error messages
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

# exit on errors
set -o errexit
# execute python file
python try.py --dataset_name=ImageNet --model_name=ResNet18 --train_index
echo finished at: `date`
exit 0;
