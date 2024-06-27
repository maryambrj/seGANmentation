#!/bin/bash
#SBATCH --job-name=train_translation
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
###SBATCH --account=data-machine
###SBATCH --nodelist=acm-[048-049,070-071],nal-[004-007]  # Force job to run on data machine nodes
#SBATCH --partition=normal
###SBATCH --gpus=a100_1g.10gb
#SBATCH --gpus=1
#SBATCH --output=train_translation.out
#SBATCH --error=train_translation.err

module load CUDA/11.3.1

source ~/miniconda3/bin/activate ~/miniconda3/envs/seganmentation

~/miniconda3/envs/seganmentation/bin/python3 ~/seGANmentation/uvcgan2/scripts/Carvana/train_translation.py