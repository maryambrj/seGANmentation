#!/bin/bash
#SBATCH --job-name=pretrain_gen
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --account=data-machine
#SBATCH --nodelist=acm-[048-049,070-071],nal-[004-007]  # Force job to run on data machine nodes
#SBATCH --gpus=1
#SBATCH --output=pretrain_gen.out
#SBATCH --error=pretrain_gen.err
###SBATCH --partition=normal
###SBATCH --gres=gpu:a100:1
###SBATCH --gpus=a100_1g.10gb:2

module load CUDA/11.3.1

source ~/miniconda3/bin/activate ~/miniconda3/envs/seganmentation

~/miniconda3/envs/seganmentation/bin/python3 ~/seGANmentation/uvcgan2/scripts/Carvana/pretrain_generator.py