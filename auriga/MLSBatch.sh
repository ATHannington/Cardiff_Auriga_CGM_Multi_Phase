#!/bin/bash --login
###
#job name
#SBATCH --job-name=MachineLearning
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#job stdout file
#SBATCH --output=bench.out.%J
#job stderr file
#SBATCH --error=bench.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=0-10:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=80 
#memory per process in MB 
#SBATCH --mem-per-cpu=8000 
#tasks to run per node (change for hybrid OpenMP/MPI) 
#SBATCH --ntasks-per-node=40
#SBATCH --mail-user=batesm1@cardiff.ac.uk
#SBATCH --mail-type=ALL
###

#now run normal batch commands 
module load tensorflow/1.11
module load CUDA/9.1

python3 NonPeriodicExponentiatedNoisyFields/NeuralNetwork.py
