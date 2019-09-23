#!/bin/bash -e
#SBATCH --job-name=ros-slurm-prof      # job name (shows up in the queue)
#SBATCH --account=massey02373     # Project Account
#SBATCH --time=00:20:00         # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=3000      # memory/cpu (in MB)
#SBATCH --ntasks=12              # number of tasks (e.g. MPI)
#SBATCH --cpus-per-task=1       # number of cores per task (e.g. OpenMP)
#SBATCH --partition=large        # specify a partition
#SBATCH --hint=nomultithread    # don't use hyperthreading
#SBATCH --profile=task          # enable Slurm profiling
#SBATCH --acctg-freq=2          # gather profiling statistics every 5 seconds

module load Julia/.1.2.0-gimkl-2018b-VTune
##module load VTune


export OPENBLAS_NUM_THREADS=1
export ENABLE_JITPROFILING=1

srun julia ros_BHM_M50_U6_W1M.jl > output-${SLURM_JOB_ID}.out

sbatch -n1 -d$SLURM_JOB_ID --wrap="sh5util -j $SLURM_JOB_ID -o profile-${SLURM_JOB_ID}.h5"
