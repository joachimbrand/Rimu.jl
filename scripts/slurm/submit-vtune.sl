#!/bin/bash -e
#SBATCH --job-name=ros-slurm-prof      # job name (shows up in the queue)
#SBATCH --account=massey02373    # Project Account
#SBATCH --time=00:20:00         # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=3000      # memory/cpu (in MB)
#SBATCH --ntasks=12              # number of tasks (e.g. MPI)
#SBATCH --nodes=1             # number of nodes
#SBATCH --cpus-per-task=1       # number of cores per task (e.g. OpenMP)
#SBATCH --partition=large        # specify a partition
#SBATCH --hint=nomultithread    # don't use hyperthreading

module load Julia/.1.1.1-gimkl-2018b-VTune
module load VTune
module load Python

export OPENBLAS_NUM_THREADS=1
export ENABLE_JITPROFILING=1

# run the code through VTune
srun amplxe-cl -collect hotspots -r vtune-${SLURM_JOB_ID} -- \
   julia ros_BHM_M50_U6_W1M.jl > output-${SLURM_JOB_ID}.out

# the name of the directory is vtune-${SLURM_JOB_ID}.${node_name}
for d in vtune-${SLURM_JOB_ID}.*; do
    # extract the node name
    node_name=$(echo $d | awk -F. '{print $2;}')
    # create a CSV file with CPU Time spent in each function
    amplxe-cl --report=hotspots -r vtune-${SLURM_JOB_ID}.${node_name} --format=csv --report-output=vtune-${SLURM_JOB_ID}-${node_name}.csv
    # produce the png file
    python showprof.py --csv=vtune-${SLURM_JOB_ID}-${node_name}.csv --title="ros_BHM_M50_U6_W1M" --save
done
