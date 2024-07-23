#!/bin/bash
#SBATCH --job-name=run
#SBATCH -p defq
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G  
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=maike.nutzel@gmail.com

# change to local scratch
module load shared slurm
source activate . .venv/bin/activate

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"
echo "== Scratch dir. : ${TMPDIR}"


# I. Define directory names [DO NOT CHANGE]
# =========================================
# get name of the temporary directory working directory, physically on the compute-node
workdir="${TMPDIR}"
# get submit directory
# (every file/folder below this directory is copied to the compute node)
submitdir="${SLURM_SUBMIT_DIR}"

# 1. Transfer to node [DO NOT CHANGE]
# ===================================
# create/empty the temporary directory on the compute node
if [ ! -d "${workdir}" ]; then
  mkdir -p "${workdir}"
else
  rm -rf "${workdir}"/*
fi

# change current directory to the location of the sbatch command
# ("submitdir" is somewhere in the home directory on the head node)
cd "${submitdir}"
# copy all files/folders in "submitdir" to "workdir"
# ("workdir" == temporary directory on the compute node)
cp -prf * ${workdir}
# change directory to the temporary directory on the compute-node
cd ${workdir}

# 3. Function to transfer back to the head node [DO NOT CHANGE]
# =============================================================
# define clean-up function
function clean_up {
  # - remove log-file on the compute-node, to prevent overwiting actual output with empty file
  rm slurm-${SLURM_JOB_ID}.out
  # - TODO delete temporary files from the compute-node, before copying. Prevent copying unnecessary files.
  # rm -r ...
  # - change directory to the location of the sbatch command (on the head node)
  cd "${submitdir}"
  # - copy everything from the temporary directory on the compute-node
  cp -prf "${workdir}"/* .
  # - erase the temporary directory from the compute-node
  rm -rf "${workdir}"/*
  rm -rf "${workdir}"
  # - exit the script
  exit
}

b_s=2068
dropout=0.64
lr=0.002
num_layers=1
lstm_config=1L_152
# index_start=1
# index_end=3
# index_step=1

# index_runs=(0 1 2 3)

# call "clean_up" function when this script exits, it is run even if SLURM cancels the job
trap 'clean_up' EXIT
set -e
#for run in $(seq $index_start $index_step $index_end); do
 # echo "run ${run}"
make lstm_config=$lstm_config lr=$lr b_s=$b_s dropout=$dropout train
make lstm_config=$lstm_config lr=$lr b_s=$b_s dropout=$dropout test
#done