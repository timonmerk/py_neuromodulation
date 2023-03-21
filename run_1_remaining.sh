#!/bin/sh
#SBATCH --ntasks=13
#SBATCH -N 1  # one node
#SBATCH --ntasks-per-node=13
#SBATCH --mem=30G  # memory in Mb
#SBATCH --partition=medium
#SBATCH -a 0-12
#SBATCH -J pynm
#SBATCH -o /data/gpfs-1/users/merkt_c/work/OUT/log/testtimon-%j-%a.out
#SBATCH -e /data/gpfs-1/users/merkt_c/work/OUT/log/errfile  # send stderr to errfile
#SBATCH -t 7-00:00:00  # time requested in days-hours:minutes:seconds
module load python
python examples/example_rns_stream_cluster_remaining_subjects.py $SLURM_ARRAY_TASK_ID
