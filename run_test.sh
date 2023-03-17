#!/bin/sh
#SBATCH --ntasks=2
#SBATCH --mem=30G  # memory in Mb
#SBATCH --partition=medium
#SBATCH -J pynm
#SBATCH -a 1-2
#SBATCH -o /data/gpfs-1/users/merkt_c/work/OUT/log/testtimon-%j.out
#SBATCH -e /data/gpfs-1/users/merkt_c/work/OUT/log/errfile  # send stderr to errfile
#SBATCH -t 7-00:00:00  # time requested in days-hours:minutes:seconds
module load python
python examples/example_rns_stream_cluster.py $SLURM_ARRAY_TASK_ID

