#!/bin/sh
#SBATCH --ntasks=32
#SBATCH --mem=30G  # memory in Mb
#SBATCH --partition=medium
#SBATCH -a 129-160
#SBATCH -J pynm
#SBATCH -o testtimon-%j-%a.out
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 7-00:00:00  # time requested in days-hours:minutes:seconds
module load python
python examples/example_rns_stream_cluster.py $SLURM_ARRAY_TASK_ID
