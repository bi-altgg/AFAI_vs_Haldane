#!/bin/bash
# Job name:
#SBATCH --job-name=run2
#
# Partition:
#SBATCH --partition=normal
#
# Processors:
#SBATCH --ntasks=16
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=00:59:00
#
# Email at end:
#SBATCH --mail-user=bkg230000@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o run2.o%j
#------------------------------------------------------

echo "Starting job run2"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=run2.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job run2 complete"
