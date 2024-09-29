#!/bin/bash

#SBATCH     --partition=normal
#SBATCH     --nodes=2
#SBATCH     --ntasks=32
#SBATCH     --time=1-12:00:00
#SBATCH     --mail-type=ALL
#SBATCH     --mail-user=your.email@utdallas.edu
#SBATCH     --error=slurm.err                # Error file name
#SBATCH     --output=slurm.out               # Output file name

prun python WO2_long_time.py
