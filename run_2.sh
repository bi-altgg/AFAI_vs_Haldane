#!/bin/bash

#SBATCH     --partition=normal
#SBATCH     --nodes=2
#SBATCH     --ntasks=32
#SBATCH     --time=1-12:00:00
#SBATCH     --mail-type=ALL
#SBATCH     --mail-user=bkg230000@utdallas.edu
#SBATCH     --error=slurm.err                # Error file name
#SBATCH     --output=slurm.out               # Output file name

prun python3 WO2_long_time.py
