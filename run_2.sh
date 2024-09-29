#!/bin/bash

#SBATCH     --partition=normal
#SBATCH     --nodes=2
#SBATCH     --ntasks=32
#SBATCH     --time=1-12:00:00
#SBATCH     --mail-type=ALL
#SBATCH     --mail-user=your.email@utdallas.edu

prun python WO2_long_time.py