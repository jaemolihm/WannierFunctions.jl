#!/bin/sh
#SBATCH --partition=G1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name=pos_si
#SBATCH --output="error.%x"
QE=/home/jmlim/dipole/q-e/bin
#QE=/home/jmlim/program/qe-dev/bin
W90=/home/jmlim/position/wannier90/wannier90.x
POSTW90=/home/jmlim/position/wannier90/postw90.x

PREFIX=si

# run parallel
#srun -n $SLURM_NTASKS $QE/pw.x -nk 10 -in scf.in > scf.out
#srun -n $SLURM_NTASKS $QE/pw.x -nk 10 -in bands.in > bands.out
#srun -n $SLURM_NTASKS $QE/pw.x -nk 10 -in nscf.in > nscf.out
#srun -n 1 $W90 -pp si
#srun -n $SLURM_NTASKS $QE/pw2wannier90.x -nk 1 -pd .true. -in pw2wan.in > pw2wan.out
srun -n $SLURM_NTASKS $W90 si
