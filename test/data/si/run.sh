#!/bin/sh
#SBATCH --partition=G4
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=pos_si
#SBATCH --output="error.%x"
QE=/home/jmlim/dipole/q-e/bin
#QE=/home/jmlim/program/qe-dev/bin
W90=/home/jmlim/position/wannier90/wannier90.x
POSTW90=/home/jmlim/position/wannier90/postw90.x

PREFIX=si

# run parallel
srun -n 20 $QE/pw.x -nk 10 -in scf.in > scf.out
srun -n 20 $QE/pw.x -nk 10 -in bands.in > bands.out
srun -n 20 $QE/pw.x -nk 10 -in nscf.in > nscf.out
srun -n 1 $W90 -pp si
srun -n 20 $QE/pw2wannier90.x -nk 1 -pd .true. -in pw2wan.in > pw2wan.out
srun -n 20 $W90 si
