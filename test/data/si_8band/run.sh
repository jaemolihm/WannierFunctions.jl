#!/bin/bash
set -ex

QE=/home/jmlim/dipole/q-e/bin
W90=/home/jmlim/position/wannier90/wannier90.x

mpirun -np 4 $QE/pw.x -nk 4 -in scf.in > scf.out
mpirun -np 4 $QE/pw.x -nk 4 -in bands.in > bands.out
my_qe_bands.py si temp
mpirun -np 4 $QE/pw.x -nk 4 -in nscf.in > nscf.out

mpirun -np 1 $W90 -pp si
mpirun -np 4 $QE/pw2wannier90.x -nk 1 -in pw2wan.in > pw2wan.out
mpirun -np 4 $W90 si
