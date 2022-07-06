# WannierFunctions

[![Build Status](https://github.com/jaemolihm/WannierFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaemolihm/WannierFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jaemolihm/WannierFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jaemolihm/WannierFunctions.jl)

Construct maximally localized Wannier functions

Largely adapted from https://github.com/antoine-levitt/wannier

## Functionalities
* Spread functional: Marzari and Vanderbilt (1997)
* Disentanglement: frozen window
* Variational Wannier functions

## TODO
- [ ] Outer window (disentanglement window)
- [ ] spread functional by Stengel and Spaldin
- [ ] Symmetry
- [ ] run_wannier_minimization modifies U
- [ ] Computing wbs
- [ ] Modularize params
- [ ] Reduce compilation time
- [ ] Test systems where W90 line search fails
- [ ] Read kpoints, lattice, ... from win file (kpoint: currently kmesh.pl order is required.)
