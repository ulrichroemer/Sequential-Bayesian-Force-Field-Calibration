# Sequential-Bayesian-Force-Field-Calibration

This repository contains code and data related to the publication:

Raabe G, Chheda V, Römer U. Sequential Bayesian Force Field Calibration of Lennard-Jones Parameters with Experimental Data. 
ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-0htxq  

## Description of Bayesian sequential design files:

### sequentialGP.py: 
Core functions for adaptive Gaussian process modelling, including evaluation of Bayes risk

### main_LJ.py:
Main file to carry out a single iteration of Bayes risk minimization for Lennard Jones parameters, including data described in the paper


### Requirements: 
The sequential design code has been developed and tested with Python 3.12. Besides numpy, scipy and a couple of standard packages, it requires: 
UQpy (https://github.com/SURGroup/UQpy)
chaospy (https://chaospy.readthedocs.io/en/master/)

## Description of molecular simulation files:

### towhee_ff_R1130E: 
Force Field file for the modelling of the component R-1130E in the Monte Carlo molecular simulation code towhee: https://sourceforge.net/projects/towhee/

### FIELD: 
Force Field file for the modelling of the component R-1130E in the general purpose molecular dynamics simulation package DL_POLY (classic): https://gitlab.com/DL_POLY_Classic


