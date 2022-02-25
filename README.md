# schwarzschild_or_ledoux

This repository contains all of the code required to produce the results in "Schwarzschild and Ledoux are equivalent on evolutionary timescales" by Evan H. Anders et al.

## Running Dedalus simulations

The simulation presented in the paper was produced by running the triLayer_model.py file on [commit 1339061 of dedalus-v2](https://github.com/DedalusProject/dedalus/commit/13390610a148c40e5aa100fb6f9e0e2b0db59ab0). A sample mpi-run call for the simulation presented in the paper can be found in job_scripts/3D/turbSpan_Rinv10_pe3.2e3_pr5e-1, and the simulation in the paper ran over about 5.7e6 timesteps.

## Post-processing simulations

The plotting/ directory contains many scripts used in post-processing. Many of these scripts rely on Evan Anders' [plotpal repo and arefunctional on commit 3ba06c on the d3 branch](https://github.com/evanhanders/plotpal/commit/3ba06ca817f5c3cfacdc75b31e7763a7564d427e). If the data from a simulation run are contained in folder DATADIR, the simplest way to run the post-processing scripts on N cores is to use the full_post_process.sh script like ./full_post_process -d /path/to/DATADIR -n N

## Creating the figures in the paper

The publication_figures/ directory contains the scripts required to make the figures in the paper. To make the figures, download the data from the Zenodo repository associated with the paper, and unpack the figure_data.tar tarball inside of the publication_figures directory. Figure 1 was created by running the full publication_dynamics.ipynb jupyter noteboook. Figures 2 and 3 were made by running publication_profiles.py and publication_kippenhahn from the command line.
