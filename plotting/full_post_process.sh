#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
DIR=""
NCORE=""

while getopts ":d:n:h?:" opt; do
    case "$opt" in
    h|\?)
        echo "specify dir with -d and core number with -n" 
        exit 0
        ;;
    d)  DIR=$OPTARG
        ;;
    n)  NCORE=$OPTARG
        ;;
    esac
done
echo $DIR
echo $NCORE
echo "Processing $DIR on $NCORE cores"

mpiexec_mpt -n $NCORE python3 plot_rolled_structure.py $DIR --roll_writes=50
mpiexec_mpt -n $NCORE python3 find_top_cz.py $DIR
mpiexec_mpt -n $NCORE python3 plot_3d_slices.py $DIR
mpiexec_mpt -n $NCORE python3 plot_3d_slices_horiz.py $DIR
mpiexec_mpt -n $NCORE python3 fig2_profiles.py $DIR

cd $DIR
$OLDPWD/png2mp4.sh top_cz/top_cz top_cz.mp4 30
$OLDPWD/png2mp4.sh snapshots/ snapshots.mp4 30
$OLDPWD/png2mp4.sh snapshots_horiz/ snapshots_horiz.mp4 30
$OLDPWD/png2mp4.sh rolled_structure/ rolled_structure.mp4 30
$OLDPWD/png2mp4.sh paper_profiles/paper paper_profiles.mp4 30
#tar -cvf avg_profs.tar avg_profs/
cd $OLDPWD
