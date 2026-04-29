#!/usr/bin/env bash
#
# Author: S.B. Lunowa

usage() {
    echo
    echo "Usage: $0 [-m [N]]"
    echo
    echo "    -m, --mpi [N] allows to run using MPI parallelization with N cores (Default N=32)."
    exit $@
}
error() {
    echo "Error: Unknown arguments: $@"
    usage 1
}

if [ $# -eq 0 ]; then
    N=0
elif [ "$1" = "-m" ] || [ "$1" = "--mpi" ]; then
    N=32
elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
else
    error $@
fi

if [ $# -eq 2 ]; then
    N=$2
elif [ $# -gt 2 ]; then
    error $@
fi

for case in "cube_spiral" "cylinder_spiral"; do
    dirname="stokes/${case}/"
    mkdir -p $dirname
    if [ $N -ge 1 ]; then
        mpirun -np $N --bind-to core python3 stokes.py -c $case --BDM --cholesky |& tee "${dirname}/result.log"
    else
        time python3 stokes.py -c $case --BDM --cholesky |& tee "${dirname}/result.log"
    fi
done
