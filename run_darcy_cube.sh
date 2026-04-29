#!/usr/bin/env bash
#
# Author: S.B. Lunowa

usage() {
    echo
    echo "Usage: $0 [-m [N]]"
    echo
    echo "    -m, --mpi [N] allows to run using MPI parallelization with N cores (Default N=16)."
    exit $@
}
error() {
    echo "Error: Unknown arguments: $@"
    usage 1
}

if [ $# -eq 0 ]; then
    N=0
elif [ "$1" = "-m" ] || [ "$1" = "--mpi" ]; then
    N=16
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

eps=0.25

for case in "isotropic" "anisotropic" "anisotropic_p" "anisotropic_v" "anisotropic_avg" "homogenized"; do
    dirname="darcy/cube_eps${eps}/"
    mkdir -p $dirname
    if [ $N -ge 1 ]; then
        mpirun -np $N --bind-to core python3 darcy.py -c $case -e $eps |& tee "${dirname}/${case}.log" &
    else
        time python3 darcy.py -c $case -e $eps |& tee "${dirname}/${case}.log"
    fi
done
wait
