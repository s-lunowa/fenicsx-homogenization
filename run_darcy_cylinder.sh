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

for c in "isotropic" "anisotropic" "anisotropic_p" "anisotropic_v" "anisotropic_avg"; do
    for r in 0.25 0.3 0.35 0.4 0.45 0.5; do
        for f in "box" "gaussian"; do
            for b in "avg" "fluid" "reflection" "wall"; do
                dirname="darcy/cylinder_${c}"
                mkdir -p $dirname
                if [ $N -ge 1 ]; then
                    mpirun -np $N --bind-to core python3 darcy_cylinder.py -c $c -b $b -f $f -r $r |& tee "${dirname}/REV_${r}_Filter_${f}_BC_${b}.log" &
                else
                    time python3 darcy_cylinder.py -c $c -b $b -f $f -r $r |& tee "${dirname}/REV_${r}_Filter_${f}_BC_${b}.log"
                fi
            done
            wait
        done
    done
done
