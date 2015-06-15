#!/bin/bash

machines=(160.98.22.21 160.98.22.22 160.98.22.23 160.98.22.24 160.98.22.25 160.98.22.8 160.98.22.9)

stamp=$1

if [ ! -d run/${stamp} ]; then
    echo "Invalid stamp"
    exit
fi

grep=/usr/bin/zgrep

cd run

echo "Stamp Summary: $stamp"

for machine in ${!machines[@]}; do
    echo "Machine $machine (${machines[machine]})"
    if [ "all" == "$2" ]; then
        echo "  Train results"
        echo "    G-MAP " `${grep} map ${stamp}/${machine}_train_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
        echo "    G-RP " `${grep} R-prec ${stamp}/${machine}_train_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
        echo "    L-MAP " `${grep} map ${stamp}/${machine}_train_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
        echo "    L-RP " `${grep} R-prec ${stamp}/${machine}_train_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    fi
    echo "  Valid results"
    echo "    G-MAP " `${grep} map ${stamp}/${machine}_valid_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    G-RP " `${grep} R-prec ${stamp}/${machine}_valid_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-MAP " `${grep} map ${stamp}/${machine}_valid_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-RP " `${grep} R-prec ${stamp}/${machine}_valid_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  Test results"
    echo "    G-MAP " `${grep} map ${stamp}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    G-RP " `${grep} R-prec ${stamp}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-MAP " `${grep} map ${stamp}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-RP " `${grep} R-prec ${stamp}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
done
