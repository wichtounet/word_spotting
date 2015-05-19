#!/bin/bash

machines=(160.98.22.21 160.98.22.22 160.98.22.23 160.98.22.24 160.98.22.25 160.98.22.8 160.98.22.9)

stamp=$1

if [ ! -d run/${stamp} ]; then
    echo "Invalid stamp"
    exit
fi

cd run

echo "Stamp Summary: $stamp"

for machine in ${!machines[@]}; do
    echo "Machine $machine (${machines[machine]})"
    if [ "all" == "$2" ]; then
        echo "  Train results"
        echo "    G-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_train_global_eval | /usr/bin/zgrep all | cut -f3`
        echo "    G-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_train_global_eval | /usr/bin/zgrep all | cut -f3`
        echo "    L-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_train_local_eval | /usr/bin/zgrep all | cut -f3`
        echo "    L-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_train_local_eval | /usr/bin/zgrep all | cut -f3`
    fi
    echo "  Test results"
    echo "    G-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_test_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "    G-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_test_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "    L-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_test_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "    L-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_test_local_eval | /usr/bin/zgrep all | cut -f3`
done
