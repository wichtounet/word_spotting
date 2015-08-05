#!/bin/bash

# Bash script to generate the trec_eval summary from a run given its stamp

stamp=$1

if [ ! -d run/${stamp} ]; then
    echo "Invalid stamp"
    exit
fi

grep=/usr/bin/zgrep

cd run/${stamp}

echo "Stamp Summary: $stamp"

for log_file in *.log; do
    machine=${log_file%log}
    machine=${machine%?}

    echo "Machine $machine"
    if [ "all" == "$2" ]; then
        echo "  Train results"
        echo "    G-MAP " `${grep} map ${machine}_train_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
        echo "    G-RP " `${grep} R-prec ${machine}_train_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
        echo "    L-MAP " `${grep} map ${machine}_train_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
        echo "    L-RP " `${grep} R-prec ${machine}_train_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    fi
    echo "  Valid results"
    echo "    G-MAP " `${grep} map ${machine}_valid_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    G-RP " `${grep} R-prec ${machine}_valid_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-MAP " `${grep} map ${machine}_valid_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-RP " `${grep} R-prec ${machine}_valid_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  Test results"
    echo "    G-MAP " `${grep} map ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    G-RP " `${grep} R-prec ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-MAP " `${grep} map ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "    L-RP " `${grep} R-prec ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
done
