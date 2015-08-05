#!/bin/bash

# Bash script to extract trec eval values of several runs spanning different cvs

cv=$1

if [ ! -d run/${cv} ]; then
    echo "Invalid cv1 stamp"
    exit
fi

grep=/usr/bin/zgrep

cd run/${cv}

echo "Global Summary:"

cv_best_total=0
cv_best_machine=0

for log_file in *.log; do
    machine=${log_file%log}
    machine=${machine%?}

    gmap=`${grep} map ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ $(echo " $total > $cv_best_total" | bc) -eq 1 ]; then
        cv_best_total=$total
        cv_best_machine=$machine
    fi
done

echo "$cv_best_machine is the best machine for run $cv ($cv_best_total)"

gmap=`${grep} map ${cv_best_machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
grp=`${grep} R-prec ${cv_best_machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
lmap=`${grep} map ${cv_best_machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
lrp=`${grep} R-prec ${cv_best_machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

echo "G-MAP: $gmap"
echo "G-RP:  $grp"

echo "L-MAP: $lmap"
echo "L-RP:  $lrp"
