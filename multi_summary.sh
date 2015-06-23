#!/bin/bash

# Bash script to extract trec eval values of several runs spanning different cvs

machines=(160.98.22.21 160.98.22.22 160.98.22.23 160.98.22.24 160.98.22.25 160.98.22.8 160.98.22.9)

cv1=$1
cv2=$2
cv3=$3
cv4=$4

if [ ! -d run/${cv1} ]; then
    echo "Invalid cv1 stamp"
    exit
fi

if [ ! -d run/${cv2} ]; then
    echo "Invalid cv2 stamp"
    exit
fi

if [ ! -d run/${cv3} ]; then
    echo "Invalid cv3 stamp"
    exit
fi

if [ ! -d run/${cv4} ]; then
    echo "Invalid cv4 stamp"
    exit
fi

grep=/usr/bin/zgrep

cd run

echo "Global Summary:"

cv1_best_total=0
cv1_best_machine=0

cv2_best_total=0
cv2_best_machine=0

cv3_best_total=0
cv3_best_machine=0

cv4_best_total=0
cv4_best_machine=0

for machine in ${!machines[@]}; do
    gmap=`${grep} map ${cv1}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${cv1}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${cv1}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${cv1}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ "$total" > "$cv1_best_total" ]; then
        cv1_best_total=$total
        cv1_best_machine=$machine
    fi
done

for machine in ${!machines[@]}; do
    gmap=`${grep} map ${cv2}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv2_ | cut -f3`
    grp=`${grep} R-prec ${cv2}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv2_ | cut -f3`
    lmap=`${grep} map ${cv2}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv2_ | cut -f3`
    lrp=`${grep} R-prec ${cv2}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv2_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ "$total" > "$cv2_best_total" ]; then
        cv2_best_total=$total
        cv2_best_machine=$machine
    fi
done

for machine in ${!machines[@]}; do
    gmap=`${grep} map ${cv3}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv3_ | cut -f3`
    grp=`${grep} R-prec ${cv3}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv3_ | cut -f3`
    lmap=`${grep} map ${cv3}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv3_ | cut -f3`
    lrp=`${grep} R-prec ${cv3}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv3_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ "$total" > "$cv3_best_total" ]; then
        cv3_best_total=$total
        cv3_best_machine=$machine
    fi
done

for machine in ${!machines[@]}; do
    gmap=`${grep} map ${cv4}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv4_ | cut -f3`
    grp=`${grep} R-prec ${cv4}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv4_ | cut -f3`
    lmap=`${grep} map ${cv4}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv4_ | cut -f3`
    lrp=`${grep} R-prec ${cv4}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv4_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ "$total" > "$cv4_best_total" ]; then
        cv4_best_total=$total
        cv4_best_machine=$machine
    fi
done

echo "$cv1_best_machine is the best machine for CV1 ($cv1_best_total)"
echo "$cv2_best_machine is the best machine for CV2 ($cv2_best_total)"
echo "$cv3_best_machine is the best machine for CV3 ($cv3_best_total)"
echo "$cv4_best_machine is the best machine for CV4 ($cv4_best_total)"
