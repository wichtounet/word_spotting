#!/bin/bash

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
    echo "Invalid cv1 stamp"
    exit
fi

if [ ! -d run/${cv3} ]; then
    echo "Invalid cv1 stamp"
    exit
fi

if [ ! -d run/${cv4} ]; then
    echo "Invalid cv1 stamp"
    exit
fi

grep=/usr/bin/zgrep

cd run

echo "Global Summary:"

best_total=0
best_machine=0

for machine in ${!machines[@]}; do
    gmap=`${grep} map ${cv1}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${cv1}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${cv1}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${cv1}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    echo "total score: $total"

    if [ "$total" > "$best_total" ]; then
        best_total=$total
        best_machine=$machine
    fi
done

echo $best_total
echo $best_machine
