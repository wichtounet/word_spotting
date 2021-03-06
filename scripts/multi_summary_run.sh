#!/bin/bash

# Bash script to extract trec eval values of several runs spanning different cvs

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

echo "Global Summary:"

cv1_best_total=0
cv1_best_machine=0

cv2_best_total=0
cv2_best_machine=0

cv3_best_total=0
cv3_best_machine=0

cv4_best_total=0
cv4_best_machine=0

cd run/${cv1}

for log_file in *.log; do
    machine=${log_file%log}
    machine=${machine%?}

    gmap=`${grep} map ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ $(echo " $total > $cv1_best_total" | bc) -eq 1 ]; then
        cv1_best_total=$total
        cv1_best_machine=$machine
    fi
done

cd ../${cv2}

for log_file in *.log; do
    machine=${log_file%log}
    machine=${machine%?}

    gmap=`${grep} map ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ $(echo " $total > $cv2_best_total" | bc) -eq 1 ]; then
        cv2_best_total=$total
        cv2_best_machine=$machine
    fi
done

cd ../${cv3}

for log_file in *.log; do
    machine=${log_file%log}
    machine=${machine%?}

    gmap=`${grep} map ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ $(echo " $total > $cv3_best_total" | bc) -eq 1 ]; then
        cv3_best_total=$total
        cv3_best_machine=$machine
    fi
done

cd ../${cv4}

for log_file in *.log; do
    machine=${log_file%log}
    machine=${machine%?}

    gmap=`${grep} map ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    grp=`${grep} R-prec ${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lmap=`${grep} map ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    lrp=`${grep} R-prec ${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`

    total="$(echo "${gmap} + ${grp} + ${lmap} + ${lrp}" | bc -l)"

    if [ $(echo " $total > $cv4_best_total" | bc) -eq 1 ]; then
        cv4_best_total=$total
        cv4_best_machine=$machine
    fi
done

cd ..

echo "$cv1_best_machine is the best machine for CV1 ($cv1_best_total)"
echo "$cv2_best_machine is the best machine for CV2 ($cv2_best_total)"
echo "$cv3_best_machine is the best machine for CV3 ($cv3_best_total)"
echo "$cv4_best_machine is the best machine for CV4 ($cv4_best_total)"

rm -f global_rel_file
rm -f global_top_file

rm -f local_rel_file
rm -f local_top_file

cat ${cv1}/${cv1_best_machine}_test_global_rel_file >> global_rel_file
cat ${cv2}/${cv2_best_machine}_test_global_rel_file | sed -e "s/cv1/cv2/" >> global_rel_file
cat ${cv3}/${cv3_best_machine}_test_global_rel_file | sed -e "s/cv1/cv3/" >> global_rel_file
cat ${cv4}/${cv4_best_machine}_test_global_rel_file | sed -e "s/cv1/cv4/" >> global_rel_file

cat ${cv1}/${cv1_best_machine}_test_global_top_file >> global_top_file
cat ${cv2}/${cv2_best_machine}_test_global_top_file | sed -e "s/cv1/cv2/" >> global_top_file
cat ${cv3}/${cv3_best_machine}_test_global_top_file | sed -e "s/cv1/cv3/" >> global_top_file
cat ${cv4}/${cv4_best_machine}_test_global_top_file | sed -e "s/cv1/cv4/" >> global_top_file

cat ${cv1}/${cv1_best_machine}_test_local_rel_file >> local_rel_file
cat ${cv2}/${cv2_best_machine}_test_local_rel_file | sed -e "s/cv1/cv2/" >> local_rel_file
cat ${cv3}/${cv3_best_machine}_test_local_rel_file | sed -e "s/cv1/cv3/" >> local_rel_file
cat ${cv4}/${cv4_best_machine}_test_local_rel_file | sed -e "s/cv1/cv4/" >> local_rel_file

cat ${cv1}/${cv1_best_machine}_test_local_top_file >> local_top_file
cat ${cv2}/${cv2_best_machine}_test_local_top_file | sed -e "s/cv1/cv2/" >> local_top_file
cat ${cv3}/${cv3_best_machine}_test_local_top_file | sed -e "s/cv1/cv3/" >> local_top_file
cat ${cv4}/${cv4_best_machine}_test_local_top_file | sed -e "s/cv1/cv4/" >> local_top_file

trec=/home/wichtounet/dev/trec_eval/trec_eval

gmap=`$trec -q global_rel_file global_top_file | ${grep} "map\s*all" | cut -f3`
grp=`$trec -q global_rel_file global_top_file | ${grep} "\(R-prec\)\s*all" | cut -f3`
lmap=`$trec -q local_rel_file local_top_file | ${grep} "map\s*all" | cut -f3`
lrp=`$trec -q local_rel_file local_top_file | ${grep} "\(R-prec\)\s*all" | cut -f3`

echo "G-MAP: $gmap"
echo "G-RP:  $grp"

echo "L-MAP: $lmap"
echo "L-RP:  $lrp"
