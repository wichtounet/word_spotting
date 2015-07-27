#!/bin/bash

# Bash script to extract trec eval values of several dtw runs spanning different cvs

cv1=$1
cv2=$2
cv3=$3
cv4=$4

if [ ! -d dtw_results/${cv1} ]; then
    echo "Invalid cv1 stamp"
    exit
fi

if [ ! -d dtw_results/${cv2} ]; then
    echo "Invalid cv2 stamp"
    exit
fi

if [ ! -d dtw_results/${cv3} ]; then
    echo "Invalid cv3 stamp"
    exit
fi

if [ ! -d dtw_results/${cv4} ]; then
    echo "Invalid cv4 stamp"
    exit
fi

grep=/usr/bin/zgrep
trec=/home/wichtounet/dev/trec_eval/trec_eval

rm -f global_rel_file
rm -f global_top_file

rm -f local_rel_file
rm -f local_top_file

cat dtw_results/${cv1}/global_rel_file >> global_rel_file
cat dtw_results/${cv2}/global_rel_file | sed -e "s/cv1/cv2/" >> global_rel_file
cat dtw_results/${cv3}/global_rel_file | sed -e "s/cv1/cv3/" >> global_rel_file
cat dtw_results/${cv4}/global_rel_file | sed -e "s/cv1/cv4/" >> global_rel_file

cat dtw_results/${cv1}/global_top_file >> global_top_file
cat dtw_results/${cv2}/global_top_file | sed -e "s/cv1/cv2/" >> global_top_file
cat dtw_results/${cv3}/global_top_file | sed -e "s/cv1/cv3/" >> global_top_file
cat dtw_results/${cv4}/global_top_file | sed -e "s/cv1/cv4/" >> global_top_file

cat dtw_results/${cv1}/local_rel_file >> local_rel_file
cat dtw_results/${cv2}/local_rel_file | sed -e "s/cv1/cv2/" >> local_rel_file
cat dtw_results/${cv3}/local_rel_file | sed -e "s/cv1/cv3/" >> local_rel_file
cat dtw_results/${cv4}/local_rel_file | sed -e "s/cv1/cv4/" >> local_rel_file

cat dtw_results/${cv1}/local_top_file >> local_top_file
cat dtw_results/${cv2}/local_top_file | sed -e "s/cv1/cv2/" >> local_top_file
cat dtw_results/${cv3}/local_top_file | sed -e "s/cv1/cv3/" >> local_top_file
cat dtw_results/${cv4}/local_top_file | sed -e "s/cv1/cv4/" >> local_top_file

gmap=`$trec -q global_rel_file global_top_file | ${grep} "map\s*all" | cut -f3`
grp=`$trec -q global_rel_file global_top_file | ${grep} "\(R-prec\)\s*all" | cut -f3`
lmap=`$trec -q local_rel_file local_top_file | ${grep} "map\s*all" | cut -f3`
lrp=`$trec -q local_rel_file local_top_file | ${grep} "\(R-prec\)\s*all" | cut -f3`

echo "Global Summary:"

echo "G-MAP: $gmap"
echo "G-RP:  $grp"

echo "L-MAP: $lmap"
echo "L-RP:  $lrp"
