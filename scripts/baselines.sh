#!/bin/bash

# Bash script to extract trec_eval from a standard result (result or dtw_result)

# 1. Build the project

config="release_debug"
exec_file="./$config/bin/spotter"
options="-notrain -novalid"

make -j9 $config > /dev/null 2>&1

# 2. Collect results for Washington

echo "Washington Database"
echo ""

db_path="/home/wichtounet/datasets/washington"
db_option="-washington"

for db_cv in "cv1" "cv2" "cv3" "cv4"
do
    # Print the header
    printf "%20s | %10s | %10s | %10s | %10s |\n" "$db_cv" "G-MAP" "G-RP" "L-MAP" "L-RP"

    # Compute the results on each method

    for method in "-0" "-3" "-4" "-5" "-6" "-7"
    do
        results=`$exec_file $method $db_option $options train $db_path $db_cv`
        method_name=$(echo "$results" | grep "Method: " | cut -d" " -f4)
        folder=$(echo "$results" | grep "dtw_results" | cut -d" " -f2)
        trec_results=`./scripts/trec_eval.sh $folder`
        gmap=$(echo "$trec_results" | head -n1 | cut -d" " -f2)
        grp=$(echo "$trec_results" | head -n2 | tail -n1 | cut -d" " -f3)
        lmap=$(echo "$trec_results" | head -n3 | tail -n1 | cut -d" " -f2)
        lrp=$(echo "$trec_results" | tail -n1 | cut -d" " -f3)

        printf "%20s | %10s | %10s | %10s | %10s |\n" "$method_name" "$gmap" "$grp" "$lmap" "$lrp"
    done

    echo ""
done

echo ""
