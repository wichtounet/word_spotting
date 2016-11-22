#!/bin/bash

# Bash script to extract trec_eval from a standard result (result or dtw_result)

folder=$1

echo "TREC evaluation of $1"

if [ -d "$folder" ]; then
    /home/wichtounet/dev/trec_eval/trec_eval -q $folder/global_rel_file $folder/global_top_file
else
    echo "The directory \"$folder\" does not exist"
    if [ -d "results/$folder" ]; then
        $0 results/$folder
    fi
fi
