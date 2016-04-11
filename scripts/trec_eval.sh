#!/bin/bash

# Bash script to extract trec_eval from a standard result (result or dtw_result)

folder=$1

if [ -d "$folder" ]; then
    gmap=`/home/wichtounet/dev/trec_eval/trec_eval -q $folder/global_rel_file $folder/global_top_file | /usr/bin/zgrep "map\s*all" | cut -f3`
    grp=`/home/wichtounet/dev/trec_eval/trec_eval -q $folder/global_rel_file $folder/global_top_file | /usr/bin/zgrep "\(R-prec\)\s*all" | cut -f3`
    lmap=`/home/wichtounet/dev/trec_eval/trec_eval -q $folder/local_rel_file $folder/local_top_file | /usr/bin/zgrep "map\s*all" | cut -f3`
    lrp=`/home/wichtounet/dev/trec_eval/trec_eval -q $folder/local_rel_file $folder/local_top_file | /usr/bin/zgrep "\(R-prec\)\s*all" | cut -f3`

    echo "G-MAP: $gmap"
    echo "G-RP:  $grp"
    echo "L-MAP: $lmap"
    echo "L-RP:  $lrp"
else
    echo "The directory \"$folder\" does not exist"
fi

