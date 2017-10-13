#!/bin/bash

# Bash script to extract trec_eval from a standard result (result or dtw_result)

folder=$1

echo "TREC evaluation of $1"

if [ -f "/usr/bin/zgrep" ]; then
    GREP="/usr/bin/zgrep"
elif [ -f "/bin/grep" ]; then
    GREP="/bin/grep"
else
    GREP="grep"
fi

if [ -f "/home/wichtounet/dev/trec_eval/trec_eval" ]; then
    TREC="/home/wichtounet/dev/trec_eval/trec_eval"
elif [ -f "/localhome/wicht/dev/trec_eval/trec_eval" ]; then
    TREC="/localhome/wicht/dev/trec_eval/trec_eval"
else
    TREC="trec_eval"
fi

if [ -d "$folder" ]; then
    gmap=`$TREC -q $folder/global_rel_file $folder/global_top_file | $GREP "map\s*all" | cut -f3`
    grp=`$TREC -q $folder/global_rel_file $folder/global_top_file | $GREP "\(R-prec\)\s*all" | cut -f3`
    lmap=`$TREC -q $folder/local_rel_file $folder/local_top_file | $GREP "map\s*all" | cut -f3`
    lrp=`$TREC -q $folder/local_rel_file $folder/local_top_file | $GREP "\(R-prec\)\s*all" | cut -f3`

    echo "G-MAP: $gmap"
    echo "G-RP:  $grp"
    echo "L-MAP: $lmap"
    echo "L-RP:  $lrp"
else
    echo "The directory \"$folder\" does not exist"
    if [ -d "results/$folder" ]; then
        $0 results/$folder
    fi
fi
