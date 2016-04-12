#!/bin/bash

machine=$1
number=$2

stamp=`cat results/stamp`
new_stamp=$((stamp+1))
echo "$new_stamp" > results/stamp

password=`cat .grid_passwd`

echo "Downloading results/$number from $machine"
sshpass -p $password scp -r wicht@$machine:/home/wicht/dev/word_spotting/results/$number results/$stamp
echo "Results have been downloaded in results/$stamp"

if [ "$3" == "trec" ]; then
    echo "TREC evaluation of the results"
    ./scripts/trec_eval.sh results/$stamp
fi
