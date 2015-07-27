#!/bin/bash

# Script to run on several machines

if [ "$1" == "third" ]; then
    mode="third"
    option="-third"
elif [ "$1" == "half" ]; then
    mode="half"
    option="-half"
elif [ "$1" == "full" ]; then
    mode="full"
    option=""
else
    echo "The first parameter must be one of [full,half,third]"
    exit 1
fi

set=$2

all_option=""
dataset="washington"
dataset_option=""

if [ "$3" == "all" ]; then
    all_option="-all"
fi

if [ "$3" == "parzival" ]; then
    dataset_option="-parzival"
    dataset="parzival"
fi

options="$option $all_option $dataset_option"

config_file="config_${mode}.hpp"

machines=(160.98.22.21 160.98.22.22 160.98.22.23 160.98.22.24 160.98.22.25 160.98.22.8 160.98.22.9)
user=wicht
password=`cat .passwd`

grep=/usr/bin/zgrep

mkdir -p run
cd run

if [ ! -f stamp ]; then
    echo "1" >> stamp
fi

stamp=`cat stamp`
new_stamp=$((stamp+1))
echo "$new_stamp" > stamp

echo "Stamp: $stamp"
echo "Mode: $mode"
echo "Dataset: $dataset"
echo "Set: $set"
echo "Options: $all_option"

mkdir -p "$stamp"

# 1.Prepare all the files

for machine in ${!machines[@]}; do
    echo "//This file will be put in ${machines[machine]}" > config_${machine}.hpp
    cat ../include/${config_file} >> config_${machine}.hpp
    vim config_${machine}.hpp
    cp config_${machine}.hpp ${stamp}/${machine}_config.hpp
done

# 2. Move all the files with scp

for machine in ${!machines[@]}; do
    (
    sshpass -p "$password" scp config_${machine}.hpp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/include/${config_file}
    sshpass -p "$password" ssh ${user}@${machines[machine]} 'cd /home/wicht/dev/word_spotting; make release_debug;'
    ) &
done

wait

# 3. Execute all the scripts and get the output back

for machine in ${!machines[@]}; do
    (
    echo "Start execution on ${machines[machine]}"
    sshpass -p "$password" ssh ${user}@${machines[machine]} "cd ~/dev/word_spotting; rm -rf results/*; ./release_debug/bin/spotter -2 ${options} train ~/datasets/${dataset} ${set} > grid.log ;"
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/grid.log ${stamp}/${machine}.log
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/method_2_${mode}.dat ${stamp}/${machine}.dat
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/global_rel_file ${stamp}/${machine}_train_global_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/global_top_file ${stamp}/${machine}_train_global_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/local_rel_file ${stamp}/${machine}_train_local_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/local_top_file ${stamp}/${machine}_train_local_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/global_rel_file ${stamp}/${machine}_valid_global_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/global_top_file ${stamp}/${machine}_valid_global_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/local_rel_file ${stamp}/${machine}_valid_local_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/local_top_file ${stamp}/${machine}_valid_local_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/3/global_rel_file ${stamp}/${machine}_test_global_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/3/global_top_file ${stamp}/${machine}_test_global_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/3/local_rel_file ${stamp}/${machine}_test_local_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/3/local_top_file ${stamp}/${machine}_test_local_top_file
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_train_global_rel_file ${stamp}/${machine}_train_global_top_file > ${stamp}/${machine}_train_global_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_train_local_rel_file ${stamp}/${machine}_train_local_top_file > ${stamp}/${machine}_train_local_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_valid_global_rel_file ${stamp}/${machine}_valid_global_top_file > ${stamp}/${machine}_valid_global_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_valid_local_rel_file ${stamp}/${machine}_valid_local_top_file > ${stamp}/${machine}_valid_local_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_test_global_rel_file ${stamp}/${machine}_test_global_top_file > ${stamp}/${machine}_test_global_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_test_local_rel_file ${stamp}/${machine}_test_local_top_file > ${stamp}/${machine}_test_local_eval
    echo "Execution finished on machine $machine (${machines[machine]})"
    echo "Train results"
    echo "  G-MAP " `${grep} map ${stamp}/${machine}_train_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  G-RP " `${grep} R-prec ${stamp}/${machine}_train_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  L-MAP " `${grep} map ${stamp}/${machine}_train_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  L-RP " `${grep} R-prec ${stamp}/${machine}_train_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "Valid results"
    echo "  G-MAP " `${grep} map ${stamp}/${machine}_valid_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  G-RP " `${grep} R-prec ${stamp}/${machine}_valid_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  L-MAP " `${grep} map ${stamp}/${machine}_valid_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  L-RP " `${grep} R-prec ${stamp}/${machine}_valid_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "Test results"
    echo "  G-MAP " `${grep} map ${stamp}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  G-RP " `${grep} R-prec ${stamp}/${machine}_test_global_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  L-MAP " `${grep} map ${stamp}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    echo "  L-RP " `${grep} R-prec ${stamp}/${machine}_test_local_eval | ${grep} all | ${grep} -v cv1_ | cut -f3`
    ) &
done

wait

# 4. Final summary

echo "All machines have finished"
cd ..
bash ./summary.sh ${stamp}