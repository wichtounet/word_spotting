#!/bin/bash

# Script to run on several machines

machines_one=(160.98.22.10)
machines_seven=(160.98.22.21 160.98.22.22 160.98.22.23 160.98.22.24 160.98.22.25 160.98.22.8 160.98.22.9)
machines_sevenb=(160.98.22.10 160.98.22.11 160.98.22.12 160.98.22.13 160.98.22.14 160.98.22.15 160.98.22.16)
machines_threeb=(160.98.22.17 160.98.22.18 160.98.22.19)
machines_ten=(160.98.22.10 160.98.22.11 160.98.22.12 160.98.22.13 160.98.22.14 160.98.22.15 160.98.22.16 160.98.22.17 160.98.22.18 160.98.22.19)

if [ "$1" == "7" ]; then
    machines=("${machines_seven[@]}")
elif [ "$1" == "1" ]; then
    machines=("${machines_one[@]}")
elif [ "$1" == "7b" ]; then
    machines=("${machines_sevenb[@]}")
elif [ "$1" == "3b" ]; then
    machines=("${machines_threeb[@]}")
elif [ "$1" == "10" ]; then
    machines=("${machines_ten[@]}")
else
    echo "The first parameter must be one of [7,1,3b,7b,10]"
    exit 1
fi

options="-2 -fix"

if [ "$2" == "third" ]; then
    mode="third"
    options="$options -third"
elif [ "$2" == "half" ]; then
    mode="half"
    options="$options -half"
elif [ "$2" == "full" ]; then
    mode="full"
else
    echo "The second parameter must be one of [full,half,third]"
    exit 1
fi

set=$3

dataset="washington"

# Discards the first three parameters
shift 3

while [ "$1" ]
do
    if [ "$1" == "all" ]; then
        options="$options -all"
    fi

    if [ "$1" == "parzival" ]; then
        options="$options -parzival"
        dataset="parzival"
    fi

    if [ "$1" == "iam" ]; then
        options="$options -iam"
        dataset="iam"
    fi

    if [ "$1" == "hmm" ]; then
        options="$options -hmm -htk"
    fi

    shift
done

config_file="config_${mode}.hpp"

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
echo "Options: $options"

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
    sshpass -p "$password" ssh ${user}@${machines[machine]} 'cd /home/wicht/dev/word_spotting; make clean; make -j9 release;'
    ) &
done

wait

# 3. Execute all the scripts and get the output back

for machine in ${!machines[@]}; do
    (
    echo "Start execution on ${machines[machine]}"
    sshpass -p "$password" ssh ${user}@${machines[machine]} "cd ~/dev/word_spotting; rm -rf results/*; ./release/bin/spotter ${options} train ~/datasets/${dataset} ${set} > grid.log ;"
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
bash ./scripts/summary.sh ${stamp}
