#!/bin/bash

machines=(160.98.22.21 160.98.22.22 160.98.22.23 160.98.22.24 160.98.22.25 160.98.22.8 160.98.22.9)
user=wicht
password=`cat .passwd`

mkdir -p run
cd run

if [ ! -f stamp ]; then
    echo "1" >> stamp
fi

stamp=`cat stamp`
new_stamp=$((stamp+1))
echo "$new_stamp" > stamp

echo "Stamp: $stamp"

mkdir -p "$stamp"

# 1.Prepare all the files

for machine in ${!machines[@]}; do
    echo "//This file will be put in ${machines[machine]}" > config_${machine}.hpp
    cat ../include/config_third.hpp >> config_${machine}.hpp
    vim config_${machine}.hpp
    cp config_${machine}.hpp ${stamp}/${machine}_config.hpp
done

# 2. Move all the files with scp

for machine in ${!machines[@]}; do
    (
    sshpass -p "$password" scp config_${machine}.hpp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/include/config_third.hpp
    sshpass -p "$password" ssh ${user}@${machines[machine]} 'cd /home/wicht/dev/word_spotting; CXX=g++-4.9 LD=g++-4.9 make release_debug;'
    ) &
done

wait

# 3. Execute all the scripts and get the output back

for machine in ${!machines[@]}; do
    (
    echo "Start execution on ${machines[machine]}"
    sshpass -p "$password" ssh ${user}@${machines[machine]} 'cd ~/dev/word_spotting; rm -rf results/*; ./release_debug/bin/spotter -2 -third train ~/datasets/washington cv3 > grid.log ;'
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/grid.log ${stamp}/${machine}.log
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/method_2_third.dat ${stamp}/${machine}.dat
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/global_rel_file ${stamp}/${machine}_train_global_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/global_top_file ${stamp}/${machine}_train_global_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/local_rel_file ${stamp}/${machine}_train_local_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/1/local_top_file ${stamp}/${machine}_train_local_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/global_rel_file ${stamp}/${machine}_test_global_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/global_top_file ${stamp}/${machine}_test_global_top_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/local_rel_file ${stamp}/${machine}_test_local_rel_file
    sshpass -p "$password" scp ${user}@${machines[machine]}:/home/wicht/dev/word_spotting/results/2/local_top_file ${stamp}/${machine}_test_local_top_file
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_train_global_rel_file ${stamp}/${machine}_train_global_top_file > ${stamp}/${machine}_train_global_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_train_local_rel_file ${stamp}/${machine}_train_local_top_file > ${stamp}/${machine}_train_local_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_test_global_rel_file ${stamp}/${machine}_test_global_top_file > ${stamp}/${machine}_test_global_eval
    ~/dev/trec_eval/trec_eval -q ${stamp}/${machine}_test_local_rel_file ${stamp}/${machine}_test_local_top_file > ${stamp}/${machine}_test_local_eval
    echo "Execution finished on machine $machine (${machines[machine]})"
    echo "Train results"
    echo "  G-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_train_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "  G-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_train_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "  L-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_train_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "  L-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_train_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "Test results"
    echo "  G-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_test_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "  G-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_test_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "  L-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_test_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "  L-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_test_local_eval | /usr/bin/zgrep all | cut -f3`
    ) &
done

wait

# 4. Final summary

echo "All machines have finished"
echo "Final summary:"

for machine in ${!machines[@]}; do
    echo "Machine $machine (${machines[machine]})"
    echo "  Train results"
    echo "    G-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_train_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "    G-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_train_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "    L-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_train_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "    L-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_train_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "  Test results"
    echo "    G-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_test_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "    G-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_test_global_eval | /usr/bin/zgrep all | cut -f3`
    echo "    L-MAP " `/usr/bin/zgrep map ${stamp}/${machine}_test_local_eval | /usr/bin/zgrep all | cut -f3`
    echo "    L-RP " `/usr/bin/zgrep R-prec ${stamp}/${machine}_test_local_eval | /usr/bin/zgrep all | cut -f3`
done
