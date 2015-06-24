#!/bin/bash

# Bash script to inspect runs and gather information about them

grep=/usr/bin/zgrep

for stamp in `ls -v run`; do
    # Filter directories
    if [ ! -d "run/${stamp}" ]; then
        continue
    fi

    # Make sure there are some logs
    if [ ! -f run/${stamp}/0.log ]; then
        continue
    fi

    resolution=`${grep} resolution run/${stamp}/0.log | head -n 1 | sed -e 's/Use full resolution/full/' | sed -e 's/Use a third of the resolution/third/' | sed -e 's/Use a half of the resolution/half/'`

    cv=`${grep} "Set: " run/${stamp}/0.log | sed -e 's/Set: //' | sed -e 's/^[ \t]*//' -e 's/[ \t]*$//'`

    echo "${stamp}: ${cv} - ${resolution}"
done
