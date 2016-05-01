#!/usr/bin/env bash
logdir=/home/prastog3/projects/neural-context/results/transducer_deeplog_test
fold=0
regex='^ *Set args.saveto= \K(.*)$'
awk_filter () { awk '{if($2!=$3){print $0}}' $( dirname $1 )/current.test.txt > $2 ; }
set -x
for task in 13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA
do
    test_file_name=$( grep -oP "$regex" $logdir/finaltest.transducer_deep.${task}.$fold.o* ; )
    awk_filter $test_file_name ../../log/errors_transducer_deep.$task.$fold
done

for lang in basque english irish tagalog
do
    test_file_name=$( grep -oP "$regex" $logdir/finaltest.lemmatizefold.$lang.$fold.* )
    awk_filter $test_file_name ../../log/errors_lemmatizer_deep.$lang.$fold
done
