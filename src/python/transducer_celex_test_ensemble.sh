#!/usr/bin/env bash
#-----------------------------------------------------------------------------------#
# The basic idea is to ensemble the predictions of the models on the test portions. #
#-----------------------------------------------------------------------------------#
dirs () {
    task=$1
    fold=$2
    for m in deep  ablate_1l ablate_1bl ablate_externalandcopyatmax ablate_windowing # ablate_0bl ablate_4l ablate_tying
    do
        printf "../../results/transtest_transducer_%s_task=%s_fold=%s_0/current.test.txt  " $m $task $fold ;
    done
}

echo '  Task         f0    f1    f2    f3    f4    Avg'
for task in  13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA
do
    printf "%-14s" $task
    for fold in {0..4}
    do
        path="$( dirs $task $fold )"
        acc=$( ./tabulate_accuracy_of_ensemble_of_predictions.py --path $path | grep Majority | cut -f 3 -d ' ' )
        echo $acc
    done | awk 'BEGIN{a=0.0; c=0;}{a += $1; c += 1; printf "%.3f ", $1}END{print a/c}'
done
