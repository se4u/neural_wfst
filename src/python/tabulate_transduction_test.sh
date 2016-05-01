#!/usr/bin/env bash

cd ../../results/transducer_deeplog_test/
awk_average='BEGIN{a=0.0; c=0;}{a += $1; c += 1; printf "%7.1f ", $1}END{print a/c}'
tabulator () {
variant=${1-transducer_deep}
echo $variant
echo '              f~0      f~1     f~2     f~3     f~4  average'
for task in ${2-13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA}
do
    printf "%-12s" $task
    for fold in {0..4}
    do
        prefix=${3-finaltest.}
        files="${prefix}$variant.$task.$fold.o*"
        err=$( tail -n 1 $files ); #echo $err
        acc=$( python -c  "print 100 - $err" 2> /dev/null )
        if [ $? -eq 0 ]
        then
            printf "%f\n" $acc
        else
            echo "-1"
        fi
    done  | awk "$awk_average"
done
}

#-----------------------#
# Tabulate Basic Tasks. #
#-----------------------#
# tabulator ;
#-------------------------------------------------------#
# Tabulate the results of increasing the training data. #
#-------------------------------------------------------#
# tabulator transducer_deepswap ;
#-----------------------------------------------------#
# Tabulate the results of increasing the layers to 8. #
#-----------------------------------------------------#
# tabulator transducer_level8

#-----------------------------------------------------#
# Tabulate the results of increasing the layers to 6. #
#-----------------------------------------------------#
# tabulator transducer_level6

#---------------------------------------#
# Tabulate the results of crunching100. #
#---------------------------------------#
for crunching in 1000 10000 100000;
do
    tabulator transducer_deep_26jancrunching$crunching
done

#--------------------------------------#
# Tabulate the learning curve results. #
#--------------------------------------#
# for samples in 50  100 300 # 500
# do
#     tabulator "transducer_lcurve.$samples" "2PKE-z 13SIA-13SKE"  '';
# done

#----------------------------------------#
# Tabulate the results of lemmatization. #
#----------------------------------------#
# variant=lemmatizefold
# echo 'fold          f0    f1     f2     f3     f4     f5     f6     f7     f8      f9'
# for lang in basque english irish tagalog
# do
#     printf "%-12s" $lang
#     for fold in {0..9}
#     do
#         err=$( tail -n 1 finaltest.$variant.$lang.$fold.o* )
#         printf "%.3f\n" $( bc -l <<< "100 - $err" )
#     done | awk "$awk_average"
# done
