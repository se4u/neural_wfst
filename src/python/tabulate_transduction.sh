#!/usr/bin/env bash
# A trap is a piece of code that the shell would execute on receiveing a signal.
# Here we telling bash to exit when it receives SIGINT.
trap "exit" INT
echo '   Task     ,         f~1         ,         f~2       ,          f~3       ,          f~4       ,        f~5         , average'
# This receives a validation score in the 8th column,
# And after the best row according to that column, it prints info about that row.
export awk_get_col8_max='BEGIN{m=0;be=""; te=""}{if($8 >= m){m=$8; be=$11; te=$14}}END{printf "%.2f be=%s,te=%s\n", m, be, te}'
# This averages the score in the first column that it receives and
# It prints the rows that it receives in a single column.
export awk_average='BEGIN{a=0.0; c=0;}{a += $1; c += 1; str2 = sprintf("(%s)", $2); printf "%.2f%-16s", $1, str2}END{print a/c}'
debug=
tabulator () {
    for variant in ${1-transducer_deep}
    do
        echo $variant
        for task in ${2-13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA}
        do
            printf "%-12s, " $task
            for fold in {0..4}
            do
                suffix=${3-/transducer.pkl}
                $debug ./tabulate_model_performance.py --path ../../results/${variant}_task=${task}_fold=${fold}_*${suffix}  2> /dev/null  | awk "$awk_get_col8_max"
            done   | awk "$awk_average"
        done
    done
}
#-----------------------------------------#
# Tabulate the non-swapped model variants #
#-----------------------------------------#
# transducer transducer_swap transducer_drop transducer_drop_swap transducer_drop2
# tabulator transducer_deep ;

#------------------------------------------------#
# Tabulate the train-dev swapped model variants. #
#------------------------------------------------#
# tabulator transducer_deepswap ;

#-------------------------------------------#
# Tabulate the 8 level deep BiLSTMs results #
#-------------------------------------------#
# tabulator transducer_level8

#-------------------------------------------#
# Tabulate the 6 level deep BiLSTMs results #
#-------------------------------------------#
# tabulator transducer_level6

#-----------------------------#
# Tabulate the ablation jobs. #
#-----------------------------#
# tabulator transducer_ablate_1bl ;
# tabulator transducer_ablate_0bl ;
# tabulator transducer_ablate_tying ;
# tabulator transducer_ablate_windowing ;
# tabulator transducer_ablate_1l ;
# tabulator transducer_ablate_4l ;
# tabulator transducer_ablate_staggerexternal
# tabulator transducer_ablate_externalandcopyatmax
# tabulator transducer_ablate_0context0bl
# tabulator transducer_ablate_staggerexternalmult
# tabulator transducer_ablate_staggerextendedmult
# tabulator transducer_ablate_stagger.external.mult.1bl
# tabulator transducer_ablate_stagger.external.inplu.mult.1bl
# tabulator transducer_ablate_runbilstmseparately
# tabulator transducer_ablate_condition_copyval_on_state
# tabulator transducer_ablate_condition_copyval_on_arc

#-------------------#
# Reviewers' Choice #
#-------------------#
# tabulator transducer_ablate_tie_copyemb_in_tensor
# tabulator transducer_ablate_penalty_dropout

# tabulator transducer_simple_decomp_jason
# tabulator transducer_tensor_decomp_ta_h_prod
# tabulator transducer_tensor_decomp_ta_h_prodrelu

#------------------------------#
# Jason's Encoding Suggestions #
#------------------------------#
# tabulator transducer_full_decomp_jason_win1_amach rP-pA
# tabulator transducer_control_decomp_jason_win1
# tabulator transducer_full_decomp_jason_win3_amach # rP-pA
# tabulator transducer_control_decomp_jason_win3


# cm () { grep -h "'epoch_id:', $3" ../../results/transducer_deeplog/transducer_control_decomp_jason_win1_$1.$2.*  \
#               | rev | cut -c 2- | rev \
#               | awk 'BEGIN{a=-1}{if(a == -1){a=$NF}else{a= (a>$NF)?a:$NF; }}END{print a}'; }

# echo $( cm 13SIA-13SKE 0 173; cm 13SIA-13SKE 1 184; cm 13SIA-13SKE 2 10; cm 13SIA-13SKE 3 264 ; cm 13SIA-13SKE 4 7)
# echo $( cm 2PIE-13PKE  0 83 ; cm 2PIE-13PKE  1 23 ; cm 2PIE-13PKE  2 28; cm 2PIE-13PKE  3 25  ; cm 2PIE-13PKE  4 69 )
# echo $( cm 2PKE-z      0 118; cm 2PKE-z      1 245; cm 2PKE-z      2 122;cm 2PKE-z      3 69  ; cm 2PKE-z      4 14 )
# echo $( cm rP-pA       0 45 ; cm rP-pA       1 40 ; cm rP-pA       2 96 ;cm rP-pA       3 121 ; cm rP-pA       4 471)

# cm () { grep -h "'epoch_id:', $3" ../../results/transducer_deeplog/transducer_control_decomp_jason_win3_$1.$2.*  \
#               | rev | cut -c 2- | rev \
#               | awk 'BEGIN{a=-1}{if(a == -1){a=$NF}else{a= (a>$NF)?a:$NF; }}END{print a}'; }

# echo $( cm 13SIA-13SKE 0 353 ; cm 13SIA-13SKE 1 342; cm 13SIA-13SKE 2 151; cm 13SIA-13SKE 3 98; cm 13SIA-13SKE 4 103; )
# echo $( cm 2PIE-13PKE 0 202  ; cm 2PIE-13PKE 1 59;   cm 2PIE-13PKE 2 144;  cm 2PIE-13PKE 3 244; cm 2PIE-13PKE 4 151;  )
# echo $( cm 2PKE-z 0 180      ; cm 2PKE-z 1 156;      cm 2PKE-z 2 163;      cm 2PKE-z 3 530;     cm 2PKE-z 4 337;      )
# echo $( cm rP-pA 0 268       ; cm rP-pA 1 253;       cm rP-pA 2 203;       cm rP-pA 3 370;      cm rP-pA 4 267;       )

#--------------------#
# My Encoding Method #
#--------------------#
tabulator transducer_my_decomp_h10_win3
tabulator transducer_my_decomp_h5_win3
tabulator transducer_deep_my_decomp_h10_win3
# tabulator transducer_my_decomp_goodinit_h10_win3
# tabulator transducer_my_decomp_goodinit_h100_win3
# tabulator transducer_my_decomp_h100_win3
#-----------------------------------#
# Tabulate the training curve jobs. #
#-----------------------------------#
# for samples in 50 100 300
# do
#     echo 'sample=' $samples
#     tabulator transducer_lcurve "2PKE-z 13SIA-13SKE" "_samples=${samples}_*/transducer.pkl" ;
# done

#------------------------------------#
# Tabulate the lemmatization models. #
#------------------------------------#
# variant=lemmatizer_deep
# variant=lemmatizefold
# debug=
# for lang in basque # english irish tagalog
# do
#     printf "%-10s" $lang
#     for fold in {0..9}
#     do
#         $debug ./tabulate_model_performance.py --path ../../results/${variant}_lang=${lang}_fold=${fold}_*/transducer.pkl 2> /dev/null   | awk "$awk_get_col8_max"
#     done  | awk "$awk_average"
# done
