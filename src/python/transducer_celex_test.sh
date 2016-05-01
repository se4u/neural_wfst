#!/usr/bin/env bash
trap "exit" INT
debug=${debug=echo} # Call as (export debug=; ./transducer_celex_test.sh )
Q () { echo " ./transducer_flurry_impl.sh " ; }
awk_get_pklname_by_dev_perf='BEGIN{m=0;f=""}{if($8 >= m){m=$8;f=$2}}END{print f}'
testor () {
    variant=${1-transducer_deep}
    extra_arg=${2-}
    test_fn_basename=${3-test} # This can be dev.
    save_variant=${4-$variant} # Copy variant unless marked as different.
    for task in 13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA
    do
        for fold in {0..4}
        do
            q_cmd=$( Q finaltest.$save_variant.$task.$fold )
            data_dir=$HOME/projects/neural-context/res/celex/$task/0500/$fold
            # Figure out the correct trial using the tabulation script.
            pp_pklfile_paths="../../results/${variant}_task=${task}_fold=${fold}_*/transducer.pkl"
            pp_pklfile=$( ./tabulate_model_performance.py --path $pp_pklfile_paths 2> /dev/null | awk "$awk_get_pklname_by_dev_perf" )
            if [ -z $pp_pklfile ]
            then
                echo PP_Pklfile were $pp_pklfile_paths
                exit 1
            fi
            $debug $q_cmd train_fn="'$data_dir/train'",dev_fn="'$data_dir/dev'",test_fn="'$data_dir/$test_fn_basename'",pretrained_param_pklfile="'$pp_pklfile'",perform_training=0,perform_testing=1,nepochs=-1,folder="'../../results/transtest_${save_variant}_task=${task}_fold=${fold}',$extra_arg" 1
            $debug sleep 5
        done
    done
}
export -f testor ;

if [ ] ; then
#-----------------------------------------#
#     Test the non-swapped model variants #
#-----------------------------------------#
testor ;

#------------------------------------------------#
#     Test the train-dev swapped model variants. #
#------------------------------------------------#
testor transducer_deepswap ;

#--------------------------#
# Test the ablated models. #
#--------------------------#
testor transducer_ablate_1bl use_1bl=1
testor transducer_ablate_0bl use_0bl=1
testor transducer_ablate_1l use_1l=1
testor transducer_ablate_4l use_4l=1
testor transducer_ablate_windowing win=1
testor transducer_ablate_tying penalty_tie_copy_param=0
# # # testor transducer_ablate_staggerexternal
testor transducer_ablate_externalandcopyatmax "bilstm_externalandcopyatmax=1"

#-----------------------------------#
# Test the crunching based decoding #
#-----------------------------------#
for crunching in 1000 10000 100000; # 1 100;
do
    testor transducer_deep crunching=$crunching dev transducer_deep_26jancrunching$crunching ;
done

#----------------------------------------------#
# Test the 8 layer deep transduction variants. #
#----------------------------------------------#
testor transducer_level8 use_8bl=1;

#----------------------------------------------#
# Test the 6 layer deep transduction variants. #
#----------------------------------------------#
testor transducer_level6 use_6lbl=1;

#--------------------------#
# Test the lrcurve models. #
#--------------------------#
for samples in 50 100 300 # 500
do
    for task in 2PKE-z 13SIA-13SKE
    do
        for fold in 0 1 2 3 4
        do
            variant=transducer_lcurve.$samples
            data_dir=$HOME/projects/neural-context/res/celex/$task/0500/$fold
            q_cmd=$( Q ${variant}.$task.$fold )
            pp_pklfile=$( ./tabulate_model_performance.py --path ../../results/transducer_lcurve_task=${task}_fold=${fold}_trial=*_samples=${samples}_*/transducer.pkl 2> /dev/null | awk "$awk_get_pklname_by_dev_perf"  )
            $debug $q_cmd train_fn="'$data_dir/train'",dev_fn="'$data_dir/dev'",test_fn="'$data_dir/test'",pretrained_param_pklfile="'$pp_pklfile'",perform_training=0,perform_testing=1,nepochs=-1,folder="'../../results/transtest_${variant}_task=${task}_fold=${fold}'" 1
            $debug sleep 5
        done
    done
done


#------------------------------------#
#     Test the lemmatization models. #
#------------------------------------#
variant=lemmatizefold
debug=
for lang in basque english irish tagalog
do
    for fold in {0..9}
    do
        data_dir=$HOME/projects/neural-context/res/wicentowski_split/${lang}-10fold/$fold
        pp_pklfile=$( ./tabulate_model_performance.py --path ../../results/${variant}_lang=${lang}_fold=${fold}_*/transducer.pkl 2> /dev/null | awk "$awk_get_pklname_by_dev_perf" );
        q_cmd=$( Q finaltest.$variant.$lang.$fold )
        $debug $q_cmd train_fn="'$data_dir/train.uniq'",dev_fn="'$data_dir/dev.uniq'",test_fn="'$data_dir/test.uniq'",pretrained_param_pklfile="'$pp_pklfile'",perform_training=0,perform_testing=1,nepochs=-1,folder="'../../results/lemmatest_${variant}_lang=${lang}_fold=${fold}'"   1
        $debug sleep 5
    done
done
fi
