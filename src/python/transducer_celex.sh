#!/usr/bin/env bash
# USAGE: ./transducer_celex.sh | tee ../../log/transducer_celex_q.log
# Cleanup : rm ~/projects/neural-context/results/transducer_deeplog/* ; rm -rf ~/projects/neural-context/results/transducer_deep_*; ~/projects/neural-context/results/lemmatizer_deep_*
# Alternative way to avoid slow machines. "![ah]0[123457]"
debug=
Q () { echo " echo ./transducer_flurry_impl.sh "; }

submitor () {
    variant=${1-transducer_deep}
    keyval=${2-}
    trials=${3-1 2 3}
    train_basename=${4-train}
    dev_basename=${5-dev}
    for task in 13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA
    do
        for fold in {0..4}
        do
            for trial in $trials
            do
                q_cmd=$( Q ${variant}_$task.$fold.$trial )
                data_dir=$PWD/../../res/celex/$task/0500/$fold
                $debug $q_cmd train_fn="'$data_dir/$train_basename'",dev_fn="'$data_dir/$dev_basename'",test_fn="'$data_dir/test'",folder="'../../results/${variant}_task=${task}_fold=${fold}_trial=${trial}',${keyval}" 1
            done
        done
    done
}
export -f submitor;
export -f Q;

if [ ] ; then
# Submit the main transduction jobs.
submitor

# Submit the 8 level deep LSTM jobs.
submitor transducer_level6 use_8bl=1

# Submit the 6 level deep LSTM jobs.
submitor transducer_level6 use_6bl=1

# Submit the MonoLSTM ablation jobs.
submitor transducer_ablate_1l use_1l=1 1
submitor transducer_ablate_4l use_4l=1 1
submitor transducer_ablate_1bl use_1bl=1 1
submitor transducer_ablate_0bl use_0bl=1 1
submitor transducer_ablate_tying penalty_tie_copy_param=0 1
submitor transducer_ablate_windowing win=1 1
submitor transducer_ablate_staggerexternal "bilstm_stagger_schedule='external'" 1
submitor transducer_ablate_externalandcopyatmax "bilstm_externalandcopyatmax=1" 1
submitor transducer_ablate_0context0bl "use_0bl=1,win=1" 1
submitor transducer_ablate_staggerextendedmult "bilstm_stagger_schedule='extended_multiplicative'" 1
submitor transducer_ablate_stagger.external.mult.1bl "bilstm_stagger_schedule='external_multiplicative',use_1bl=1" 1
submitor transducer_ablate_stagger.external.inplu.mult.1bl "bilstm_stagger_schedule='external_inplusive_multiplicative',bilstm_forward_out_dim=30,use_1bl=1" 1
submitor transducer_ablate_runbilstmseparately "bilstm_runbilstmseparately=1" 1
submitor transducer_ablate_condition_copyval_on_state "penalty_condition_copyval_on_state=1" 1
submitor transducer_ablate_condition_copyval_on_arc "penalty_condition_copyval_on_arc=1" 1
submitor transducer_ablate_penalty_dropout "penalty_do_dropout=1,penalty_dropout_retention_freq=0.8" 1
submitor transducer_ablate_tie_copyemb_in_tensor "penalty_tie_copy_param=0,penalty_tie_copyemb_in_tensor=1" 1
# JOB TO SWAP TRAIN WITH TEST
submitor transducer_deepswap "" "1 2 3" dev train

submitor transducer_simple_decomp_jason "penalty_simple_decomp_jason=1" 1
submitor transducer_tensor_decomp_ta_h_prod "penalty_tensor_decomp_ta_h_prod=1" 1
submitor transducer_tensor_decomp_ta_h_prodrelu "penalty_tensor_decomp_ta_h_prodrelu=1" 1
submitor transducer_tensor_decomp_t_a_h_prod "penalty_tensor_decomp_t_a_h_prod=1"
submitor transducer_control_decomp_jason_win3 "use_1bl=1,bilstm_stagger_schedule='external',win=3"
submitor transducer_control_decomp_jason_win1 "use_1bl=1,bilstm_stagger_schedule='external',win=1"
submitor transducer_full_decomp_jason_win1_amach "penalty_full_decomp_jason=1,use_1bl=1,bilstm_stagger_schedule='external',win=1"
submitor transducer_full_decomp_jason_win3_amach "penalty_full_decomp_jason=1,use_1bl=1,bilstm_stagger_schedule='external',win=3"


submitor transducer_my_decomp_h10_win3 "penalty_my_decomp=1,penalty_my_decomp_h_dim=10,use_1bl=1,win=3"
submitor transducer_deep_my_decomp_h10_win3 "penalty_my_decomp=1,penalty_my_decomp_h_dim=10,win=3"
submitor transducer_my_decomp_h5_win3 "penalty_my_decomp=1,penalty_my_decomp_h_dim=5,use_1bl=1,win=3"
submitor transducer_my_decomp_h100_win3 "penalty_my_decomp=1,penalty_my_decomp_h_dim=100,use_1bl=1,win=3"
submitor transducer_my_decomp_goodinit_h10_win3 "penalty_my_decomp=1,penalty_my_decomp_h_dim=10,use_1bl=1,win=3" 1
submitor transducer_my_decomp_goodinit_h100_win3 "penalty_my_decomp=1,penalty_my_decomp_h_dim=100,use_1bl=1,win=3" 1
fi

#-----------------------------#
# Submit training curve jobs. #
#-----------------------------#
if [ ] ; then
for task in 2PKE-z 13SIA-13SKE
do
    for fold in 0 1 2 3 4
    do
        for trial in 1 2
        do
            for samples in 50 100 300 # 500
            do
                q_cmd=$( Q "" transducer_lcurve_$task.$fold.$trial.$samples )
                data_dir=$HOME/projects/neural-context/res/celex/$task/0500/$fold
                $debug $q_cmd train_fn="'$data_dir/train'",dev_fn="'$data_dir/dev'",test_fn="'$data_dir/test'",folder="'../../results/transducer_lcurve_task=${task}_fold=${fold}_trial=${trial}_samples=${samples}'",limit_corpus=$samples $threads
                $debug sleep 5
            done
        done
    done
done
fi

#----------------------------#
# Submit Lemmatization Jobs. #
#----------------------------#
# for lang in basque english irish tagalog
# do
#     for fold in {0..9}
#     do
#         data_dir=$HOME/projects/neural-context/res/wicentowski_split/${lang}-10fold/$fold
#         for trial in {1..3}
#         do
#             q_cmd=$(Q lemmatizefold.$lang.$fold.$trial)
#             $debug $q_cmd train_fn="'$data_dir/train.uniq'",dev_fn="'$data_dir/dev.uniq'",test_fn="'$data_dir/test.uniq'",folder="'../../results/lemmatizefold_lang=${lang}_fold=${fold}_trial=${trial}'" 1
#             $debug sleep 5
#         done
#     done
# done
