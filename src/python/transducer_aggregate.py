#!/usr/bin/env python
'''
| Filename    : transducer_aggregate.py
| Description : Aggregate the predictions of pretrained transducer models.
| Author      : Pushpendre Rastogi
| Created     : Thu Dec 17 17:43:20 2015 (-0500)
| Last-Updated: Fri Dec 18 01:08:46 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 54
USAGE: ./transducer_aggregate.py
For example when `pkl_to_combine` are:
['../../results/transducer_3/transducer.pkl',
 '../../results/transducer_4/transducer.pkl',
 '../../results/transducer_lr=.05_Tie_Copy_'
 'Param_lstm_dropout_0.6/transducer.pkl.86.2',]
Then the performance is
('Test F1', 86.4)
'''
from lstm_seqlabel_circuit_compilation import \
    compile_args, load_params_from_pklfile_to_stack_config, set_dropout_to_zero
import rasengan
pkl_to_combine = [
    # ('../../results/transducer_3/transducer.pkl',
    #  dict(wemb1_out_dim=8)),
    # ('../../results/transducer_4/transducer.pkl',
    #  dict(wemb1_out_dim=8)),
    ('../../results/transducer_lr=.05_Tie_Copy_'
     'Param_lstm_dropout_0.6/transducer.pkl.86.2',
     dict(wemb1_out_dim=10)),
    ('../../results/transducer_17/transducer.pkl',
     dict(clippping_value=4.5)),
    ('../../results/transducer_5/transducer.pkl',
     dict()),
    ('../../results/transducer_8/transducer.pkl',
     dict())
]

class Aggregator(object):
    def __init__(self, models, data):
        self.models = models
        self.data = data
    def test_f_classify(self, x):
        predictions = [tuple(model_i.test_f_classify(x))
                       for model_i
                       in self.models]
        return list(rasengan.majority(predictions))

def main():
    import transducer_score
    args = transducer_score.args
    set_dropout_to_zero(args)
    data = transducer_score.data
    #--------------------------#
    # Compile disparate models #
    #--------------------------#
    models = []
    for pkl_fn, changes in pkl_to_combine:
        args_clone = rasengan.Namespace(**args)
        #--------------------#
        # Update args_clone. #
        #--------------------#
        rasengan.warn('NOTE: Seting pretrained_param_pklfile')
        args_clone.pretrained_param_pklfile = pkl_fn
        for (k,v) in changes.items():
            setattr(args_clone, k, v)
            print 'Setting args_clone.%s=%s'%(k,str(v))
        #---------------------#
        # Compile args_clone. #
        #---------------------#
        ttns_i = rasengan.Namespace('ttns').update_and_append_prefix(
            compile_args(args_clone), 'test_')
        load_params_from_pklfile_to_stack_config(
            pkl_fn, ttns_i.test_stack_config)
        models.append(ttns_i)

    #----------------------------#
    # Aggregate disparate model. #
    #----------------------------#
    ttns = Aggregator(models, data)
    #-----------------------------------------------#
    # Test performance of Aggregated decision rule. #
    #-----------------------------------------------#
    with rasengan.debug_support():
        stats_valid = args.validate_predictions_f(
            data.valid_lex,
            data.idx2label,
            args,
            ttns.test_f_classify,
            data.valid_y,
            data.words_valid,
            fn='/combined.valid.txt')
        print 'stats_valid', stats_valid
        # stats_test = args.validate_predictions_f(
        #     data.test_lex,
        #     data.idx2label,
        #     args,
        #     ttns.test_f_classify,
        #     data.test_y,
        #     data.words_test,
        #     fn='/combined.test.txt')
        # print 'stats_test', stats_test


if __name__ == '__main__':
    main()
