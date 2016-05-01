'''
| Filename    : lstm_seqlabel_load_save_model.py
| Description : Functions to load and save trained lstm_seqlabel models
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 19:59:59 2015 (-0400)
| Last-Updated: Mon Dec 28 04:28:00 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 38
'''
import cPickle as pickle
import theano

def save_parameters_to_file(test_stack_config, pickle_name):
    print "Started saving parameters"
    params_to_save = {}
    for k in test_stack_config:
        val = test_stack_config[k]
        if hasattr(val, '__call__'):
            params_to_save[k] = 'Not pickled because it was a function'
        elif hasattr(val, 'dont_pickle') and val.dont_pickle == 1:
            params_to_save[k] = 'I was told not to pickle this'
        elif hasattr(val, 'get_value'):
            params_to_save[k] = val.get_value()
        else:
            params_to_save[k] = val

    with open(pickle_name, 'wb') as f:
        pickle.dump(params_to_save, f, protocol=-1)

    print "Saved parameters %s \n to %s"%(
        ' '.join(params_to_save.keys()), pickle_name)
    return


# def load_parameters_from_file(test_stack_config, pickle_name):
#     if pickle_name is not '':
#         data = pickle.load(open(pickle_name))
#         for k in data:
#             if k in test_stack_config:
#                 v = test_stack_config[k]
#                 if (isinstance(v, theano.tensor.sharedvar.TensorSharedVariable)
#                     and k.startswith('tparam')):
#                     test_stack_config[k].set_value(data[k])
#                 elif (isinstance(v, float) or isinstance(v, int)):
#                     assert v == data[k]
#     return
