'''
| Filename    : class_stack_config.py
| Description : Stack Configuration class
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 20:14:51 2015 (-0400)
| Last-Updated: Fri Apr  8 11:42:31 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 36
'''
import collections
from class_serializable_lambda import SerializableLambda
from class_array_init import ArrayInit
from rasengan import Namespace

class StackConfig(collections.MutableMapping):

    '''A dictionary like object that would automatically recognize
    keys that end with the following pattern and return appropriate keys.
    _out_dim  :
    _initializer :
    The actions to take are stored in a list for easy composition.
    '''
    actions = [
        (lambda key: key.endswith('_out_dim'),
         SerializableLambda('lambda x: x')),
        (lambda key: key.endswith('_T_initializer'),
         ArrayInit(ArrayInit.onesided_uniform)),
        (lambda key: key.endswith('_t_initializer'),
         ArrayInit(ArrayInit.onesided_uniform)),
        (lambda key: key.endswith('_U_initializer'),
         ArrayInit(ArrayInit.ortho, multiplier=1)),
        (lambda key: key.endswith('_W_initializer'),
         ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
        (lambda key: key.endswith('_N_initializer'),
         ArrayInit(ArrayInit.normal)),
        (lambda key: key.endswith('_Y_initializer'),
         ArrayInit(ArrayInit.normal)),
        (lambda key: key.endswith('_b_initializer'),
         ArrayInit(ArrayInit.zero)),
        (lambda key: key.endswith('_P_initializer'),
         ArrayInit(ArrayInit.penalty, multiplier=0.01)),
        (lambda key: key.endswith('_viterbi_p'), True),
        (lambda key: key.endswith('_decode_type'), 'viterbi'),
        (lambda key: key.endswith('_learn_l'), 0),
        (lambda key: key.endswith('_reg_weight'), 0),
        (lambda key: key.endswith('_begin'), 1),
        (lambda key: key.endswith('_end'), -1),
        (lambda key: key.endswith('_activation_fn'),
         SerializableLambda('lambda x: x + theano.tensor.abs_(x)')),
        (lambda key: key.endswith('_forward_go_backwards'), False),
        (lambda key: key.endswith('_backward_go_backwards'), True),
        # We set the default initializer to NotImplemented
        # to make sure that we set it explicitly
        # Ideally the multiplier should be 1/dimension so that
        # we end up taking an average.
        (lambda key: key.endswith('_v_initializer'),
         ArrayInit(ArrayInit.ones, multiplier=0.001)),
        (lambda key: key.endswith('_l_initializer'),
         ArrayInit(ArrayInit.zero)),
        (lambda key: key.endswith('_pool_size'), 5),
        (lambda key: key.endswith('_penalty_vector_max_length'), 100),
        (lambda key: key.endswith('_do_dropout'), 0),
        (lambda key: key.endswith('_dropout_retention_freq'), 0.5),
        (lambda key: key.endswith('_add_bias'), 1),
        (lambda key: key.endswith('_block_update'), 0),
        (lambda key: key == 'multi_embed', 0),
        (lambda key: key == 'multi_embed_count', -1),
        (lambda key: key == 'batch_input', 0),
        (lambda key: key == 'cautious_update', 0),
        (lambda key: key == 'endpoint', None),
        (lambda key: key == 'on_unused_input', 'raise'),
        (lambda key: key.endswith('stagger_schedule'), 'extended'),
        (lambda key: key.endswith('chop_bilstm'), 0),
        (lambda key: key.endswith('tie_copy_param'), 0),
        (lambda key: key.endswith('do_backward_pass'), 1),
        (lambda key: key.endswith('forcefully_copy_embedding_to_output'), 0),
        (lambda key: key.endswith('segregate_bilstm_inputs'), 0),
        (lambda key: key.endswith('condition_copyval_on_arc'), 0),
        (lambda key: key.endswith('condition_copyval_on_state'), 0),
        (lambda key: key.endswith('tie_copyemb_in_tensor'), 0),
        (lambda key: key.endswith('tensor_decomp_ta_h_prod'), 0),
        (lambda key: key.endswith('tensor_decomp_ta_h_prodrelu'), 0),
        (lambda key: key.endswith('simple_decomp_jason'), 0),
        (lambda key: key.endswith('tensor_decomp_t_a_h_prod'), 0),
        (lambda key: key.endswith('penalty_full_decomp_jason'), 0),
        (lambda key: key.endswith('my_decomp_h_dim'), 10),
    ]

    def __init__(self, dictionary):
        self.store = collections.OrderedDict()
        self.update(dictionary)

    def __getitem__(self, key):
        if key in self.store:
            return self.store[key]
        for (predicate, retval) in self.actions:
            if predicate(key):
                return retval
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def reset(self):
        for k in self.store:
            if k.startswith('tparam_'):
                del self.store[k]
        return

    def differentiable_parameters(self):
        return [p for p in self.values() if hasattr(p, 'is_regularizable')]

    def regularizable_parameters(self):
        return [p for p in self.differentiable_parameters() if p.is_regularizable]

    def updatable_parameters(self):
        return [p for p in self.differentiable_parameters() if not p.block_update]

    def __repr__(self):
        return '\n'.join(self.keys())
