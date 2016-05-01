'''
| Filename    : class_chip.py
| Description : The class_chip base class.
| Author      : Pushpendre Rastogi
| Created     : Wed Oct 21 19:02:46 2015 (-0400)
| Last-Updated: Tue Jan  5 07:46:55 2016 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 24
'''
import theano


def name_tv(*params):
    ''' Join the params as string using '_' and also add a unique id,
    since every node in a theano graph should have a unique name.

    Example:
    name_tv(a, b) -> tparam_a_b_1
    '''
    # Attach an attribute to the function itself. This way we can simulate
    # a c static variable.
    if not hasattr(name_tv, "uid"):
        name_tv.uid = 0
    name_tv.uid += 1
    return 'tparam_%s_%d' % ("_".join(params), name_tv.uid)


class Chip(object):

    ''' A Chip object requires name and a param dictionary
    that contains param[name+'_'+out_dim] (This can be a function that depends on the input_dim)
    Other than that it must also contain appropriate initializers for all the parameters.

    The params dictionary is updated to contain 'tparam_<param name>_uid'
    '''

    def __init__(self, name, params=None):
        ''' I set the output dimension of every node in the parameters.
            The input dimension would be set when prepend is called.
            (Since prepend method receives the previous class_chip)
        '''
        self.name = name
        self.debug_tv_list = []
        if params is not None:
            self.out_dim = params[self.kn('out_dim')]
            self.params = params
        return

    def prepend(self, previous_class_chip):
        ''' Note that my input_dim of self = output_dim of previous_class_chip
        Also we keep track of absolute_input (the first input) to the layer.

        When we prepend a class_chip to the current class_chip then we infer the input
        dimension based on the "prepended" class_chip's output dimension and we set
        the previous class_chip attribute as the "prepended" class_chip.

        We also carry over the absolute input theano variable from the "prepended"
        class_chip and call the virtual function "construct" that has been over-ridden
        in a concrete implementation of this class.

        During the call to "construct" an output_tv is created and that output_tv
        is then properly named with a unique id by the call to name_tv.
        '''
        self.previous_class_chip = previous_class_chip
        self.in_dim = previous_class_chip.out_dim
        self.depth = previous_class_chip.depth + 1
        if hasattr(self.out_dim, '__call__'):
            self.out_dim = self.out_dim(self.in_dim)
        self.absolute_input_tv = previous_class_chip.absolute_input_tv
        # self.output_tv = None # Do not set output_tv
        # Let the assert take care of everything
        internal_params = self.construct(previous_class_chip.output_tv)
        # Update the debut_tv_list
        self.debug_tv_list = previous_class_chip.debug_tv_list + self.debug_tv_list
        assert self.output_tv is not None
        self.output_tv.name = name_tv(self.name, 'output_tv')
        self.params['chipoutput_' + self.output_tv.name] = self.output_tv
        # register_params(*internal_params)
        for k in internal_params:
            k.block_update = self.prm('block_update')
            assert k.name is not None
            self.params[k.name] = k
        return self

    def prm(self, name):
        return self.params[self.kn(name)]

    def construct(self, input_tv):
        ''' Note that input_tv = previous_class_chip.output_tv
        This method returns a dictionary of internal weight params
        and This method sets self.output_tv
        '''
        raise NotImplementedError

    def kn(self, thing, thing_is_matrix=False):
        ''' kn stands for key name.
        We create a name for the array or hyperparameter that needs to be stored
        as part of this class_chip.
        '''
        # If len(thing) == 1, then its most probably ['U', 'W', 'b', 'T', 'N']
        # (some Matrix)
        if len(thing) == 1 or thing_is_matrix:
            keyname_suffix = '_initializer'
        else:
            keyname_suffix = ''
        return self.name + '_' + thing + keyname_suffix

    def needed_key(self):
        return self._needed_key_impl()

    def _needed_key_impl(self, *things):
        return [self.kn(e) for e in ['out_dim'] + list(things)]

    def _declare_mat(self, name, *dim, **kwargs):
        '''
        Params
        ------
        name     : A part of the name of the final theano variable.
        *dim     : The dimensions of the matrix.
        **kwargs : `multiplier`, `is_matrix`, `is_regularizable`
        '''
        if len(kwargs) == 0:
            is_regularizable=True
        elif len(kwargs) == 1:
            is_regularizable=kwargs['is_regularizable']
        else:
            raise NotImplementedError
        # Matrices have a special naming convention where I add
        # an initializer to the name of the matrix. By doing so
        # the `params` object (which is actually just the
        # stack_config) can spit out the appropriate object upon
        # indexing that has an initialize method.
        var_name = self.kn(name, thing_is_matrix=True)
        np_arr = self.params[var_name].initialize(*dim)
        # `name_tv` just ensures that every shared variable created
        # has a unique id and that it starts with the name tparam.
        var = theano.shared(np_arr, name=name_tv(self.name, name))
        # All differentiable parameters have the attribute
        # `is_regularizable`. The value of this attribute may be true
        # or false.
        var.is_regularizable = is_regularizable
        return var
