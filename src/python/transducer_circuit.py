# -*- coding: utf-8 -*-
'''
| Filename    : transducer_circuit.py
| Description : A class to hold transducer related theano circuitry.
| Author      : Pushpendre Rastogi
| Created     : Fri Dec  4 18:42:33 2015 (-0500)
| Last-Updated: Fri Apr  8 13:42:59 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 124
'''
import class_chip
import theano
from lstm_seqlabel_circuit import dropout_mask_creator
import numpy
from class_serializable_lambda import SerializableLambda, RELU_FN, SIGMOID_FN, TANH_FN
SUBSTITUTION = 0
DELETION = 1
INSERTION = 2

#----------------------------------------#
# Define the Transducer Scoring Function #
#----------------------------------------#
class Penalty(class_chip.Chip):
    def _declare_mat(self, name, np_arr, is_regularizable=True, clip_project=False):
        var = theano.shared(
            np_arr, name=class_chip.name_tv(self.name, name))
        var.is_regularizable = is_regularizable
        self.tv_list.append(var)
        if clip_project:
            var.clip_gradient = self.prm('clip_gradient')
            var.l2_project = self.prm('l2_project')
            var.l2_projection_axis = 0
        return var

    def tying_helper(self):
        ''' The output is tensor of shape [len(str) + 1, sigma, sigma, 3]
        What I need to do is to generate two sequences of
        indices that are `len(str)` long that can be used to index
        the one-th and the two-th dimensions of the tensor
        respectively. Using these sequences I can set the weight
        of these specific locations higher by a tunable
        parameter.
        '''
        #---------------------------------------------------------#
        # First: we specify the string that was actually given as #
        # input. because of the way I feed input to my neural net #
        # the string is actually one longer than the string that  #
        # is actually fed to the final transducer that consumes   #
        # the tensor.                                             #
        #---------------------------------------------------------#
        if self.absolute_input_tv.ndim == 1:
            in_str = self.absolute_input_tv[1:]
        elif self.absolute_input_tv.ndim == 2:
            in_str = self.absolute_input_tv[1:, self.prm('mid_col')]
        else:
            raise NotImplementedError
        #----------------------------------------------#
        # The zeroth dimension is range(1, len(str)+1) #
        #----------------------------------------------#
        zeroth_idi = theano.tensor.arange(1, in_str.shape[0] + 1)
        #------------------------------------------------------#
        # The oneth dimension is zero concatenated to str[:-1] #
        #------------------------------------------------------#
        th_zero = numpy.array([0], dtype='int32')
        oneth_idi = theano.tensor.concatenate((th_zero, in_str[:-1]), axis=0)
        #-----------------------------#
        # The two-th dimension is str #
        #-----------------------------#
        twoth_idi = in_str
        return zeroth_idi, oneth_idi, twoth_idi

    def tie_copyemb_helper(self, input_tv, output_tv):
        copy_emb = self._declare_mat(
            'copy_emb',
            numpy.array([0.01]* self.in_dim, dtype='float32'),
            is_regularizable=True)
        input_str_embedding = input_tv[1:]
        input_str_embedding.name = 'input_str_embedding'
        copy_arr = theano.tensor.dot(input_str_embedding, copy_emb)
        zeroth_idi, oneth_idi, twoth_idi = self.tying_helper()
        output_tv = theano.tensor.set_subtensor(
            output_tv[zeroth_idi,
                      oneth_idi,
                      twoth_idi,
                      SUBSTITUTION],
            copy_arr)
        return output_tv

    def tie_copy_param_helper(self, output_tv):
        ''' We can specify whether we want state specific tying, arc
        specific tying or just a single number that is used to
        tie the copy actions.
        '''
        zeroth_idi, oneth_idi, twoth_idi = self.tying_helper()
        if (self.prm('condition_copyval_on_state')
            or self.prm('condition_copyval_on_arc')):
            copy_val = self._declare_mat(
                'copy_val',
                numpy.array([3.0]* self.prm('vocsize'), dtype='float32'),
                is_regularizable=False)
            copy_arr = (copy_val[oneth_idi]
                        if self.prm('condition_copyval_on_state')
                        else copy_val[twoth_idi])
        else:
            copy_val = self._declare_mat(
                'copy_val', numpy.array(3.0, dtype='float32'),
                is_regularizable=False)
            copy_arr = copy_val
            pass
        output_tv = theano.tensor.set_subtensor(
            output_tv[zeroth_idi,
                      oneth_idi,
                      twoth_idi,
                      SUBSTITUTION],
            copy_arr)
        return output_tv

    def get_arr_ta(self, vec_dim=None):
        arr_ta = - numpy.random.rand(
            (self.in_dim if vec_dim is None else vec_dim),
            self.prm('vocsize'), # t
            3 # a
        ).astype('float32')
        T_ta = self._declare_mat('T_ta', arr_ta, clip_project=True)
        T_tad = T_ta.dimshuffle((0, 'x', 1, 2))
        return T_ta, T_tad

    def get_arr_h(self):
        arr_h = - numpy.random.rand(
            self.in_dim,
            self.prm('vocsize'), # h
        ).astype('float32')
        T_h = self._declare_mat('T_h', arr_h, clip_project=True)
        T_hd = T_h.dimshuffle((0, 1, 'x', 'x'))
        return T_h, T_hd

    def get_score_hta(self):
        arr_hta = - numpy.random.rand(
            self.prm('vocsize'), # h
            self.prm('vocsize'), # t
            3).astype('float32') # a
        T_hta = self._declare_mat('T_hta', arr_hta, clip_project=True)
        T_htad = T_hta.dimshuffle(('x', 0, 1, 2))
        return T_hta, T_htad

    def get_arr_hta(self):
        arr = - numpy.random.rand(
            self.in_dim,
            self.prm('vocsize'), # Previous Chracter h
            self.prm('vocsize'), # Current Character h'
            3).astype('float32') # Edit Type
        #-----------------------------------------------------------#
        # Set the embedding for unused action combinations to zero. #
        #-----------------------------------------------------------#
        arr[:, :, 0, SUBSTITUTION] = 0
        arr[:, :, 1:, DELETION] = 0
        arr[:, :, 0, INSERTION] = 0
        return arr

    def composition_logic_ta_h_prod(self, input_tv, relu=False):
        assert not self.prm('do_dropout')
        _, T_tad = self.get_arr_ta()
        _, T_hd = self.get_arr_h()
        if relu:
            T_tmp = T_tad * T_hd
            T_tmp.name = 'T_tmp'
            T_ = RELU_FN(T_tmp)
        else:
            T_ = T_tad * T_hd
        output_tv = theano.tensor.tensordot(input_tv, T_, axes=[1, 0])
        return output_tv

    def composition_logic_ta_h_prodrelu(self, input_tv):
        return self.composition_logic_ta_h_prod(input_tv, relu=True)

    def composition_logic_t_a_h_prod(self, input_tv):
        assert not self.prm('do_dropout')
        arr_t = - numpy.random.rand(
            self.in_dim,
            self.prm('vocsize'), # t
        ).astype('float32')
        T_t = self._declare_mat('T_t', arr_t, clip_project=True)
        T_td = T_t.dimshuffle((0, 'x', 1, 'x'))
        arr_a = - numpy.random.rand(
            self.in_dim,
            3 # a
            ).astype('float32')
        T_a = self._declare_mat('T_a', arr_a, clip_project=True)
        T_ad = T_a.dimshuffle((0, 'x', 'x', 1))
        _, T_hd = self.get_arr_h()
        T_ = T_td * T_ad * T_hd
        output_tv = theano.tensor.tensordot(input_tv, T_, axes=[1, 0])
        return output_tv

    def composition_logic_simple_decomp_jason(self, input_tv):
        assert not self.prm('do_dropout')
        T_ta, _ = self.get_arr_ta()
        in_T_ta = theano.tensor.tensordot(input_tv, T_ta, axes=[1, 0])
        in_T_tad = in_T_ta.dimshuffle((0, 'x', 1, 2))
        _, T_htad = self.get_score_hta()
        output_tv = T_htad + in_T_tad
        return output_tv

    def composition_logic_full_decomp_jason(self, input_tv):
        # e = M <type,s,t,abs(t-s)>
        # γ = <αβ, h>
        # γ^T M e
        # e = M <(type,t), s, abs(s-(type,t))>
        # γ = <αβ, h>
        # γ^T M e
        assert self.prm('full_decomp_jason')
        # arr_t = - numpy.random.rand(self.in_dim, self.prm('vocsize')).astype(
        #     'float32')
        # arr_a = - numpy.random.rand(self.in_dim, 3).astype('float32')
        # T_t = self._declare_mat('T_t', arr_t, clip_project=True)
        # T_a = self._declare_mat('T_a', arr_a, clip_project=True)
        T_s = self.previous_class_chip.previous_class_chip.output_tv
        T_s_dim = self.previous_class_chip.previous_class_chip.out_dim

        arr_M = numpy.random.rand(self.in_dim, 3 * T_s_dim).astype('float32')
        T_M = self._declare_mat('T_M', arr_M, clip_project=True)

        # First multiply input with T_M
        iptm = theano.tensor.tensordot(input_tv, T_M, axes=[1, 0])
        iptmd = iptm.dimshuffle((0, 'x', 1)) # strlen, 3T_s_dim
        arr_h = - numpy.random.rand(self.prm('vocsize'), 3 * T_s_dim).astype(
            'float32')
        T_h = self._declare_mat('T_h', arr_h, clip_project=True).dimshuffle('x', 0, 1)
        # γ^T M
        T_i_h_vec = iptmd + T_h
        # At each location we have a particular s
        arr_ta = - numpy.random.rand(T_s_dim, self.prm('vocsize'), 3).astype(
            'float32')
        T_ta = self._declare_mat('T_ta', arr_ta, clip_project=True)
        assert T_i_h_vec.ndim == 3
        assert T_s.ndim == 2
        # How to concatenate s with type,t?
        # The memory intensive pattern to create a concatenated thing is:
        # t.concatenate((t.repeat(m2, 2, axis=0).eval(),
        #                t.tile(m, (2, 1)).eval()),
        #               axis=1).eval()
        # Unfortunately the above is useless.
        # First let's see how we can dot product with <ta, s>?
        # The simplest way is through a scan operation !!!!!!
        def _step(mat_h_vec, s_vec, _T_ta):
            s_vec_tile = theano.tensor.tile(
                s_vec, (self.prm('vocsize'), 3, 1)).dimshuffle((2, 0, 1))
            s_T_ta = theano.tensor.concatenate(
                (_T_ta, theano.tensor.abs_(_T_ta - s_vec_tile), s_vec_tile),
                axis=0)
            # Finally I need i-h-t-a and that's what I get
            h_t_a_tv = theano.tensor.tensordot(mat_h_vec, s_T_ta, axes=[1, 0])
            return h_t_a_tv
        output_tv, _ = theano.scan(_step,
                                   sequences=[T_i_h_vec, T_s],
                                   outputs_info=None,
                                   non_sequences=[T_ta],
                                   name='full_decomp_jason_scan',
                                   strict=True)
        return output_tv

    def composition_logic_my_decomp(self, input_tv):
        # For each h learn a matrix. Then multiply matrix with input.
        arr_h = (numpy.random.rand(self.in_dim, self.prm('vocsize'), self.prm('my_decomp_h_dim')).astype('float32') - 0.5) * 0.01
        T_h = self._declare_mat('T_h', arr_h, clip_project=True)
        input_tv_h = theano.tensor.tensordot(input_tv, T_h, axes=[1, 0])
        # Now we have a vector per (input,h) pair
        T_ta, _ = self.get_arr_ta(self.prm('my_decomp_h_dim'))
        if self.prm('do_dropout'):
            T_ta.dropout_retention_freq = self.prm('dropout_retention_freq')
        # Then take dot product of (input_tv,h,vec) with (vec, ta)
        output_tv = theano.tensor.tensordot(input_tv_h, T_ta, axes=[2, 0])
        return output_tv

    def composition_logic_default(self, input_tv):
        arr_hta = self.get_arr_hta()
        T_ = self._declare_mat('T', arr_hta, clip_project=True)
        if self.prm('do_dropout'):
            T_.dropout_retention_freq = self.prm('dropout_retention_freq')
        output_tv = theano.tensor.tensordot(input_tv, T_, axes=[1, 0])
        return output_tv

    def composition_logic(self, input_tv):
        if self.prm('tensor_decomp_ta_h_prod'):
            return self.composition_logic_ta_h_prod(input_tv)
        elif self.prm('tensor_decomp_ta_h_prodrelu'):
            return self.composition_logic_ta_h_prodrelu(input_tv)
        elif self.prm('tensor_decomp_t_a_h_prod'):
            return self.composition_logic_t_a_h_prod(input_tv)
        elif self.prm('simple_decomp_jason'):
            return self.composition_logic_simple_decomp_jason(input_tv)
        elif self.prm('full_decomp_jason'):
            return self.composition_logic_full_decomp_jason(input_tv)
        elif self.prm('my_decomp'):
            return self.composition_logic_my_decomp(input_tv)
        else:
            return self.composition_logic_default(input_tv)

    def tying_logic(self, input_tv, output_tv):
        #----------------------------#
        # Logic for Tying Parameters #
        #----------------------------#
        if self.prm('tie_copyemb_in_tensor') or self.prm('tie_copy_param'):
            assert not (
                self.prm('tie_copyemb_in_tensor') and self.prm('tie_copy_param')), \
                "Tying parameter by both embedding, and by score is probably an error."
            output_tv = (self.tie_copy_param_helper(output_tv)
                         if self.prm('tie_copy_param')
                         else self.tie_copyemb_helper(input_tv, output_tv))
        else:
            pass
        return output_tv

    def construct(self, input_tv):
        '''
        The first position of W encodes the position in the *input* string.
        The second position encodes the character in the lower part of the transduction,
        the third position encodes the character in output part of the transduction.
        The fourth position encodes whether it is a insertion, deletion or substitution.

        this chip produces a score that is related to the
        (position in input string, lower_character, upper_character, action)
        This simple architecture simply embeds all the 3Σ^2 combinations of
        (lower char, upper char, action) individually and then takes its dot product
        with the embedding of the character at the ith position.

        Params
        ------
        input_tv : A 2D matrix of shape (n_characters, vec_dim)

        Returns
        -------
        '''
        assert input_tv.ndim == 2
        self.tv_list = []
        # We only need to change input_tv during dropout.
        if self.prm('do_dropout'):
            dropout_mask = dropout_mask_creator(
                self.in_dim, self.prm('dropout_retention_freq'))
            input_tv = input_tv * dropout_mask
            pass
        output_tv = self.composition_logic(input_tv)
        assert output_tv.ndim == 4
        output_tv = self.tying_logic(input_tv, output_tv)
        self.output_tv = output_tv
        return tuple(self.tv_list)



    def needed_key(self):
        return self._needed_key_impl(
            'T', 'clip_gradient', 'l2_project', 'do_dropout',
            'dropout_retention_freq', 'tie_copy_param',
            'condition_copyval_on_state', 'condition_copyval_on_arc',
            'vocsize', 'tie_copyemb_in_tensor')
