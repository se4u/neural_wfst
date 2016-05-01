# Filename: lstm_seqlabel_circuit.py
# Description:
# Author: Pushpendre Rastogi
# Created: Fri Mar 20 00:22:34 2015 (-0400)
# Last-Updated: Fri Apr 15 14:45:48 2016 (-0400)
#           By: Pushpendre Rastogi
#     Update #: 949
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from class_chip import Chip, name_tv
from lstm_seqlabel_circuit_order_one_crf_decode_and_partition import \
    tagged_sequence_unnormalized_score_in_order_one_crf, \
    find_highest_scoring_sequence_in_order_one_crf, \
    sum_all_paths_of_order_one_crf, \
    find_score_of_viterbi_sequence_in_order_one_crf
import objective
from util_theano import th_floatX, th_reverse, th_logsumexp, th_access_each_row_of_x_by_index, th_choose, th_batch_choose
from util_lstm_seqlabel import np_floatX
from rasengan import Namespace, flatten


def newer(fa, fb):
    ''' Returns true if file fa was last modified after file fb
    '''
    ma = os.path.getmtime(fa)  # Modification time.
    mb = os.path.getmtime(fb)
    return ma > mb


def dropout_mask_creator(dim, retention_freq, seed=None):
    ''' Create a binary vector of dimensionality dim, that is set to one with
    frequency "retention_freq". Note that this dropout_mask is a vector not a
    matrix but it is broadcastable.
    '''
    return th_floatX(
        RandomStreams(seed=seed).binomial(size=(dim,),
                                          p=retention_freq))


class Start(object):

    ''' A start object which has all the necessary attributes that
    any class_chip object that would call it would need.
    '''

    def __init__(self, out_dim, output_tv, depth=0):
        self.out_dim = out_dim
        self.output_tv = output_tv
        self.absolute_input_tv = output_tv
        self.depth = depth
        self.debug_tv_list = []


class Slice(Chip):

    ''' This class_chip simply slices the input along the second dimension.
    This is useful if the sentences are being batched into batches like
    (sentence_idx, token_idx, vector) and we want to slice each sentence
    by it beginning and end token in the batch.
    '''

    def construct(self, input_tv):
        self.output_tv = input_tv[:, self.prm('begin'): self.prm('end'), :]
        return tuple()

    def needed_key(self):
        return self._needed_key_impl('begin', 'end')


class Linear(Chip):

    ''' A Linear Chip is a matrix Multiplication
    It requires a U_initializer

    DROPOUT NOTE:
    # Dropout is a regularization method that was introduced in the paper
    # "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    # Nitish Srivastava et. al. (2014), JMLR.
    # The dropout procedure works as follows:
    # First we randomly set some of the dimensions of the incoming vector to
    # zero before the matrix multiplication with the weights. At test time
    # the weights of this layer are scaled down to pW to account for the
    # fact that the weights in this layer tended to see a thinned input.
    '''

    def construct(self, input_tv):
        '''
        Params
        ------
        input_tv : a matrix of size (n_sentences, n_tokens, vecdim)
        '''
        # N is the linear transformation matrix.
        N = self._declare_mat('N', self.in_dim, self.out_dim)
        N.clip_gradient = self.prm('clip_gradient')
        N.l2_project = self.prm('l2_project')
        N.l2_projection_axis = 0
        if self.prm('do_dropout'):
            N.dropout_retention_freq = self.prm('dropout_retention_freq')
            # Create a dropout mask.
            dropout_mask = dropout_mask_creator(
                self.in_dim, N.dropout_retention_freq)
            # Apply dropout mask to input variable.
            # Note that dropout_mask is a vector and input_tv is a
            # matrix. We are broadcasting this multiplication.
            # Essentially we are dropping entire columns from input_tv.
            dropout_input_tv = (input_tv * dropout_mask)
            dropout_input_tv.name = self.kn('dropout_input_tv')
            # During train time the output is the matrix multiplication of
            # dropped out variables with the matrix.
            transformed_tv = T.tensordot(
                dropout_input_tv, N, axes=[dropout_input_tv.ndim - 1, 0])
        else:
            transformed_tv = T.dot(input_tv, N)

        if self.prm('add_bias'):
            b = self._declare_mat('b', self.out_dim, is_regularizable=False)
            b.l2_project = self.prm('l2_project')
            b.l2_projection_axis = 0
            self.output_tv = transformed_tv + b
            return (N, b)
        else:
            self.output_tv = transformed_tv
            return (N,)

    def needed_key(self):
        return self._needed_key_impl(
            'N', 'do_dropout', 'dropout_retention_freq', 'clip_gradient',
            'l2_project', 'b', 'add_bias')


class MultiEmbedding(Chip):

    def lut_names(self):
        return ['%s_%d_T' % (lut_name, lut_idx)
                for lut_idx, (lut_name, (_, __))
                in enumerate(self.prm('lut_dims'))]

    def initialize_luts(self):
        lut_dims = self.prm('lut_dims')
        lut_list = []
        lut_names = self.lut_names()
        for (_, (lut_in_dim, lut_out_dim)), lut_name in zip(lut_dims, lut_names):
            T_lut = self._declare_mat(
                lut_name,  # name
                lut_in_dim, lut_out_dim,  # *dim
                is_regularizable=True)
            T_lut.clip_gradient = self.prm('clip_gradient')
            T_lut.l2_project = self.prm('l2_project')
            T_lut.l2_projection_axis = 1
            lut_list.append(T_lut)
        return lut_list

    def construct(self, input_tv_list):
        ''' The input_tv_list is a fixed size list of 3D tensor of integers of type
        (sentence_idx, token_idx, features)

        The number of members in the list equals the number of luts instantiated.
        IOW: len(input_tv_list) == len(lut_dims) == len(lut_wins)

        The number of features in each `input_tv` tensor, i.e. the size of the third
        dimension equals the corresponding value of lut_wins.

        Corresponding to each member of the list we create a lookup table that
        contains the vector representation of those discrete feature. The lut
        table dimensions are specified in `lut_dims` along with name of the `lut`.

        The output from each lut is computed by flattening -> embedding -> reshaping
        the windowed input and the final output is created by concatenating the
        representations of each feature.

        A multi embedding uses
        lut_dims = list of (name, (in_dim, out_dim)) of lookup tables.
        lut_wins = list of windowing dimensions used with each lut.
        along with `do_dropout` and `dropout_retention_freq`

        Params
        ------
        input_tv :
        '''
        lut_dims = self.prm('lut_dims')
        lut_wins = self.prm('lut_wins')
        lut_list = self.initialize_luts()
        n_sequences = input_tv_list[0].shape[0]
        n_timesteps = input_tv_list[0].shape[1]
        output_list = []
        out_dim = 0
        for (tv, lut, (_, (__, lut_out_dim)), win) in zip(
                input_tv_list, lut_list, lut_dims, lut_wins):
            lut_dim_contribution = win * lut_out_dim
            # Note that a single lut is a tensor of size.
            # (n_sequences, n_timesteps, win)
            lut_output_tv = lut[tv.flatten()].reshape(
                [n_sequences, n_timesteps, lut_dim_contribution])
            # Mutate lut_output_tv in case of dropout.
            if self.prm('do_dropout'):
                lut.dropout_retention_freq = self.prm('dropout_retention_freq')
                dropout_mask = dropout_mask_creator(
                    lut_dim_contribution, lut.dropout_retention_freq)
                lut_output_tv = lut_output_tv * dropout_mask
            output_list.append(lut_output_tv)
            out_dim += lut_dim_contribution
        self.output_tv = T.concatenate(output_list, axis=2)
        self.out_dim = out_dim
        return lut_list

    def needed_key(self):
        fixed_keys = ['lut_dims', 'lut_wins', 'do_dropout',  # 'out_dim',
                      'dropout_retention_freq', 'clip_gradient', 'l2_project']
        try:
            extra_keys = [self.kn(e, thing_is_matrix=True)
                          for e in self.lut_names()]
        except:
            extra_keys = []

        return ([self.kn(e) for e in fixed_keys] + extra_keys)


class Embedding(Chip):

    ''' An embedding converts  one-hot-vectors to dense vectors.
    For efficiency we don't take dot products with one-hot-vectors and instead
    use this embedding layer instead. Clearly this idea can be extended to a
    class_chip that receives a few different indices into one hot vectors and then
    concatenates or otherwise computes the resultant embeddings.
    This requires a W_initializer.
    '''

    def construct(self, input_tv):
        T_ = self._declare_mat('T', self.in_dim, self.out_dim)
        T_.clip_gradient = self.prm('clip_gradient')
        T_.l2_project = self.prm('l2_project')
        T_.l2_projection_axis = 1
        n_timesteps = input_tv.shape[0]
        window_size = self.prm('win_size')
        self.out_dim = window_size * self.out_dim
        output_tv = T_[input_tv.flatten()].reshape(
            [n_timesteps, self.out_dim])
        if self.prm('do_dropout'):
            T_.dropout_retention_freq = self.prm('dropout_retention_freq')
            dropout_mask = dropout_mask_creator(
                self.out_dim, self.prm('dropout_retention_freq'))
            self.output_tv = output_tv * dropout_mask
        else:
            self.output_tv = output_tv
        return (T_,)

    def needed_key(self):
        return self._needed_key_impl('T', 'do_dropout', 'dropout_retention_freq',
                                     'clip_gradient', 'l2_project')


class Activation(Chip):

    ''' This requires a _activation_fn parameter.
    '''

    def construct(self, input_tv):
        self.output_tv = self.prm('activation_fn')(input_tv)
        return tuple()

    def needed_key(self):
        return self._needed_key_impl('activation_fn')


# class MaxPool(Chip):

#     ''' This class_chip collapses the input tensor by max pooling along its last dimension.
#     '''

#     def construct(self, input_tv):
#         pool_size = self.prm('pool_size')
#         y = T.reshape(input_tv,
#                       ([input_tv.shape[i] for i in range(input_tv.ndim - 1)]
#                        + [T.floor_div(input_tv.shape[input_tv.ndim - 1], pool_size).astype('int32'), pool_size]),
#                       ndim=input_tv.ndim + 1)
#         self.output_tv = T.max(y, axis=y.ndim - 1)
#         return tuple()

#     def needed_key(self):
#         return self._needed_key_impl('pool_size')


class Collapse(Chip):

    ''' This class_chip collapses the input tensor by doing
    a weighted average along its last dimension.
    '''

    def construct(self, input_tv):
        sod2c = self.prm('shape_of_dim_to_collapse')
        v = self._declare_mat('v', sod2c)
        self.output_tv = T.tensordot(input_tv, v, axes=(input_tv.ndim - 1, 0))
        return (v,)

    def needed_key(self):
        return self._needed_key_impl('v', 'shape_of_dim_to_collapse')


class Average(Chip):

    ''' This class_chip collapses the input tensor by doing
    a weighted average along its last dimension.
    '''

    def construct(self, input_tv):
        self.output_tv = input_tv.mean(axis=input_tv.ndim - 1)
        return tuple()


class LSTM(Chip):

    ''' This requires W, U and b initializer.
    '''

    def construct(self, input_tv):
        ''' This LSTM uses 3D tensor inputs of shape
        (n_sentences, n_tokens, token_feature)

        Params
        ------
        input_tv :
        '''
        def __slice_3D(matrix, row_idx, stride):
            return matrix[:, row_idx * stride: (row_idx + 1) * stride]

        def __slice_2D(vector, row_idx, stride):
            return vector[row_idx * stride: (row_idx + 1) * stride]

        #-----------------------------------------------------#
        # Set the __slice function that is called from __step #
        #-----------------------------------------------------#
        if input_tv.ndim == 2:
            __slice = __slice_2D
        elif input_tv.ndim == 3:
            __slice = __slice_3D
        else:
            raise NotImplementedError

        #----------------------------------------------------#
        # The __step function is called from the scan below. #
        #----------------------------------------------------#
        def __step(x_, h_prev, c_prev, U):
            '''
            x_     = Index into a token of the Transformed and Bias incremented Input
                     In case the input_tv is 3D (A batched input) then
                     this is a matrix itself of size n_sentences, self.out_dim,
                     otherwise this is a vector.
            h_prev = previous output of the LSTM (Left output of this function)
            c_prev = previous cell value of the LSTM (Right output of this function)
            U = A projection matrix.

            This is the vanilla version of the LSTM without peephole connections
            See: Section 2, "LSTM: A Search Space Odyssey", Klaus et. al, ArXiv(2015)
            http://arxiv.org/pdf/1503.04069v1.pdf for details.
            '''
            # x_ is a matrix whose first dim corresponds to the
            # sentences in a batch.
            # h_prev is a matrix that is compatible with x_
            # i.e. its first dimension has size n_sentences.
            # its second dimensions contains the features.
            preact = T.dot(h_prev, U) + x_
            i = T.nnet.sigmoid(__slice(preact, 0, self.out_dim))  # Input gate
            f = T.nnet.sigmoid(__slice(preact, 1, self.out_dim))  # Forget gate
            o = T.nnet.sigmoid(__slice(preact, 2, self.out_dim))  # output gate
            z = T.tanh(__slice(preact, 3, self.out_dim))  # block input
            c = f * c_prev + i * z  # cell state
            h = o * T.tanh(c)  # block output # n_sentences, n_features
            return h, c

        W = self._declare_mat('W', self.in_dim, 4 * self.out_dim)
        W.clip_gradient = self.prm('clip_gradient')
        W.l2_project = self.prm('l2_project')
        W.l2_projection_axis = 0

        U = self._declare_mat('U', self.out_dim, 4 * self.out_dim)
        U.clip_gradient = self.prm('clip_gradient')
        U.l2_project = self.prm('l2_project')
        U.l2_projection_axis = 0

        #---------------#
        # Dropout BEGIN #
        #---------------#
        if self.prm('do_dropout'):
            W.dropout_retention_freq = self.prm('dropout_retention_freq')
            dropout_mask = dropout_mask_creator(
                self.in_dim, W.dropout_retention_freq)
            dropout_input_tv = (input_tv * dropout_mask)
            dropout_input_tv.name = self.kn('dropout_input_tv')
            projected_input = T.dot(dropout_input_tv, W).astype(
                theano.config.floatX)
        else:
            projected_input = T.dot(input_tv, W).astype(theano.config.floatX)
        #-------------#
        # Dropout END #
        #-------------#

        if self.prm('add_bias'):
            b = self._declare_mat(
                'b', 4 * self.out_dim, is_regularizable=False)
            projected_input = projected_input + b

        if input_tv.ndim == 3:
            n_sentences = input_tv.shape[0]
            n_steps = input_tv.shape[1]
            projected_input = projected_input.dimshuffle([1, 0, 2])
            outputs_info = [T.alloc(np_floatX(0.), n_sentences, self.out_dim),
                            T.alloc(np_floatX(0.), n_sentences, self.out_dim)]
        elif input_tv.ndim == 2:
            n_steps = input_tv.shape[0]
            outputs_info = [T.alloc(np_floatX(0.), self.out_dim),
                            T.alloc(np_floatX(0.), self.out_dim)]
        else:
            raise NotImplementedError

        (h_val, c_val), _ = theano.scan(
            __step,
            sequences=projected_input,
            outputs_info=outputs_info,
            non_sequences=[U],
            go_backwards=self.prm('go_backwards'),
            name=name_tv(self.name, 'LSTM_layer'),
            n_steps=n_steps,
            strict=True,
            # Force the LSTM to work on the CPU.
            # mode=theano.compile.mode.get_default_mode().excluding('gpu').including('cpu'),
            # Don't gc in case this makes code faster.
            # allow_gc=False
        )

        if self.prm('go_backwards'):
            h_val = th_reverse(h_val)

        if input_tv.ndim == 3:
            self.output_tv = h_val.dimshuffle([1, 0, 2])
        elif input_tv.ndim == 2:
            self.output_tv = h_val
        else:
            raise NotImplementedError

        if self.prm('add_bias'):
            return (W, U, b)
        else:
            return (W, U)

    def needed_key(self):
        return self._needed_key_impl(
            'W', 'U', 'b', 'go_backwards', 'clip_gradient', 'l2_project',
            'add_bias', 'do_dropout', 'dropout_retention_freq')


class BiLSTM(Chip):

    ''' This requires W, U and b initializer
    '''

    def construct(self, input_tv):
        assert input_tv.ndim == 2, "Batch BiLSTM not implemented"
        # Before creating the sub LSTM's set the out_dim to half
        # Basically this setting would be used by the sub LSTMs
        self.params[self.name + "_forward_go_backwards"] = False
        if self.prm('segregate_bilstm_inputs'):
            assert self.in_dim % 2 == 0
            forward_input_tv = input_tv[:, :self.in_dim / 2]
            forward_input_tv_dim = self.in_dim / 2
            backward_input_tv = input_tv[:, self.in_dim / 2:]
            backward_input_tv_dim = self.in_dim / 2
        else:
            forward_input_tv = input_tv
            backward_input_tv = input_tv
            forward_input_tv_dim = self.in_dim
            backward_input_tv_dim = self.in_dim
        forward_class_chip = LSTM(self.name + "_forward", self.params).prepend(
            Start(out_dim=forward_input_tv_dim, output_tv=forward_input_tv))
        forward_interim = forward_class_chip.output_tv
        if self.prm('do_backward_pass'):
            self.params[self.name + "_backward_go_backwards"] = True
            backward_class_chip = LSTM(self.name + "_backward", self.params).prepend(
                Start(out_dim=backward_input_tv_dim, output_tv=backward_input_tv))
            backward = backward_class_chip.output_tv
            pass
        # Historically I was doing the extended version and then lopping off the
        # 0th and -1th index in a separate slice class_chip.
        # I have realized that the separate slice class_chip was not useful
        # Since the semantics are too tightly coupled to the implementation of
        # the LSTM so I have integrated them now.
        forward = forward_interim
        forward_dim_increment = 0
        if self.prm('forcefully_copy_embedding_to_output'):
            cur_chip = self
            while not isinstance(cur_chip, Embedding):
                cur_chip = cur_chip.previous_class_chip
                pass
            forward = T.concatenate(
                [forward_interim, cur_chip.output_tv], axis=1)
            forward_dim_increment = cur_chip.out_dim
            pass
        #------------------------------------------------------------#
        # NOTE: Meaning of all the options.                          #
        # stagger_schedule=extended: We copy input emb to output.    #
        # stagger_schedule=external: We dont copy input to output.   #
        # -----------------------------------------------------------#
        # do_backward_pass: We use the output of the backward LSTM   #
        #   Default:True.                                            #
        # -----------------------------------------------------------#
        # chop_bilstm: Should we chop the first and last vectors from#
        # the sequence. Default:False                                #
        #------------------------------------------------------------#
        # extended_multiplicative: Multiply the forward and back LSTM#
        #    and concatenate the input embedding.                    #
        # external_multiplicative: Multiply the forward and back LSTM#
        #    and but dont concatenate the input embedding.           #
        #------------------------------------------------------------#
        if (self.prm('stagger_schedule') == 'extended'):
            output_tv = (T.concatenate([forward, backward, input_tv], axis=1)
                         if self.prm('do_backward_pass')
                         else T.concatenate([forward, input_tv], axis=1))
            self.output_tv = (output_tv[1:-1]
                              if self.prm('chop_bilstm')
                              else output_tv)
            pass
        elif self.prm('stagger_schedule') == 'external':
            if self.prm('chop_bilstm'):
                if self.prm('do_backward_pass'):
                    self.output_tv = T.concatenate(
                        [forward[1:-1], backward[2:]], axis=1)
                    pass
                else:
                    self.output_tv = forward[1:-1]
                    pass
                pass
            else:
                if self.prm('do_backward_pass'):
                    self.output_tv = T.concatenate(
                        [forward, backward], axis=1)
                    pass
                else:
                    self.output_tv = forward
                    pass
                pass
            pass
        elif self.prm('stagger_schedule') == 'extended_multiplicative':
            if self.prm('chop_bilstm') or (not self.prm('do_backward_pass')):
                raise NotImplementedError()
            self.output_tv = T.concatenate(
                [forward * backward, input_tv], axis=1)
            pass
        elif self.prm('stagger_schedule') == 'external_multiplicative':
            if self.prm('chop_bilstm') or (not self.prm('do_backward_pass')):
                raise NotImplementedError()
            self.output_tv = forward * backward
            pass
        elif self.prm('stagger_schedule') == 'external_inplusive_multiplicative':
            if self.prm('chop_bilstm') or (not self.prm('do_backward_pass')):
                raise NotImplementedError()
            self.output_tv = forward * backward * input_tv
            pass
        else:
            raise NotImplementedError()

        out_dim = forward_class_chip.out_dim + forward_dim_increment
        if (self.prm('do_backward_pass')
                and not self.prm('stagger_schedule').endswith('multiplicative')):
            out_dim += backward_class_chip.out_dim
            pass
        if self.prm('stagger_schedule').startswith('extended'):
            out_dim += self.in_dim
            pass
        self.out_dim = out_dim
        return tuple()

    def needed_key(self):
        return (LSTM(self.name + '_forward').needed_key()
                + LSTM(self.name + '_backward').needed_key()
                + self._needed_key_impl(
                    'stagger_schedule', 'do_backward_pass',
                    'chop_bilstm', 'forcefully_copy_embedding_to_output', 'segregate_bilstm_inputs'))


#---------------------------------------------------------------------#
# Mixture Classes: Mixture convert LSTM outputs to CRF tensor scores. #
#---------------------------------------------------------------------#


class ConjunctiveMixture(Chip):

    ''' This requires a T initializer, where T stands for transition matrix.
    A conjunctive transition
    feature is a simple linear model on top of non-linear neural
    topologies. Another way of thinking about this is that we
    have featurized the transition matrix explicitly through conjunction
    features and that these features are then going to be added to some features
    that represent the emmission affinity.

    This model does not enforce similarities in the embeddings of hidden tags.
    Instead there is a single number that represents the affinity of a tag for
    another hidden tag. In the parlance of the paper this is the semi-linear model.
    '''

    def construct(self, input_tv):
        '''
        Params
        ------
        input_tv : It is a tensor of size (n_sentences, n_tokens, m).
          Here m = number of hidden variables. This tensor represents
          a batch of sentences. At each position we know the score of
          the word and a hidden tag.

        Returns
        -------
        We create a 4D tensor with the help of a transition matrix T. The transition matrix
        is a model of the affinity that hidden variables that follow each other feel for other.
        The output is a 4D tensor of size (n_sentences, n_tokens, m, m)
        '''
        prev_tag_dim = self.out_dim
        if self.prm('embed_BOS'):
            prev_tag_dim += 1
        T_ = self._declare_mat('T', prev_tag_dim, self.out_dim)
        T_.clip_gradient = self.prm('clip_gradient')
        T_.l2_project = self.prm('l2_project')
        T_.l2_projection_axis = 0
        self.output_tv = (T_.dimshuffle('x', 'x', 0, 1)
                          + input_tv.dimshuffle(0, 1, 'x', 2))
        return (T_,)

    def needed_key(self):
        return self._needed_key_impl(
            'T', 'embed_BOS', 'clip_gradient', 'l2_project')


class PedanticConcatenativeMixture(Chip):

    '''
    This is actually pedantic with the two typed factors construction.

    This mixture creates the CRF tensor by first computing the
    previous tag -> next tag scores by computing a rank one cross sum,
    then projecting it through a non-linearity then taking a projection and
    finally adding in the input which should have been passed through the same
    non-linearity itself.
    '''

    def construct(self, input_tv):
        '''
        Params
        ------
        input_tv : The input is a 3D tensor representing a batch of sentences with
          embedded tokens.

        Returns
        -------
        The input_tv is a matrix that has the tokens as its 0th dimension and
        usually LSTM features as the first dimension.

        NOTE: We don't need to project the input of this class_chip inside. We can
        just add a Linear class_chip before ConcatenativeMixture.
        '''
        Y_prev = self._declare_mat('A', self.out_dim + 1, self.in_dim)
        Y_next = self._declare_mat('B', self.out_dim,  self.in_dim)
        Y_prev.clip_gradient = self.prm('clip_gradient')
        Y_next.clip_gradient = self.prm('clip_gradient')
        prev_next_cross = (Y_prev.dimshuffle(0, 'x', 1)
                           + Y_next.dimshuffle('x', 0, 1))
        Y_nl = self.prm('activation_fn')(prev_next_cross)
        # NOTE: The last dimension corresponds to the hidden layer
        # nodes.
        v = self._declare_mat('v', self.in_dim)
        PairWise_Factor = T.tensordot(Y_nl, v, axes=(Y_nl.ndim - 1, 0))
        self.output_tv = (PairWise_Factor.dimshuffle('x', 'x', 0, 1)
                          + input_tv.dimshuffle(0, 1, 'x', 2))
        return (Y_prev, Y_next, v)

    def needed_key(self):
        return self._needed_key_impl('A', 'B', 'activation_fn', 'v')


class AbstractConcatenativeMixture(Chip):

    ''' The Concatenative Mixture classes implement Jason's idea of
    featurizing substructures and then applyig soft conjunctions on them.
    The StrictConcatenativeMixture is Jason's original idea.
    The LooseConcatenativeMixture is what we have gravitated to in the hopes
    that it would work better inspired by the ConjunctiveMixture models.
    '''

    def conc_node(self):
        raise NotImplementedError

    def construct(self, input_tv):
        '''
        The input_tv is a matrix that has the tokens as its 0th dimension and
        usually LSTM features as the first dimension.
        '''
        (Y, toreg) = self.conc_node()
        # Y is a 3D tensor that actually embeds the hidden-bigrams.
        # Each hidden bigram is a vector that is combined via a cross
        # product with the embedded tokens. This is the most expensive
        # model and most extensive model that says that the tokens
        # have a embedding that freely interacts with the tag-bigram embedding.
        # NOTE: We don't need to project input. Just add a Linear class_chip before
        # ConcatenativeMixture.
        # NOTE: We are saying that the last dimension corresponds to the hidden
        # layer nodes.
        self.output_tv = (input_tv.dimshuffle(0, 1, 'x', 'x', 2)
                          + Y.dimshuffle('x', 'x', 0, 1, 2))
        return toreg

    def _needed_key_impl(self, *things):
        return super(ConcatenativeMixture, self)._needed_key_impl(*things)


class StrictConcatenativeMixture(AbstractConcatenativeMixture):

    '''
    Basically imagine that we have two sets of features.
    Y_prev : features for the previous tag.
    Y_next : feature for the next tag.
    The dimensions of Y_prev is (#tags + 1(BOS), #interaction_dimensions)
    Note that the input has been pre-multiplied with a linear term before being passed in.
    '''

    def conc_node(self):
        # Add two for the start state and end state.
        Y_prev = self._declare_mat('A', self.out_dim + 1, self.in_dim)
        Y_next = self._declare_mat('B', self.out_dim, self.in_dim)
        Y_prev.clip_gradient = self.prm('clip_gradient')
        Y_next.clip_gradient = self.prm('clip_gradient')
        Y = Y_prev.dimshuffle(0, 'x', 1) + Y_next.dimshuffle('x', 0, 1)
        return (Y, (Y_prev, Y_next))

    def needed_key(self):
        return self._needed_key_impl('A', 'B')


class LooseConcatenativeMixture(AbstractConcatenativeMixture):

    '''
    The loose Concatenative model contains full conjunction features for the
    hidden variables but concatenates with the lstm features.
    '''

    def conc_node(self):
        Y = self._declare_mat('Y', self.out_dim + 1, self.out_dim, self.in_dim)
        Y.clip_gradient = self.prm('clip_gradient')
        return (Y, (Y,))

    def needed_key(self):
        return self._needed_key_impl('Y')

# #---------------------------------------------------------------#
# # First Order Nonlinear Score Functions for Dependency Parsing. #
# #---------------------------------------------------------------#


# class ConjunctiveMixtureForDependencyParsing(Chip):

#     def construct(self, input_tv):
#         X_parent = input_tv[:, :self.in_dim / 2]
#         X_child = input_tv[:, self.in_dim / 2:]
#         # output_tv[i, j, k] = Xp[i, k] + Xc[j, k]
#         self.output_tv = (X_parent.dimshuffle(0, 'x', 'x', 1)
#                           * X_child.dimshuffle('x',  0, 1, 'x')).reshape(
#             (input_tv.shape[0], input_tv.shape[0],
#              self.in_dim * self.in_dim / 4))
#         return tuple()


# class UnlabeledStrictConcatenativeMixtureForDependencyParsing(Chip):

#     ''' See: `LabeledStrictConcatenativeMixtureForDependencyParsing`
#     '''

#     def construct(self, input_tv):
#         X_parent = input_tv[:, :self.in_dim / 2]
#         X_child = input_tv[:, self.in_dim / 2:]
#         if (self.kn('low_rank_tensor_like') in self.params
#                 and self.prm('low_rank_tensor_like')):
#             self.output_tv = X_parent.dimshuffle ( 0 , 'x', 1) \
#                 * X_child.dimshuffle('x',  0, 1)
#         else:
#             self.output_tv = X_parent.dimshuffle ( 0 , 'x', 1) \
#                 + X_child.dimshuffle('x',  0, 1)
#         return tuple()


# class LabeledStrictConcatenativeMixtureForDependencyParsing(Chip):

#     '''
#     We assume that the input_tv has been super-sized to a large dimension and
#     the left portion corresponds to the parent words, the right portion corresponds to the child words.
#     Then we create features for tags.

#     Finally we add them all together through dimshuffle.
#     Note : The out_dim is the number of tags.
#     '''

#     def construct(self, input_tv):
#         T_ = self._declare_mat('T', self.out_dim, self.in_dim / 2)
#         X_parent = input_tv[:, :self.in_dim / 2]
#         X_child = input_tv[:, self.in_dim / 2:]
#         self.output_tv = X_parent.dimshuffle ( 0 , 'x', 'x', 1) \
#             + X_child.dimshuffle('x',  0 , 'x', 1) \
#             + T_.dimshuffle('x', 'x',  0, 1)
#         return (T_,)

#     def needed_key(self):
#         return self._needed_key_impl('T')

# #-------------------------------#
# # Edge Length Penalty Features. #
# #-------------------------------#


# class UnaryEncodedEdgeLengthPenalty(Chip):

#     def construct(self, input_tv):
#         ''' The input is assumed to be a 2D matrix of arc scores.
#         The arc length is unary encoded by the following quantization scheme
#         |_| 1, if i <= j-99
#         ...
#         |_| 1, if i <= j-2
#         |_| 1, if i <= j-1
#         |_| 1, if i <= j-0
#         |_| 1, if i >= j+0
#         |_| 1, if i >= j+1
#         |_| 1, if i >= j+2
#         ...
#         |_| 1, if i >= j+99
#         Then we are supposed to take dot product of this binary vector with a W matrix.
#         In practice of course we need to do this more pragmatically.
#         We want to avoid unnecessary multiplications.
#         Run a single scan over k between [0 ... (i*(#row) + j)]
#         Then decompose k into i, j (using #row as a non sequence)
#         Then sum columns T[:T.maximum(i-j+1, 0)] and t[:T.maximum(j-i+1, 0)]
#         When i==j:
#             T[0] + t[0]
#         When i==2, j==1:
#             Columns T[:2] + t[:0]
#         When i==3, j==1:
#             Columns T[:3] + t[:0]
#         When i==1, j==2:
#             Columns T[:0] + t[:2]
#         Now note that by running a scan we would get a single matrix.
#         The first dimension would be along k.
#         The second dimension would be the the inner dimension of the factor.
#         The dimension that I need to take dot product with later on eq2.3
#         So I would need to reshape the results of scan before assigning it to output_tv
#         '''
#         n_row = input_tv.shape[0]

#         def __step(k, T_, t, n_row):
#             ''' Note that by interpreting i and j as such
#             we are coupled to the reshape function that would be
#             used outside. The reshape function had better receive
#             first_dimension along k such that the columns moved fastest.
#             '''
#             j = T.mod(k, n_row).astype('int32')
#             i = ((k - j) / n_row).astype('int32')
#             return (T_[:T.maximum(i - j + 1, 0).astype('int32')].sum(axis=0, acc_dtype='float32')
#                     + t[:T.maximum(j - i + 1, 0).astype('int32')].sum(axis=0, acc_dtype='float32'))

#         Plen = self.prm('penalty_vector_max_length')
#         T_ = self._declare_mat('T', Plen, self.out_dim)
#         t = self._declare_mat('t', Plen, self.out_dim)
#         tmp_output, _ = theano.scan(__step,
#                                     sequences=T.arange(
#                                         n_row * n_row, dtype='int32'),
#                                     non_sequences=[T_, t, n_row],
#                                     name=name_tv(
#                                         self.name, 'UnaryEdgePenalty'),
#                                     strict=True)
#         if (self.kn('low_rank_tensor_like') in self.params
#                 and self.prm('low_rank_tensor_like')):
#             self.output_tv = input_tv * \
#                 tmp_output.reshape([n_row, n_row, self.out_dim])
#         else:
#             self.output_tv = input_tv + \
#                 tmp_output.reshape([n_row, n_row, self.out_dim])
#         return (T_, t)

#     def needed_key(self):
#         return self._needed_key_impl('T', 't')


# class EdgeLengthPenalty(Chip):

#     def construct(self, input_tv):
#         ''' The input is assumed to be a 2D matrix of arc scores.
#         '''
#         assert input_tv.ndim == 2
#         Plen = self.prm('penalty_vector_max_length')
#         P = self._declare_mat('P', Plen, Plen)
#         n = input_tv.shape[0]
#         self.output_tv = input_tv + P[:n, :n]
#         return (P,)

#     def needed_key(self):
#         return self._needed_key_impl('P')


# class SaturatingEdgeLengthPenalty(Chip):

#     def construct(self, input_tv):
#         ''' The input is assumed to be a 2D matrix of arc scores.
#         '''
#         assert input_tv.ndim == 2
#         Plen = self.prm('penalty_vector_max_length')
#         P = self._declare_mat('P', Plen, Plen)
#         n = input_tv.shape[0]
#         self.output_tv = input_tv + T.minimum(T.maximum(P[:n, :n], -2), 2)
#         return (P,)

#     def needed_key(self):
#         return self._needed_key_impl('P')

#---------------------------------------------#
# Base class for the training objective class_chip. #
#---------------------------------------------#


class ScorableChip(Chip):

    ''' A ScorableChip is defined by its ability of producing a score of an
    (input, output) tuple. This type of Chip would usually be the last in a
    stack, though it's not necessary, for example Conjunctive features can use
    output classes to produce scores even if they are second last.

    Usually though the supervision does not penetrate down.
    ------------------------------------------------------------------------
    The Order Zero and Order One Max are class_chips for inference and training.
    These class_chips define two attributes, score, gold_y, and output_tv
    ------------------------------------------------------------------------
    '''

    def prepend(self, previous_class_chip):
        retval = super(ScorableChip, self).prepend(previous_class_chip)
        assert hasattr(self, 'score')
        assert hasattr(self, 'gold_y')
        return retval


class OrderZeroGreedyMax(ScorableChip):

    ''' Select the category that has the highest score in the input matrix.
    The first dimension is the index in the sequence and the second dimension
    is the index of the output label.

    Note that we assume that the inputs are scores in log space.
    '''

    def construct(self, input_tv):
        '''
        Params
        ------
        input_tv : The input_tv is a tensor of the type
                   (n_sentences, n_tokens, n_classes)
        '''
        class_axis = input_tv.ndim - 1
        # This is the output of the Order Zero Greedy Chip.
        self.output_tv = T.argmax(input_tv, axis=class_axis)

        # This is the golden_output. This would usually be used to define
        # the theano function for training. gold_y[i, j] is the golden
        # label of the j-th token of the i-th sentence.
        if input_tv.ndim - 1 == 1:
            gold_y_fnc = T.ivector
        elif input_tv.ndim - 1 == 2:
            gold_y_fnc = T.imatrix
        else:
            raise NotImplementedError

        gold_y = gold_y_fnc(name_tv(self.name, 'gold_y')).astype('int32')
        self.gold_y = gold_y

        # The numerator score is the score of the correct class at
        # each location.
        # n_sentences = input_tv.shape[0]
        numerator_score = th_choose(input_tv, gold_y)
        # The following denominator scores are made under the assumption of
        # that the loss of misprediction is uniformly 1.
        if self.prm('objective') == objective.ConditionalLikelihood:
            denominator_score = th_logsumexp(
                input_tv, axis=class_axis, keepdims=False)
        elif self.prm('objective') == objective.SoftMaxMargin:
            decrement_correct_class_score_by_one = theano.tensor.inc_subtensor(
                th_choose(input_tv, gold_y, reshape=False), -1).reshape(input_tv.shape)
            denominator_score = th_logsumexp(
                decrement_correct_class_score_by_one,
                axis=class_axis,
                keepdims=False)
        elif self.prm('objective') == objective.HingeLoss:
            decrement_correct_class_score_by_one = theano.tensor.inc_subtensor(
                th_choose(input_tv, gold_y, reshape=False), -1).reshape(input_tv.shape)
            denominator_score = decrement_correct_class_score_by_one.max(
                axis=class_axis)
        elif self.prm('objective') == objective.MixedLoss:
            raise NotImplementedError
        numerator_score.name = 'numerator_score'
        denominator_score.name = 'denominator_score'
        hypothesis_score = numerator_score - denominator_score
        self.score = T.mean(hypothesis_score)
        return tuple()

    def needed_key(self):
        return self._needed_key_impl('objective')


class OrderOnePathMax(ScorableChip):

    ''' The goal of this class_chip is to perform inference in an order one CRF.
    Also we need to produce a differentiable score of a supervised
    (sequence, tag) pair.
    The tag is assumed to be known `gold_y`, we only need to produce `output_tv`
    and `score`

    decode_type can be `viterbi` or `mbr`. `mbr` really only makes sense if you
    have done CLL training. Actually what you really want to do is MBR training
    using the expectation semiring but that is a specialized op.
    '''

    def construct(self, input_tv):
        '''
        Params
        ------
        input_tv : input_tv is a tensor of scores. Let's abbreviate it by Q.
        Q is a 4D tensor of shape(n_sentences, n_tokens, n_tags+1, n_tags)

        Let T = Q[k] for some k:
        T(i,j,k)  = Transition(state k | state j) + Emission(output i | state k)
        'transition to state k given that we were in state j'.
        j represents the previous tag.
        k represents the current tag.
        Let: [I, J, K] be dimensions of the tensor.
           : N be length of sentence,
           : S be the number of states
        Then: I = N; J = S + 1; K = S

        T[0, -1, k] contains the score of starting from <BOS> and
        then producing hidden tag `k` and output token t0.
        Since this depends on the actual word we don't store it as a parameter
        of this layer.
        As opposed to p(EOS | tag) which is a parameter independent of the word.
        '''
        assert input_tv.ndim == 4

        # The golden tag sequence matrix.
        # gold_y.shape = (n_sentences, n_tokens)
        gold_y = T.imatrix(name_tv(self.name, 'gold_y')).astype('int32')
        self.gold_y = gold_y

        # A vector of scores that represents score(EOS, prev_tag)
        # These parameters are learnt during training.
        l = self._declare_mat('l', self.out_dim)

        #--------------------------------------------------------#
        # The output_tv is used only during test-time inference. #
        # It is a batch of highest scoring hidden sequence in an #
        # order one CRF                                          #
        #--------------------------------------------------------#
        viterbi_p = self.prm('viterbi_p')
        assert (self.prm('objective') == objective.ConditionalLikelihood
                or viterbi_p)

        self.output_tv, _ = theano.scan(
            find_highest_scoring_sequence_in_order_one_crf,
            sequences=input_tv,
            non_sequences=[l, viterbi_p],
            outputs_info=[None],
            name='OrderOnePathMax_batch_inference',
            strict=True)
        #---------------------------------------------#
        # The following members are used in training. #
        #---------------------------------------------#
        # The batch of unnormalized scores of tagged sequences
        gold_y_unnormalized_score, _ = theano.scan(
            tagged_sequence_unnormalized_score_in_order_one_crf,
            sequences=[input_tv, gold_y],
            non_sequences=[l],
            outputs_info=None,
            name='OrderOnePathMax_batch_unnorm_score',
            strict=True)
        gold_y_unnormalized_score.name = 'gold_y_unnormalized_score'
        # Depending on the training objective, the partition function
        #  computed changes.
        if self.prm('objective') == objective.ConditionalLikelihood:
            partition_val, _ = theano.scan(
                sum_all_paths_of_order_one_crf,
                sequences=[input_tv],
                non_sequences=[l],
                outputs_info=[None],
                name='OrderOnePathMax_batch_CLL_partition',
                strict=True)
        elif self.prm('objective') == objective.SoftMaxMargin:
            # Essentially the computation we have to do is that
            # For each input_tv[i, :, t] if gold_y[i] == t then substract 1.
            # Remember here that input_tv is a 4D tensor.
            # and gold_y[i, j] is the index of the correct tag
            # (axis=3) essentially I need to set
            # input_tv[i, j, :, gold_y[i, j]] -= 1
            input_tv = T.inc_subtensor(
                th_batch_choose(input_tv, gold_y, reshape=False),
                -1).reshape(input_tv.shape)
            partition_val, _ = theano.scan(
                sum_all_paths_of_order_one_crf,
                sequences=[input_tv],
                non_sequences=[l],
                outputs_info=[None],
                name='OrderOnePathMax_batch_SoftMaxMargin_partition',
                strict=True)
        elif self.prm('objective') == objective.HingeLoss:
            input_tv = T.inc_subtensor(
                th_batch_choose(input_tv, gold_y, reshape=False),
                -1).reshape(input_tv.shape)
            partition_val, _ = theano.scan(
                find_score_of_viterbi_sequence_in_order_one_crf,
                sequences=[input_tv],
                non_sequences=[l],
                outputs_info=[None],
                name='OrderOnePathMax_batch_HingeLoss_partition',
                strict=True)
        elif self.prm('objective') == objective.MixedLoss:
            raise NotImplementedError
        partition_val.name = 'order1pathmax_partition_val'
        self.debug_tv_list.append(gold_y_unnormalized_score)
        self.debug_tv_list.append(partition_val)
        # self.score is the normalized score of the gold sequences in batch.
        self.score = T.mean(gold_y_unnormalized_score - partition_val)
        return (l,)

    def needed_key(self):
        return self._needed_key_impl('viterbi_p', 'l', 'objective')


class RegularizerMixin(object):
    # def prepend(self, previous_class_chip):
    #     self.previous_class_chip = previous_class_chip
    #     return super(RegularizerMixin, self).prepend(previous_class_chip)

    def __getattr__(self, item):
        ''' Inherit all the attributes of the previous class_chip.
        At present I can only see this functionality being useful
        for the case of the Regularization class_chip. Maybe we would move
        this down in case it is found necessary later on, but there is
        chance of abuse.
        '''
        try:
            return getattr(self.previous_class_chip, item)
        except KeyError:
            raise AttributeError(item)

    def needed_key(self):
        return self._needed_key_impl('reg_weight')


class L2Reg(RegularizerMixin, Chip):

    ''' This supposes that the previous class_chip would have a score attribute.
    And it literally only changes the score attribute by adding the
    regularization term on top of it.
    '''

    def construct(self, input_tv):
        rv_list = self.params.regularizable_parameters()
        for rv_idx, rv in enumerate(rv_list):
            if rv_idx == 0:
                L2 = T.sum(rv * rv)
            else:
                L2 += T.sum(rv * rv)

        L2.name = name_tv(self.name, 'L2')
        self.score = self.score - self.prm('reg_weight') * L2
        return tuple()


class L1Reg(RegularizerMixin, Chip):

    ''' This supposes that the previous class_chip would have a score attribute.
    And it literally only changes the score attribute by adding the
    regularization term on top of it.
    '''

    def construct(self, input_tv):
        rv_list = self.params.regularizable_parameters()
        for rv_idx, rv in enumerate(rv_list):
            if rv_idx == 0:
                L1 = T.sum(T.abs_(rv))
            else:
                L1 += T.sum(T.abs_(rv))

        L1.name = name_tv(self.name, 'L1')
        self.score = self.score - self.prm('reg_weight') * L1
        return tuple()


def calculate_params_needed(class_chips):
    l = ['in_dim', 'batch_input']
    for c in class_chips:
        l += c[0](c[1]).needed_key()
    return l


def metaStackMaker(stack, stack_config):
    ''' Create a circuit from a stack of `chips` and the associated
    `stack_config` object.
    Params
    ------
    stack        : The stack contains a list of chips. These chips essentially
      contain a prepend method
    stack_config : The stack_config contains the parameters that influence the
      building of the circuits and training and testing. This function does not
      really need stack_config right now but it is passed because we might want
      to change some aspects of the circuit building based on the `stack_config`
    '''
    #---------------------------------------------------------#
    # Show all parameters that would be needed in this system #
    #---------------------------------------------------------#
    unavailable_params = [k
                          for k in calculate_params_needed(stack)
                          if k not in stack_config]
    assert len(unavailable_params) == 0, str(unavailable_params)
    #---------------------------------------------------------------#
    # Build the Circuit Now from the Stack of Chips that was input. #
    #---------------------------------------------------------------#
    print '\n', 'Building Stack now'
    print 'Start: in_dim:', stack_config['in_dim'], \
        'batch_input:', stack_config['batch_input']
    start_tv_f = (T.itensor3
                  if stack_config['batch_input']
                  else T.imatrix)
    start_tv = ([start_tv_f('batch_input_%d' % i)
                 for i
                 in range(stack_config['multi_embed_count'])]
                if stack_config['multi_embed']
                else start_tv_f('absolute_input_tv'))
    current_class_chip = Start(stack_config['in_dim'], start_tv)
    for e in stack:
        current_class_chip = e[0](
            e[1], stack_config).prepend(current_class_chip)
        print e[1], "In_dim:", current_class_chip.in_dim, \
            "Out_dim:", current_class_chip.out_dim, \
            "Output ndim:", current_class_chip.output_tv.ndim
        for e in current_class_chip.needed_key():
            print (e, stack_config[e])
    #----------------------------#
    # Prepare the output values. #
    #----------------------------#
    #-- `absolute_input_tv`, `pred_y`, `gold_y` are inputs and outputs.
    absolute_input_tv = current_class_chip.absolute_input_tv

    pred_y = current_class_chip.output_tv
    assert (hasattr(current_class_chip, 'score')
            == hasattr(current_class_chip, 'gold_y'))
    #-- `cost_or_known_grads_tv`, `grads` are used during training.
    tmp_dp_list = stack_config.differentiable_parameters()
    if hasattr(current_class_chip, 'score'):
        gold_y = current_class_chip.gold_y
        cost_or_known_grads_tv = (-current_class_chip.score)
        grads = T.grad(cost_or_known_grads_tv, wrt=tmp_dp_list)
    else:
        gold_y = None
        known_grads_tv = T.TensorType(
            'float32', broadcastable=[False] * pred_y.ndim)('known_grads_tv')
        cost_or_known_grads_tv = known_grads_tv
        grads = T.grad(None,
                       wrt=tmp_dp_list,
                       known_grads={pred_y: known_grads_tv})
    for tmp_idx in range(len(tmp_dp_list)):
        grads[tmp_idx].wrt_name = tmp_dp_list[tmp_idx].name
    stack_ns = Namespace()
    stack_ns.absolute_input_tv = absolute_input_tv
    stack_ns.gold_y = gold_y
    stack_ns.pred_y = pred_y
    stack_ns.cost_or_known_grads_tv = cost_or_known_grads_tv
    stack_ns.grads = grads
    stack_ns.f_debug = NotImplemented
    stack_ns.debug_tv_list = current_class_chip.debug_tv_list
    return stack_ns


def conjunctiveOrderZeroLSTMCRF(stack_config):  # simpleCrf0LSTM1
    stack = [
        (Embedding, 'wemb1'),
        (LSTM, 'lstm'),
        (BiasedLinear, 'lstm_output_transform'),
        (OrderZeroGreedyMax, 'order0classifier'),
        (L2Reg, 'l2reg')
    ]
    return metaStackMaker(stack, stack_config)


def conjunctiveOrderZeroBiLSTMCRF(stack_config):  # simplecrf0
    stack = [
        (Embedding, 'wemb1'),
        (BiLSTM, 'bilstm'),
        (BiasedLinear, 'lstm_output_transform'),
        (OrderZeroGreedyMax, 'order0classifier'),
        (L2Reg, 'l2reg')
    ]
    return metaStackMaker(stack, stack_config)


def conjunctiveOrderOneBiLSTMCRF(stack_config):  # simplecrf1
    stack = [
        (Embedding,         'wemb1'),
        (BiLSTM,            'bilstm'),
        (BiasedLinear,      'lstm_output_transform'),
        (ConjunctiveMixture, 'conjunction'),
        (OrderOnePathMax,   'order1pathmax'),
        (L2Reg,             'l2reg'),
    ]
    return metaStackMaker(stack, stack_config)


def conjunctiveOrderOneDeepBiLSTMCRF(stack_config):  # deepcrf
    stack = [
        (Embedding,         'wemb1'),
    ] + (
        [(BiLSTM,            'bilstm')] * stack_config['nlstm']
    ) + [
        (BiasedLinear,      'lstm_output_transform'),
        (ConjunctiveMixture, 'conjunction'),
        (OrderOnePathMax,   'order1pathmax'),
        (L2Reg,             'l2reg'),
    ]
    return metaStackMaker(stack, stack_config)


def strictConcatenativeOrderOneBiLSTMCRF(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [
        (Embedding, 'wemb1'),
    ] + [(BiLSTM, 'bilstm%d' % i) for i in range(bilstm_depth)
         ] + [  # After slicing we are left with vector based outputs
        # from the lstm, that we then pass through a biased linear unit
        # Note that its output_dim is used by the next layer in a
        # non-standard way
        (BiasedLinear, 'lstm_output_transform'),
        # This node has parameters for the substructures
        # (Count of output_values x output_dim of previous layer)
        # that are conjoined with embeddings of the
        # hidden substructures. (Jason's model reduces to this implementation
        # when done efficiently.)
        # and the output of this layer is the
        (StrictConcatenativeMixture, 'concatenation'),
        (Activation, 'scm_for_seqlabel_activation'),
        (MaxPool, 'scm_for_seqlabel_maxpool'),
        (Collapse, 'scm_for_seqlabel_collapse'),
        #
        (OrderOnePathMax, 'order1pathmax'),
        (L2Reg, 'l2reg'),
    ]
    return metaStackMaker(stack, stack_config)


def pedanticConcatenativeOrderOneBiLSTMCRF(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [
        (Embedding, 'wemb1'),
    ] + [(BiLSTM, 'bilstm%d' % i) for i in range(bilstm_depth)
         ] + [(BiasedLinear, 'lstm_output_transform'),
              (PedanticConcatenativeMixture, 'concatenation'),
              (OrderOnePathMax, 'order1pathmax'),
              (L2Reg, 'l2reg'),
              ]
    return metaStackMaker(stack, stack_config)


def strictConcatenativeOrderOneDepParserOnBiLSTM(stack_config):
    stack = [
        (Embedding, 'wemb1'),
        (BiLSTM, 'bilstm'),
        (BiasedLinear, 'lstm_output_transform'),
        (LabeledStrictConcatenativeMixtureForDependencyParsing, 'scm_for_dp'),
        (Activation, 'scm_for_dp_activation'),
        (Collapse, 'scm_for_dp_collapse'),
    ]
    return metaStackMaker(stack, stack_config)


def unlabeledConcatenativeOrderOnePDPDeepBiLSTM(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [(Embedding, 'wemb1'),
             ] + [(BiLSTM, 'bilstm%d' % i)
                  for i in range(bilstm_depth)
                  ] + [(Linear, 'lstm_output_transform'),
                       (UnlabeledStrictConcatenativeMixtureForDependencyParsing,
                        'scm_for_dp'),
                       (Activation, 'scm_for_dp_activation'),
                       (Collapse, 'scm_for_dp_collapse'),
                       ]
    return metaStackMaker(stack, stack_config)


def unlabeledConcatenativeOrderOnePDPDeepBiLSTMWithEdgeLengthPenalty(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [(Embedding, 'wemb1'),
             ] + [(BiLSTM, 'bilstm%d' % i)
                  for i in range(bilstm_depth)
                  ] + [(Linear, 'lstm_output_transform'),
                       (UnlabeledStrictConcatenativeMixtureForDependencyParsing,
                        'scm_for_dp'),
                       (Activation, 'scm_for_dp_activation'),
                       (Collapse, 'scm_for_dp_collapse'),
                       (EdgeLengthPenalty, 'edge_length_penalty'),
                       ]
    return metaStackMaker(stack, stack_config)


def unlabeledConcatenativeOrderOnePDPDeepBiLSTMWithSaturatingEdgeLengthPenalty(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [(Embedding, 'wemb1'),
             ] + [(BiLSTM, 'bilstm%d' % i)
                  for i in range(bilstm_depth)
                  ] + [(Linear, 'lstm_output_transform'),
                       (UnlabeledStrictConcatenativeMixtureForDependencyParsing,
                        'scm_for_dp'),
                       (Activation, 'scm_for_dp_activation'),
                       (Collapse, 'scm_for_dp_collapse'),
                       (SaturatingEdgeLengthPenalty, 'edge_length_penalty'),
                       ]
    return metaStackMaker(stack, stack_config)


def unlabeledConcatenativeOrderOnePDPDeepBiLSTMWithUnaryEdgeLengthPenalty(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [(Embedding, 'wemb1'),
             ] + [(BiLSTM, 'bilstm%d' % i)
                  for i in range(bilstm_depth)
                  ] + [(Linear, 'lstm_output_transform'),
                       (UnlabeledStrictConcatenativeMixtureForDependencyParsing,
                        'scm_for_dp'),
                       (UnaryEncodedEdgeLengthPenalty,
                        'unary_edge_penalty'),
                       (Activation, 'scm_for_dp_activation'),
                       (Collapse, 'scm_for_dp_collapse'),
                       ]
    return metaStackMaker(stack, stack_config)


def unlabeledLinearOrderOnePDPDeepBiLSTMWithUnaryEdgeLengthPenalty(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [(Embedding, 'wemb1'),
             ] + [(BiLSTM, 'bilstm%d' % i)
                  for i in range(bilstm_depth)
                  ] + [(Linear, 'lstm_output_transform'),
                       (UnlabeledStrictConcatenativeMixtureForDependencyParsing,
                        'scm_for_dp'),
                       (UnaryEncodedEdgeLengthPenalty,
                        'unary_edge_penalty'),
                       (Average, 'scm_for_dp_collapse'),
                       ]
    return metaStackMaker(stack, stack_config)


def unlabeledPedanticLinearOrderOnePDPDeepBiLSTMWithUnaryEdgeLengthPenalty(stack_config):
    bilstm_depth = stack_config['bilstm_depth']
    stack = [(Embedding, 'wemb1'),
             ] + [(BiLSTM, 'bilstm%d' % i)
                  for i in range(bilstm_depth)
                  ] + [(Linear, 'lstm_output_transform'),
                       (UnlabeledStrictConcatenativeMixtureForDependencyParsing,
                        'scm_for_dp'),
                       (UnaryEncodedEdgeLengthPenalty,
                        'unary_edge_penalty'),
                       (Collapse, 'scm_for_dp_collapse'),
                       ]
    return metaStackMaker(stack, stack_config)


def correctUnlabeledPedanticLinearOrderOnePDPDeepBiLSTMWithUnaryEdgeLengthPenalty(stack_config):
    stack_config['unary_edge_penalty_low_rank_tensor_like'] = 1
    stack_config['scm_for_dp_low_rank_tensor_like'] = 1
    return unlabeledPedanticLinearOrderOnePDPDeepBiLSTMWithUnaryEdgeLengthPenalty(stack_config)
