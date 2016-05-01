'''
| Filename    : lstm_seqlabel_circuit_order_one_crf_decode_and_partition.py
| Description : Viterbi decoding and Partition calculation for Order one CRFs.
| Author      : Pushpendre Rastogi
| Created     : Sat Oct 24 12:53:47 2015 (-0400)
| Last-Updated: Sun Nov 15 23:58:09 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 26
The main function of this module is find_decode_and_partition_of_order_one_crf.
This function is supposed to
'''
from util_theano import th_logsumexp, th_reverse
import theano
import theano.tensor
import numpy

#-------------------------------------#
# These functions are self contained. #
#-------------------------------------#


def tagged_sequence_unnormalized_score_in_order_one_crf(input_tv, y, l):
    '''
    Simply sum the log-scores along the path suggested by `y` in the `input_tv`
    tensor.

    Params
    ------
    input_tv : A 3D tensor of (token, prev_pos, cur_pos) log scores.
      the input_tv also contains scores of
    y    : The true sequence that was actually followed.
    l    : The score of (EOS | tag)
    '''
    def _score_step(o, y, p_, y_):
        return ((p_ + o[y_, y]), y)
    [rval, _], _ = theano.scan(_score_step,
                               sequences=[input_tv[1:, :-1], y[1:]],
                               #sequences=[input_tv, y],
                               outputs_info=[input_tv[0, -1, y[0]], y[0]],
                               #outputs_info=[0.0, numpy.int32(-1)],
                               name='OrderOnePathMax_scan_score_step',
                               strict=True)
    return rval[-1] + l[y[-1]]


def retrieve_path_from_backpointers(bp, starting_point):
    '''
    Theano scan loop to follow backpointers, starting from a given spot.
    Params
    ------
    bp             : The trail of backpointers. Think of this is as a list of
        lists where we start from the back `bp = list[N][starting_point]` and
        then go to list[N-1][bp] and so on.
    starting_point :
    '''
    vp_prefix = th_reverse(
        theano.scan(
            lambda p, y: p[y],
            sequences=bp,
            outputs_info=starting_point,
            go_backwards=True,
            name='OrderOnePathMax_scan__bkpntr',
            strict=True)[0])
    return theano.tensor.concatenate([vp_prefix, starting_point.dimshuffle('x')])


def forward_pass_order_one_crf(input_tv, l, viterbi_p, add_contribution_of_l=True):
    '''
    Params
    ------
    input_tv : A 3D tensor of scores. Let's abbv it by T.
        T(i,j,k) = Transition(state k | state j) + Emission(output i | state k)
        'transition to state k given that we WERE in state j'
        + Score of emitting token i given state k.
    viterbi_p : This boolean influences whether we use the (Plus, Max) semiring,
        or the (Plus, Log-Sum-Exp) semiring.
    l : It is the score for reaching `EOS` from any particular tag.
    Returns
    -------
    ret_val : ret_val is a list of [forward_scores, bp]
    '''

    def _forward_step(o_, p_):
        '''
        Params
        ------
        o_ : This contains the edge score of going from tag j to tag k
             o_[j, k] = T[i+1, j, k]
        p_ : p_[j] is the $\alpha$ score of reaching the current tag, `tag j`
        '''
        # Each row of o_ corresponds to a particular `previous-tag`.
        # we want to add the same value of p_[j] to each row of o_, therefore
        # we duplicate/broadcast it along columns.
        f_ = p_.dimshuffle(0, 'x') + o_

        # The backpointers corresponding to f_.
        # By computing argmax along axis=0, we are finding the best previous-tag
        # for a particular current-tag.
        # so bp[k] where k is the current tag, gives us the best previous tag.
        bp = f_.argmax(axis=0).astype('int32')

        if viterbi_p:
            # If we want to pick the highest scoring sequence, then we should set
            # viterbi_p to True and find the highest scoring sequence.
            # This is Viterbi decoding.
            p = f_.max(axis=0)
        else:
            # This route must be taken if we want to compute the total sum of
            # all sequences. The forward equation then becomes:
            # new_p[k] = log(sum(exp(o_[j, k] + p_[j]) over j))
            p = th_logsumexp(f_, axis=0)
        return p, bp

    # We are keeping all tokens except for 0th token, and all `prev-tag` except
    # for the `prev-tag` = -1 which indicates BOS.
    input_sequence = input_tv[1:, :-1]

    # outputs_info has two members
    # first one `input_tv[0, -1]` is the score of emitting token 0 along with
    # previous tag being BOS.
    # the second one, `bp` is the backpointer.
    outputs_info = [input_tv[0, -1], None]

    # The ret_val is a collection of forward_scores/viterbi scores and
    # back-pointers.
    [forward_scores, bp], _ = theano.scan(
        _forward_step,
        sequences=input_sequence,
        outputs_info=outputs_info,
        name='OrderOnePathMax_scan__step',
        strict=True)
    # The forward_scores did not contain `l` - the cost of
    # going from a tag to EOS, so now we add them.
    if add_contribution_of_l:
        forward_scores = theano.tensor.inc_subtensor(
            forward_scores[-1], l)
    return [forward_scores, bp]


def do_mbr_decoding(forward_scores, input_tv, l):
    ''' When doing MBR decoding we are doing a backward pass over the lattice.
    '''
    def _backward_step(o_, b_):
        '''
        Backward Equation: b[i, l] = log_sum_exp(T[i+1, l, k] + b[i+1, k] over k)
        o_[l, k] = T[i+1, l, k]
        b_[k]    = b[i+1,    k]
        '''
        # In contrast with the forward step, here, we want to think of b_ as
        # a single row. We broadcast b_ along the rows and add to all rows of o_.
        # Then we need to log_sum_exp all the columns in a row.
        f_ = (b_.dimshuffle('x', 0) + o_)
        return th_logsumexp(f_, axis=1)
    # Compute the backward_scores, since we go_backwards, we call a th_reverse
    # at the top.
    backward_scores = th_reverse(theano.scan(_backward_step,
                                             # Cost of emitting token given previous and
                                             # current tag
                                             # Its shape is (#tag+1, #tag)
                                             sequences=input_tv[1:, :-1],
                                             outputs_info=[l],
                                             name='OrderOnePathMax_backscan_step',
                                             strict=True,
                                             go_backwards=True)[0])
    # Concatenate these scores with `l`
    backward_scores = theano.tensor.concatenate(
        [backward_scores, l.dimshuffle('x', 0)])
    mbr_scores = backward_scores + forward_scores
    return theano.tensor.argmax(mbr_scores, axis=1)

#------------------------------#
# Self contained functions END #
#------------------------------#


def sum_all_paths_of_order_one_crf(input_tv, l):
    ''' We want to sum all the paths in an order one crf under the
    (sum, log-sum-exp) semiring. i.e. we assume that the scores present in the
    input_tv are log-scores that need to be added to get the score of a sequence
    and they need to be log-sum-exp'ed to get the total score of two sequences.

    Params
    ------
    input_tv : A 3D tensor of (token, prev_pos, cur_pos) log scores.
    '''
    forward_scores = forward_pass_order_one_crf(input_tv, l, False)[0]
    return th_logsumexp(forward_scores[-1], axis=0)


def find_score_of_viterbi_sequence_in_order_one_crf(
        input_tv, l):
    '''
    Params
    ------
    input_tv :
    l :
    Returns
    -------
    Note here we are only interested in finding the score of
    the highest scoring sequence. We don't need to backtrack
    with backpointers.
    '''
    [forward_scores, _bp] = forward_pass_order_one_crf(
        input_tv, l, True)
    return forward_scores[-1].max(axis=0)


def find_highest_scoring_sequence_in_order_one_crf(
        input_tv, l, viterbi_p=True):
    ''' This is an implementation of the viterbi algorithm for finding highest
    scoring paths in a regular trellis. We have not implemented pruning here.

    Params
    ------
    input_tv    : A 3D tensor of (token, prev_pos, cur_pos) log scores.
    viterbi_p   : Should we use viterbi decoding, or posterior-decoding?
                  posterior-decoding is MBR under a probabilistic model.
    l           : A vector of scores that represent score(EOS, prev_tag)

    NOTE: The viterbi_p argument only applies to the partition
    value. The backpointers and the actual decode is always viterbi
    in this code.
    '''
    # Compute the forward pass and backpointers.
    # Tell the function whether you want to compute just the viterbi_decode
    # or you want a full forward pass to later on do MBR decoding?
    if viterbi_p:
        [forward_scores, bp] = forward_pass_order_one_crf(
            input_tv, l, viterbi_p)
        starting_point = forward_scores[-1].argmax(axis=0).astype('int32')
        decode_path = retrieve_path_from_backpointers(bp, starting_point)
    else:
        # This only makes sense in a CLL setting really.
        # We don't care about the back-pointers since we are doing posterior
        # decoding.
        [forward_scores, _] = forward_pass_order_one_crf(
            input_tv, l, viterbi_p, add_contribution_of_l=False)
        # We are concatenating the value of BOS -> Tag scores to the forward
        # scores along axis 0.
        augmented_forward_score = theano.tensor.concatenate(
            [input_tv[0, -1].dimshuffle('x', 0), forward_scores])
        decode_path = do_mbr_decoding(
            augmented_forward_score, input_tv, l)

    return decode_path
