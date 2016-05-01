'''
| Filename    : test_conjunctivemixture.py
| Description : Test ConjunctiveMixture Chip
| Author      : Pushpendre Rastogi
| Created     : Mon Nov 16 01:00:45 2015 (-0500)
| Last-Updated: Mon Nov 16 01:08:20 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 3
'''
import rasengan
import lstm_seqlabel_circuit
import lstm_seqlabel_circuit_compilation
import util_lstm_seqlabel

args = rasengan.Namespace()
args.conjmix_embed_BOS = 1
args.conjmix_clip_gradient = 0
chips = [(lstm_seqlabel_circuit.ConjunctiveMixture, 'conjmix')]
with util_lstm_seqlabel.debug_support():
    ttns = lstm_seqlabel_circuit_compilation.get_train_test_namespace(args)

# Test value of ttns
