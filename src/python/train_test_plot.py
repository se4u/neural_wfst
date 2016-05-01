#!/usr/bin/env python
'''
| Filename    : train_test_plot.py
| Description : Plot the training and validation convergence performance
| Author      : Pushpendre Rastogi
| Created     : Thu Apr  7 13:32:40 2016 (-0400)
| Last-Updated: Thu Apr  7 14:17:00 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 18
'''
import cPickle as pkl
plt = None
def load_modules():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
    matplotlib.pyplot.style.use('ggplot')
    global plt
    plt = matplotlib.pyplot
    return
load_modules()
for task in '13SIA-13SKE 2PIE-13PKE  2PKE-z rP-pA'.split():
    for fold in range(5):
        for trial in range(1,4):
            if (task, fold, trial) in [('rP-pA', 4, 3),
                                        ('13SIA-13SKE', 3, 3),
                                        ('2PKE-z', 1, 1)]:
                title = 'task=%s_fold=%d_trial=%d'%(task, fold, trial)
                title_out = 'task=%s_fold=%d_trial=%d'%(task, fold + 1, trial)
                f = '../../results/transducer_full_decomp_jason_win1_amach_%s_0/transducer.pkl'%title
                d = pkl.load(open(f))
                if len(d['training_result']) > 100:
                    training_result = [e['f1'] for e in d['training_result']]
                    validation_result = [e['f1'] for e in d['validation_result']]
                    x = range(len(training_result))
                    plt.plot(x, training_result)
                    plt.plot(x, validation_result)
                    plt.xlabel('Iterations')
                    plt.ylabel('Accuracy')
                    plt.title(title_out)
                    plt.grid(True)
                    plt.legend(['training_result', 'validation_result'], loc='upper left')
                    plt.savefig(title_out + '.png')
                    plt.close()
