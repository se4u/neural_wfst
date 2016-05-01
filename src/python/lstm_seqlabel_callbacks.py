'''
| Filename    : lstm_seqlabel_callbacks.py
| Description : Callbacks used in lstm_seqlabel training.
| Author      : Pushpendre Rastogi
| Created     : Mon Oct 26 19:57:29 2015 (-0400)
| Last-Updated: Sat Dec 19 12:22:18 2015 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 58
'''
import lstm_seqlabel_load_save_model
import subprocess
import rasengan
import cPickle as pickle

def validate_validation_result(validation_result):
    for k in validation_result:
        assert not isinstance(validation_result[k], str), validation_result
    return

def update_training_stats(validation_result, training_result, training_stats):
    validation_result = dict(validation_result)
    training_result = dict(training_result)
    try:
        training_stats['validation_result'].append(validation_result)
        training_stats['training_result'].append(training_result)
    except:
        training_stats['validation_result'] = [validation_result]
        training_stats['training_result'] = [training_result]

    if validation_result['f1'] >= training_stats['best_f1']:
        training_stats['best_f1'] = validation_result['f1']
        training_stats['best_epoch_id'] = training_stats['epoch_id']
        training_stats['worthy_epoch'] = 1
    else:
        training_stats['worthy_epoch'] = 0
    return


def update_saved_parameters(training_stats, test_stack_config, args):
    if training_stats['worthy_epoch']:
        test_stack_config['best_epoch_id'] = training_stats['best_epoch_id']
        test_stack_config['validation_result'] = training_stats['validation_result']
        test_stack_config['training_result'] = training_stats['training_result']
        try:
            lstm_seqlabel_load_save_model.save_parameters_to_file(
                test_stack_config, args.saveto)
        except pickle.PicklingError as e:
            print "Suffering from Pickling Error in saving \n%s \nto %s"%(
                ' '.join(test_stack_config.keys()),
                args.saveto)

    return


def update_saved_predictions(training_stats, args):
    if training_stats['worthy_epoch']:
        for data in ['train', 'valid']:
            source = args.folder + '/current.%s.txt'%data
            destination = (args.saveto.replace('.parameters.pickle', '') + '.%s.predictions'%data)
            print "Moving prediction file from", source, 'to', destination
            try:
                subprocess.call(['mv', source, destination])
            except:
                print 'Failed to move the file !'

    return

@rasengan.announce()
def update_learning_rate(training_stats, args):
    if args.skip_validation != 0:
        assert args.skip_validation >= args.decay_epochs

    epoch_id = training_stats['epoch_id']
    loss = training_stats['epoch_cost']
    val_accuracy = training_stats['validation_result']
    train_accuracy = training_stats['training_result']

    training_stats['clr'] = args.lr / (args.nepochs/10.0 + epoch_id + 1)**args.lr_decay_exponent
    # Epoch cost decreases
    #       validation error decreases slowly.
    #       validation error decreases fast enough.
    #       validation error does not decrease.
    #       validation error increases.
    # Epoch cost increases (Bug-in-code, lr too high)
    #       validation error decreases. Cost function is wrong,bug-in-code
    #       validation error increases. lr too high
    #       validation error stays same. lr too high, bug-in-code
    try:
        if loss[-1] > loss[-2] and loss[-2] > loss[-3]:
            print "LOSS INCREASED SUCCESSIVELY !! DECREASE LR"
            args.lr = float(args.lr) * args.lr_drop
        if (train_accuracy[-1] > train_accuracy[-2]
            and train_accuracy[-2] > train_accuracy[-3]
            and val_accuracy[-1] < val_accuracy[-2]
            and val_accuracy[-2] < val_accuracy[-3]):
            print "TRAIN ACC UP BUT VAL ACC DOWN SUCCESSIVELY!!"
        if (loss[-1] < loss[-2]
            and loss[-2] < loss[-3] # Loss decreased
            and train_accuracy[-1] < train_accuracy[-2] # Train acc decreased.
            and train_accuracy[-2] < train_accuracy[-3]):
            print "TRAIN LOSS DOWN BUT TRAIN ACC ALSO DOWN SUCCESSIVELY !!"
    except:
        pass
    if args.decay:
        epochs_without_improvement = (training_stats['epoch_id']
                                      - training_stats['best_epoch_id'])
        if epochs_without_improvement > args.decay_epochs:
            training_stats['clr'] *= 0.5
    print 'Learning rate is now', training_stats['clr']
    return


def update_global_db_of_test_results(test_stats, train_stats, args):
    global_db = args.folder + '/eval.csv'
    results = ' '.join([args.saveto,
                        str(train_stats['best_epoch_id']),
                        str(train_stats['validation_f1']),
                        'tf1=' + str(test_stats['f1']),
                        '\n'])
    print 'Updated global results db', global_db, 'with ', results
    open(global_db, 'a').write(results)
