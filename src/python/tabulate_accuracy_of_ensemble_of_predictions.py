#!/usr/bin/env python
'''
| Filename    : tabulate_accuracy_of_ensemble_of_predictions.py
| Description : A simple majority finder.
| Author      : Pushpendre Rastogi
| Created     : Tue Jan  5 12:01:53 2016 (-0500)
| Last-Updated: Tue Jan  5 12:35:26 2016 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 3
for task in 13SIA-13SKE  2PIE-13PKE  2PKE-z  rP-pA ; do for fold in 0 2 3 4; do ./tabulate_accuracy_of_ensemble_of_predictions.py --path  ../../results/transducer_[da][eb][el][pa][_t]*ask=${task}_fold=${fold}_trial=1_0/transducer.pkl.valid.predictions --default_idx 8; done; done
'''

import argparse
from collections import Counter

def get_data_impl(fn):
    prediction_list = []
    truth_list = []
    with open(fn) as fh:
        fh.next()
        for row in fh:
            row = row.strip().split()
            prediction_list.append(row[1])
            truth_list.append(row[2])
    return prediction_list, truth_list

def get_data():
    predictions = []
    truth = None
    for fn in args.path:
        prediction_list, truth_tmp = get_data_impl(fn)
        if truth is None:
            truth = truth_tmp
        else:
            assert truth == truth_tmp
        predictions.append(prediction_list)
        pass
    return predictions, truth

def accuracy(a, b):
    assert len(a) == len(b)
    return float(sum(e1 == e2 for e1, e2 in zip(a,b)))/len(a)


def majority(predictions, default_idx=0):
    ret_list = []
    for output_tuple in zip(*predictions):
        data = Counter(output_tuple)
        major, count = data.most_common(1)[0]
        ret_list.append((output_tuple[default_idx]
                         if count == 1
                         else major))
    return ret_list

def main():
    predictions, truth = get_data()
    majority_prediction = majority(predictions, default_idx=args.default_idx)
    for fn, prediction in zip(args.path, predictions):
        print fn, 'Acc=', accuracy(prediction, truth)
    print 'Majority Accuracy=', accuracy(majority_prediction, truth)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--path', default=[], nargs='*')
    arg_parser.add_argument('--default_idx', default=0, type=int,
                            help='The default file chosen in case of tie.')
    args=arg_parser.parse_args()
    main()
