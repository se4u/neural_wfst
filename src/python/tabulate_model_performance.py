#!/usr/bin/env python
'''
| Filename    : tabulate_model_performance.py
| Description : Tabulate the performance of saved model files.
| Author      : Pushpendre Rastogi
| Created     : Tue Dec 15 23:20:20 2015 (-0500)
| Last-Updated: Fri Apr  8 15:31:51 2016 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 65
'''
from lstm_seqlabel_circuit_compilation import print_pklfn_performance
import rasengan
import argparse
import os, re, sys
TRANSDUCER_NAME_REG_STR = r'\.\./\.\./results/([^/]+)/transducer\.pkl'
TRANSDUCER_NAME_REG = re.compile(TRANSDUCER_NAME_REG_STR)

def map_task_to_epochs(args):
    SAVETO_REG = re.compile(
        r'.*Set args\.saveto= ' + TRANSDUCER_NAME_REG_STR,
        re.DOTALL)
    EPOCH_REG = re.compile(
        r".*\('epoch_id:', (\d+), 'Training F1:', \d+\.\d+, 'Validation F1:', \d+\.\d+\)",
        re.DOTALL)
    folder = r'/home/prastog3/projects/neural-context/results/transducer_deeplog'
    log_filenames=os.listdir(folder)
    model_run_to_total_epochs_map={}
    path_to_log_filenames = [re.sub(r'../../results/(.*?)_task=(.*?)_fold=([0-9])_trial=([0-9]).*?/transducer.pkl',r'\1_\2.\3.\4.', e)
                             for e in args.path]
    compliant_log_fn = [e for e in log_filenames
                        if any(_ in e for _ in path_to_log_filenames)]
    if len(compliant_log_fn) == len(args.path):
        log_filenames = compliant_log_fn

    for fn in log_filenames:
        print >> sys.stderr, folder, fn
        log = open(os.path.join(folder, fn), 'rb').read()
        try:
            model_run = SAVETO_REG.match(log).group(1)
            total_epochs = EPOCH_REG.match(log).group(1)
            model_run_to_total_epochs_map[model_run] = (total_epochs, fn)
        except AttributeError:
            pass
    return model_run_to_total_epochs_map

def the_printing_method(path, args, model_run_to_total_epochs_map, add_newline=True):
    args.pretrained_param_pklfile = path
    try:
        pkl, arr = print_pklfn_performance(args, add_newline=False)
    except TypeError:
        return None, None

    model_run = TRANSDUCER_NAME_REG.match(path).group(1)
    try:
        total_epochs, _fn = model_run_to_total_epochs_map[model_run]
    except KeyError:
        total_epochs = -1
        pass
    arr2 = [' Total Epochs', total_epochs]
    arr.extend(arr2)
    for e in arr2:
        print e,
    if add_newline:
        print
    return pkl, arr

def server_method(args):
    import spyne, twisted
    from twisted.internet import reactor
    from twisted import web
    import urllib
    from spyne.protocol import http
    from spyne.protocol import csv
    from spyne.server import twisted

    class HelloWorldService(spyne.service.ServiceBase):
        @spyne.decorator.srpc(
            spyne.model.primitive.Unicode,
            _returns=spyne.model.complex.Iterable(
                spyne.model.primitive.Unicode))
        def my_method(path):
            path = urllib.unquote(path)
            model_run_to_total_epochs_map = map_task_to_epochs()
            yield the_printing_method(path, args, model_run_to_total_epochs_map)[1]

    application = spyne.application.Application(
        [HelloWorldService],
        tns='spyne.examples.hello',
        in_protocol=spyne.protocol.http.HttpRpc(validator='soft'),
        out_protocol=spyne.protocol.csv.Csv())

    reactor.listenTCP(
        8729,
        web.server.Site(
            spyne.server.twisted.TwistedWebResource(
                application)),
        interface='0.0.0.0')

    #------------------------------#
    # Where it all comes together. #
    #------------------------------#
    reactor.run()
    return

def client_method(path):
    import urllib, requests
    url = r'http://localhost:8729/my_method?path=%s'%(
        urllib.quote(path, safe=''))
    return requests.get(url)

def main(args, add_newline=True):
    model_run_to_total_epochs_map = map_task_to_epochs(args)
    if args.server:
        server_method(args)
        print "server setup complete"
    elif args.client:
        for path in args.path:
            val = client_method(path)
            for e in val:
                print e,
            print
    else:
        for path in args.path:
            pkl, arr = the_printing_method(
                path, args, model_run_to_total_epochs_map, add_newline=add_newline)
            #----------------------------------------------------------------------#
            # Print `keys` from pkl file that were specially mentioned on cmdline. #
            #----------------------------------------------------------------------#
            for k in args.keys:
                print k, pkl[k]
            #---------------------------------------------------------#
            # In case we want to interact with the pkl after loading. #
            #---------------------------------------------------------#
            if args.interact:
                import readline, code
                print pkl.keys()
                code.InteractiveConsole(pkl).interact()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
    description='Tabulate performance of saved model files.')
    arg_parser.add_argument(
        '--path', nargs='*', default=[],
        help='A glob of the paths to the pkls')
    arg_parser.add_argument(
        '--interact', default=0, type=int,
        help='Default={0}')
    arg_parser.add_argument(
        '--keys', nargs='*', default=[],
        help='Default={0}')
    arg_parser.add_argument('--server', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--client', default=0, type=int, help='Default={0}')
    with rasengan.debug_support():
        main(args=arg_parser.parse_args())
