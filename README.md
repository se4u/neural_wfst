# README #

NAACL 2016 submission "Structured Prediction with Neural Context Features"
By Pushpendre Rastogi, Ryan Cotterell, Jason Eisner

## Bibtex ##

    @conference{rastogi2016weighting,
	Author = {Pushpendre Rastogi and Ryan Cotterell and Jason Eisner},
	Booktitle = {Proceedings of NAACL},
	Date-Added = {2016-04-09 07:28:31 -0400},
	Date-Modified = {2016-04-09 07:32:46 -0400},
	Keywords = {Neural Network,Neural Symbolic,Finite State Transducer},
	Title = {Weighting Finite-State Transductions With Neural Context},
	Year = {2016},
	Abstract = {How should one apply deep learning to tasks such as morphological reinflection, which stochastically edit one string to get another? A recent approach to such sequence-to-sequence tasks is to compress the input string into a vector that is then used to generate the output string, using recurrent neural networks. In contrast, we propose to keep the traditional architecture, which uses a finite-state transducer to score all possible output strings, but to augment the scoring function with the help of recurrent networks. A stack of bidirectional LSTMs reads the input string from left-to-right and right-to-left, in order to summarize the input context in which a transducer arc is applied. We combine these learned features with the transducer to define a probability distribution over aligned output strings, in the form of a weighted finite-state automaton. This reduces hand-engineering of features, allows learned features to examine unbounded context in the input string, and still permits exact inference through dynamic programming. We illustrate our method on the tasks of morphological reinflection and lemmatization.}}

## Instructions ##

Run the following command to compile the WFST portion of the model:

         $ cd src/python/transducer
         $ make # Make transducer.so and copy to src
         $ cd - # Go back to toplevel


The following command will train the neural transducer model on the `4th` fold of the `rP-pA` morphological transduction task. The test file is not used at this point.

     PYTHONPATH=$PWD/src/python THEANO_FLAGS=floatX=float32 python -c "import transducer_score; print (
           transducer_score.main(train_fn='res/celex/rP-pA/0500/4/train',
                                 dev_fn='res/celex/rP-pA/0500/4/dev',
                                 test_fn='res/celex/rP-pA/0500/4/test',
                                 folder='results/tmp'))"

Once the model is trained and stored in the `tmp` directory we can test the model as follows:

    PYTHONPATH=$PWD/src/python THEANO_FLAGS=floatX=float32 python -c "import transducer_score; print (
          transducer_score.main(train_fn='res/celex/rP-pA/0500/4/train',
                                dev_fn='res/celex/rP-pA/0500/4/dev',
                                test_fn='res/celex/rP-pA/0500/4/test',
                                folder='results/tmp2',
                                pretrained_param_pklfile='results/tmp/transducer.pkl',
                                perform_training=0,
                                perform_testing=1,
                                nepochs=-1)"


For more complicated usage, including the exact parameters that were used to obtain the ablation results in the paper, see the scripts

    src/python/transducer_celex.sh
    src/python/transducer_celex_test.sh

These scripts contain the parameter strings that were used to obtain all the results in the paper.

## FAQ ##

1. How to fix `ImportError: No module named transducer`?

   If you get the following error:

         File "neural_wfst/src/python/transducer_score.py", line 25, in <module>
                 from transducer.src.transducer import Transducer
         ImportError: No module named transducer

   Then run the following command:

         $ cd src/python/transducer
         $ make # Make transducer.so and copy to src

   See the `Makefile` in `src/python/transducer` to understand what's going on.
   In case there are further errors during compilation, then please raise an issue.
   
2. Can I get the model predictions used in Table 1 and Table 2 of the paper ? 
   
   See the files `lemmatization_results.tgz` and `celex_results.zip` files. These archives contain many folders and each folder contains a single file `current.test.txt` which contains the predictions of our model. Each file contains lines formatted like:
   
   ```
   input prediction goldOutput
   ^schwefelte schwefele schwefle
   ^flunkerte flunkere flunkere
   ^kuesste kuesse kuesse
   ^erahnte erahne erahne
   ^maulte maule maule 
   ```
   
   For example the `celex_results.zip` archive contains the following folders. The tasks are the `13SIA 2PIE 2PKE rP` tasks. And the folds range from `0` to `4`.
   
   ```
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=13SIA-13SKE_fold=0_2
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=13SIA-13SKE_fold=1_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=13SIA-13SKE_fold=2_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=13SIA-13SKE_fold=3_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=13SIA-13SKE_fold=4_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PIE-13PKE_fold=0_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PIE-13PKE_fold=1_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PIE-13PKE_fold=2_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PIE-13PKE_fold=3_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PIE-13PKE_fold=4_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PKE-z_fold=0_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PKE-z_fold=1_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PKE-z_fold=2_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PKE-z_fold=3_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=2PKE-z_fold=4_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=rP-pA_fold=0_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=rP-pA_fold=1_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=rP-pA_fold=2_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=rP-pA_fold=3_0
   drwxr-xr-x 2 prastog3 fax 29 Dec 28  2015 transtest_transducer_deep_task=rP-pA_fold=4_0
   ```
   
