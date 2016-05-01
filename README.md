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

The following command would train the neural transducer model on the `4th` fold of the `rP-pA` morphological transduction task. The test file is not used at this point.

     PYTHONPATH=$PWD/src/python python -c "import transducer_score; print (
           transducer_score.main(train_fn='res/celex/rP-pA/0500/4/train',
                                 dev_fn='res/celex/rP-pA/0500/4/dev',
                                 test_fn='res/celex/rP-pA/0500/4/test',
                                 folder='results/tmp'))"

Once the model is trained and stored in the `tmp` directory we can test the model as follows:

    PYTHONPATH=$PWD/src/python python -c "import transducer_score; print (
          transducer_score.main(train_fn='res/celex/rP-pA/0500/4/train',
                                dev_fn='res/celex/rP-pA/0500/4/dev',
                                test_fn='res/celex/rP-pA/0500/4/test',
                                folder='results/tmp2',
                                pretrained_param_pklfile='results/tmp/transducer.pkl',
                                perform_training=0,
                                perform_testing=1,
                                nepochs=-1)"


For more complicated usage, including the exact parameters that were used to obtain the results in the paper, see the scripts

    src/python/transducer_celex.sh
    src/python/transducer_celex_test.sh

These scripts contain the parameter strings that were used to obtain all the results in the paper.
