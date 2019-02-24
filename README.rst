bashml (the zero dependency* ML library)
========================================

(* requires Bash 4+ for associative arrays and `bc` for floating point math)

The machine learning library in bash!

(No, I'm not seriously suggesting that anyone start doing machine learning using
shell scripting. It is too slow! Over 4 min for 10 epochs, 1 min 18 secs user time,
compared with 4 milliseconds for numpy. That is the power of vectorization and use
of optimized linear algebra libraries such as BLAS.)

The reason for the exercise was:

* Educating that ML is as much technique as tool
* Learn Bash to a deeper extent (given it is occasionally useful)
* Review understanding of the basics of ML mechanics
* As a distraction on a long plane trip; no internet, but `man` at hand

Nothing like attempting to implement an ML algorithm in a simple language (in this
case, one without native support for multidimensional arrays or even floating point
math!) to solidify understanding of both algorithm and language.

Project contents:

1. `bashml script <./bashml.sh>`_
2. Equivalent `Logistic Regression model <./logreg.py>`_ implemented using numpy
3. Equivalent `data loading and preparation utilities <./data_util.py>`_ implemented in Python and Pandas
4. `Main Python script <./main.py>`_ for benchmarking comparative performance

Try it out:

1. Ensure Bash 4+ is installed. (OS X comes with an old version of Bash (3.0) by
   default.) On OS X, you can upgrade Bash with:

   .. code-block::

     brew install bash

   This will actually install Bash 5.0.0, which is fine.

2. Download a training set, e.g. the `Titanic dataset <https://www.kaggle.com/c/titanic/download/train.csv>`_
   from the "Machine Learning from Disaster" Kaggle competition. The objective is to
   predict survival on the Titanic. (Download to project root for convenience of the
   remaining instructions.)

3. Preprocess the dataset to change the column delimiter to a character that is not
   used within a column value. (Bash and awk aren't great at processing more complex
   CSV files with quoted values. Perl or Python is typically used. However, the point
   of this exercise was to stick with pure shell script as far as possible. Therefore,
   this is a workaround step to ensure column delimiters use a unique character.)

   .. code-block::

     ./bashml.sh -p -f data/train.csv  # or
     ./pp.sh data/train.csv  # downloaded Titanic dataset

   The following files are created:

   * raw.csv - preprocessed set

4. Fit a model to the data. This step will split the file into a training and test set,
   and fit a model to the training set. The following files are created:

   * train.csv - training set
   * test.csv - test set
   * weights.txt - saved model
   * <d>_vocab.txt - a vocabulary file for each categorical feature

   .. code-block::

     ./bashml.sh -t -n 6,7,9 -c 2,4,11 -y 1 -r .2 --header -d ';' -f raw.csv -v 1 -e 50  # or
     ./train.sh 50  # number epochs

   The following attributes are being used as features:

   * SibSp (numerical, column index 6) - number of siblings/spouse aboard
   * Parch (numerical, column index 7) - number parents/children aboard
   * Fare (numerical, column index 9) - Passenger fare
   * Pclass (categorical, column index 2) - Ticket class: 1 - 1st, 2 - 2nd, 3 - 3rd
   * Sex (categorical, column index 4) - Gender
   * Embarked (categorical, column index 11) - Port of embarcation: C - Cherbourg, Q - Queenstown, S - Southhampton

5. Use the test set to make predictions and report on accuracy. This step expects the
   above files to be in place.

   .. code-block::

     ./bashml.sh -n 6,7,9 -c 2,4,11 -y 1 -d ';' -f test.csv -v 1  # or
     ./predict.sh

::

   Usage:
   ./bashml.sh ...args

   Arguments:
   -f | --file            : data file
   -o | --pp_file         : preprocessed file
   -d | --delimiter       : column delimiter
   -s | --repl_delim      : replacement column delimiter
   -a | --header          : file has header
   -n | --numer_feat_idxs : numerical feature indices
   -c | --categ_feat_idxs : categorical feature indices
   -y | --y_i             : column index of output variable
   -l | --learning_rate   : learning rate
   -e | --n_epochs        : number epochs
   -r | --test_ratio      : test ratio
   -t | --train           : training mode
   -p | --preprocess      : preprocess data file
   -v | --verbose         : print debug info
   -h | --help            : show this usage information


Accuracy on Titanic dataset using the base attributes above, and training for 50 epochs is:

**bashml:**
::

   Epoch 45
   Loss: 0.65853
   Epoch 46
   Loss: 0.65785
   Epoch 47
   Loss: 0.65718
   Epoch 48
   Loss: 0.65651
   Epoch 49
   Loss: 0.65585
   Epoch 50
   Loss: 0.65519

   Accuracy: 0.79

**Python numpy:**
::

   Epoch 45
   loss: 0.65786
   Epoch 46
   loss: 0.65719
   Epoch 47
   loss: 0.65652
   Epoch 48
   loss: 0.65585
   Epoch 49
   loss: 0.65519
   Epoch 50
   loss: 0.65454

   Accuracy: 0.79

So very similar results for given features and number of training iterations.
