from argparse import ArgumentParser
from data_util import calc_accuracy, load_data, save_train_test_split, train_test_split
from logreg import LogisticRegression
import numpy as np
import time


def run(constants):
    y_idx = constants['y_idx']
    numer_features = constants['numer_features']
    categ_features = constants['categ_features']
    model = LogisticRegression(constants['learning_rate'], constants['n_epochs'], verbose=constants['verbose'])
    delim = constants['delim']
    if constants['do_train']:
        if constants['train_set']:
            x_train, y_train, _ = load_data(constants['train_set'], constants['header'],
                                            y_idx, numer_features, categ_features, delim=delim)
        else:
            data, labels, df = load_data(constants['filename'], constants['header'],
                                         y_idx, numer_features, categ_features, shuffle=True, delim=delim)
            test_ratio = constants['test_ratio']
            x_train, y_train, x_test, y_test = train_test_split(data, labels, test_ratio)
            save_train_test_split(df, test_ratio, 'train.csv', 'test.csv')

        # flatten one-hot encoded values by row
        x_train = np.array([np.hstack(row) for row in x_train.values])
        print(x_train.shape)
        tic = time.time()
        model.fit(x_train, y_train, report_every=constants['report_every'])
        toc = time.time()
        print('Training time: {} secs'.format(toc - tic))
        model.save()
    else:
        model.load()
        x_test, y_test, _ = load_data(constants['test_set'], constants['header'],
                                      y_idx, numer_features, categ_features, delim=delim)
        # flatten one-hot encoded values by row
        x_test = np.array([np.hstack(row) for row in x_test.values])
        print(x_test.shape)
        y_pred = model.predict(x_test)
        print('Accuracy:', calc_accuracy(y_test, y_pred))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Logistic Regression model')
    parser.add_argument('--filename', dest='filename', type=str, help='raw (unsplit) data path')
    parser.add_argument('--header', dest='header', help='header indicator', action='store_true')
    parser.add_argument('--delim', dest='delim', type=str, help='column delimiter')
    parser.add_argument('--y-idx', dest='y_idx', type=int, help='index of label column')
    parser.add_argument('--numer-features', dest='numer_features', type=str, help='numerical feature indices as CSV')
    parser.add_argument('--categ-features', dest='categ_features', type=str, help='categorical feature indices as CSV')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float, help='test set size as ratio')
    parser.add_argument('--train-set', dest='train_set', type=str, help='training set')
    parser.add_argument('--test-set', dest='test_set', type=str, help='test set')
    parser.add_argument('--train', dest='do_train', help='training mode', action='store_true')
    parser.add_argument('--verbose', dest='verbose', help='verbose mode', action='store_true')
    parser.add_argument('--report-every', dest='report_every', type=int, help='report progress every nth step')
    parser.set_defaults(do_train=False, header=False, verbose=False)
    args = parser.parse_args()
    if args.numer_features is not None:
        args.numer_features = [int(x) for x in str(args.numer_features).split(',')]

    if args.categ_features is not None:
        args.categ_features = [int(x) for x in str(args.categ_features).split(',')]

    run(vars(args))
