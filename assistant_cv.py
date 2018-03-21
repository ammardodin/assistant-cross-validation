from __future__ import print_function, division
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from watson_developer_cloud import AssistantV1
from watson_developer_cloud.assistant_v1 import CreateIntent, CreateExample
from itertools import groupby, repeat
from operator import itemgetter
from threading import Thread
from multiprocessing.pool import ThreadPool
import numpy as np
import argparse
import os.path
import csv


def make_folds(data, k, monte_carlo=True):
    """
    Creates an iterator over k folds from the input data.

    Args:
        data (list): A list of (example, label) pairs.
        k (int): The number of folds.
        monte_carlo (boolean): A flag to shuffle the data before creating folds.

    Returns:
        iterator: An iterator over a list of folds.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=monte_carlo)
    folds = skf.split(data[:, 0], data[:, 1])
    return folds


def load_data(data_path):
    """
    Loads numpy friendly data from a workspace export.

    Args:
        data_path (str): The path to the workspace exported data.

    Returns:
        ndarray: A numpy array with first column containing examples,
                 and second column containing corresponding labels.
    """
    if not os.path.isfile(data_path):
        raise TypeError('{} is not a valid file'.format(data_path))
    with open(data_path) as data_file:
        csv_reader = csv.reader(data_file)
        data = [line for line in csv_reader]
    return np.array(data)


def run_cross_validation(fold, data):
    """
    Creates a workspace from the fold's training data, and
    computes classification metrics based on the fold's testing data.

    Args:
        fold (tuple): A tuple of train and test indices. 
        data (ndarray): A numpy 2d array of the intents and examples.

    Returns:
        str: A classification report string from the created workspace.
    """

    # create an assistant service instance with your credentials
    assistant = AssistantV1(username=os.environ['ASSISTANT_USERNAME'],
                            password=os.environ['ASSISTANT_PASSWORD'],
                            version='2017-05-26')

    train_indices, test_indices = fold
    train_data = data[train_indices]
    test_data = data[test_indices]
    train_map = {label: map(CreateExample, [example[0] for example in examples])
                 for label, examples in groupby(train_data, itemgetter(1))}
    create_intent_objs = [CreateIntent(label, examples=examples)
                          for label, examples in train_map.items()]

    res = assistant.create_workspace(
        name='k-fold-test', intents=create_intent_objs)
    wid = res['workspace_id']

    # wait until training is done
    while assistant.get_workspace(wid)['status'] != 'Available':
        pass

    # validate the workspace with the corresponding test data
    y_true, y_pred = [], []
    for example, label in test_data:
        res = assistant.message(workspace_id=wid,
                                input={'text': example})
        # ignore counter examples
        if len(res['intents']) < 1:
            continue
        top_label = res['intents'][0]['intent']
        y_true.append(label)
        y_pred.append(top_label)

    # delete the created workspace
    assistant.delete_workspace(workspace_id=wid)

    # return the classification report string
    return classification_report(y_true, y_pred)


def run_cross_validation_helper(params):
    return run_cross_validation(*params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help='path to the workspace csv export')
    parser.add_argument('--folds', required=False, default=3,
                        type=int, help='number of folds for cross-validation')
    args = parser.parse_args()

    # get the command line arguments
    data_path = os.path.expanduser(args.data)
    num_folds = args.folds

    # load the data
    data = load_data(data_path)

    # create the folds
    folds = make_folds(data, num_folds)

    # create worker threads
    pool = ThreadPool(processes=num_folds)

    # fire up the workers
    classification_reports = pool.map(
        run_cross_validation_helper, zip(folds, repeat(data, num_folds)))

    pool.close()
    pool.join()

    for fold_number, report in enumerate(classification_reports, 1):
        print('******************fold_{}_results********************\n'.format(fold_number))
        print(report)
        print('****************************************************\n')
