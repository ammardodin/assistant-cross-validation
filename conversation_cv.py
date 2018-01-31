from __future__ import print_function, division
from sklearn.model_selection import StratifiedKFold
from watson_developer_cloud import ConversationV1
from watson_developer_cloud.conversation_v1 import CreateIntent, CreateExample
from itertools import groupby
from operator import itemgetter
import numpy as np
import argparse
import os.path
import csv


def make_folds(data, k, monte_carlo=True):
    """
    Creates an iterator over k folds from the input data

    Args:
        data (list): A list of (example, label) pairs
        k (int): The number of folds
        monte_carlo (boolean): A flag to shuffle the data before creating folds

    Returns:
        iterator: An iterator over a list of folds
    """
    kf = KFold(n_splits=k, shuffle=monte_carlo)
    folds = kf.split(data[:, 0])
    return folds


def load_data(data_path):
    """
    Loads numpy friendly data from a workspace export

    Args:
        data_path (str): The path to the workspace exported data

    Returns:
        ndarray: A numpy array with first column containing examples,
                 and second column containing corresponding labels
    """
    if not os.path.isfile(data_path):
        raise TypeError('{} is not a valid file'.format(data_path))
    with open(data_path) as data_file:
        csv_reader = csv.reader(data_file)
        data = [line for line in csv_reader]
    return np.array(data)


def run_cross_validation(folds, data):
    """
    Creates a workspace from each fold's training data, and
    computes accuracy based on the fold's testing data

    Args:
        folds (list): An iterator over a list of folds
        data (ndarray): A numpy 2d array of the intents and examples

    Returns:
        list: A list of the accuracies from each of the created workspaces
    """

    # array to hold accuracy for each fold
    accuracies = []

    # create a conversation service instance with your credentials
    conversation = ConversationV1(username=os.environ['CONVERSATION_USERNAME'],
                                  password=os.environ['CONVERSATION_PASSWORD'],
                                  version='2017-05-26')

    # clean up from previous runs
    # existing_workspaces = conversation.list_workspaces()['workspaces']
    # for workspace in existing_workspaces:
    #     conversation.delete_workspace(workspace_id=workspace['workspace_id'])

    # loop through the k folds
    for fold in folds:
        train_indices, test_indices = fold
        train_data = data[train_indices]
        test_data = data[test_indices]
        train_map = {label: map(CreateExample, [example[0] for example in examples])
                     for label, examples in groupby(train_data, itemgetter(1))}
        create_intent_objs = [CreateIntent(label, examples=examples)
                              for label, examples in train_map.items()]

        res = conversation.create_workspace(
            name='k-fold-test', intents=create_intent_objs)
        wid = res['workspace_id']

        # wait until training is done
        while conversation.get_workspace(wid)['status'] != 'Available':
            pass

        # test the workspace with the corresponding validation data
        test_size = len(test_indices)
        correct = 0
        for example, label in test_data:
            res = conversation.message(workspace_id=wid,
                                       input={'text': example})
            # ignore counter examples
            if len(res['intents']) < 1:
                test_size -= 1
                continue
            top_label = res['intents'][0]['intent']
            if top_label == label:
                correct += 1

        # save the fold accuracy and log it
        accuracy = correct/test_size
        accuracies.append(accuracy)
        print('Accuracy: {:.4f}'.format(accuracy))

        # delete the created workspace
        conversation.delete_workspace(workspace_id=wid)

    return accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--folds', required=False, default=3, type=int)
    args = parser.parse_args()

    # get the command line arguments
    data_path = os.path.expanduser(args.data)
    num_folds = args.folds

    # load the data
    data = load_data(data_path)

    # create the folds
    folds = make_folds(data, num_folds)

    # run cross-validation
    accuracies = run_cross_validation(folds, data)
    avg_accuracy = np.average(accuracies)
    print('Average accuracy: {:.4f}'.format(avg_accuracy))
