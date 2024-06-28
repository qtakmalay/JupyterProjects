#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scores a csv file with predicted speech commands and timestamps according to the criteria specified in the task description.

For usage information, call with --help.

Author: Paul Primus
"""
import os
from argparse import ArgumentParser
import json
import csv

VALID_LABELS = {
    'Alarm an',
    'Alarm aus',
    'Fernseher an',
    'Fernseher aus',
    'Heizung an',
    'Heizung aus',
    'Licht an',
    'Licht aus',
    'Lüftung an',
    'Lüftung aus',
    'Ofen an',
    'Ofen aus',
    'Radio an',
    'Radio aus',
    'Staubsauger an',
    'Staubsauger aus'
}


def opts_parser():
    usage = \
        """
        Scores a CSV file with predicted speech commands and timestamps. 
        The result is stored in a JSON file named according to the in_file.
        """
    parser = ArgumentParser(description=usage)
    parser.add_argument('--predictions',
                        type=str, default='predictions_baseline.csv',
                        help='path to csv with predicted speech commands')
    parser.add_argument('--annotations',
                        type=str, default='development_scene_annotations.csv',
                        help='path to csv with strong annotations')
    parser.add_argument('--dump_result',
                        action='store_true',
                        help='if set, the result will be stored as a JSON file')
    parser.add_argument('--check_format',
                        action='store_true',
                        help='if set, only the structure of the predictions file will be checked')
    return parser


def compute_cost(actual_commands_per_file: dict, predicted_commands_per_file: dict) -> dict:
    """
    Computes cost from ground truth and predicted commands.

    Parameters
    ----------
    actual_commands_per_file: dict
       dictionary with filenames (without folder and file ending) as keys and a list of tuples for each event in the
       corresponding file. Each tuple has three elements: command name (str), onset in seconds (float), offset in
       seconds (float) of command.
    predicted_commands_per_file: dict
       dictionary with filenames (without folder and file ending) as keys and a list of tuples with the events in the
       corresponding file. Each tuple has two elements: command name (str), timestamp in seconds (float).

    Returns
    -------
    a dictionary holding the cost, TP, FP, FN and CT
    """

    files = list(actual_commands_per_file.keys()) + list(actual_commands_per_file.keys())
    command_labels = sorted(list(VALID_LABELS))

    # initialize counters
    TP = {l: 0 for l in command_labels}
    FP = {l: 0 for l in command_labels}
    FN = {l: 0 for l in command_labels}
    CT = {l: {k: 0 for k in command_labels} for l in command_labels}

    collar = 0.0  # could be used to allow some tolerance; not needed for this task
    # True positives first
    for f in files:
        actual_commands = actual_commands_per_file.get(f, [])  # return empty list if no annotations
        predicted_commands = predicted_commands_per_file.get(f, [])  # return empty list if no predictions
        # iterate over predictions
        for predicted in [i for i in predicted_commands]:
            cp, timestamp = predicted
            # try to find matching ground truth
            for actual in [i for i in actual_commands]:
                c, s, e = actual
                if (s - collar) <= timestamp <= (e + collar) and cp == c:
                    # matching ground truth found
                    TP[c] = TP.get(c, 0) + 1
                    # remove event from ground truth
                    predicted_commands.remove(predicted)
                    actual_commands.remove(actual)
                    continue

    # Cross triggers
    for f in actual_commands_per_file:
        actual_commands = actual_commands_per_file.get(f, [])  # return empty list if no annotations
        predicted_commands = predicted_commands_per_file.get(f, [])  # return empty list if no predictions
        # iterate over predictions
        for predicted in [i for i in predicted_commands]:
            cp, timestamp = predicted
            # try to find any matching event in  ground truth
            for actual in [i for i in actual_commands]:
                c, s, e = actual
                if (s - collar) <= timestamp <= (e + collar):
                    # matching event with mismatching label found
                    CT[c][cp] += 1
                    # remove from ground truth
                    predicted_commands.remove(predicted)
                    actual_commands.remove(actual)
                    continue

    # FP, FN
    # sort remaining events in ground truth and predictions into FP an FN
    for f in actual_commands_per_file:
        actual_commands = actual_commands_per_file.get(f, [])  # return empty list if no annotations
        predicted_commands = predicted_commands_per_file.get(f, [])  # return empty list if no predictions
        # FP
        for predicted in [i for i in predicted_commands]:
            cp, _ = predicted
            FP[cp] += 1
            predicted_commands.remove(predicted)

        # FN
        for actual in [i for i in actual_commands]:
            c, _, _ = actual
            FN[c] += 1
            actual_commands.remove(actual)

        assert len(predicted_commands) == 0
        assert len(actual_commands) == 0

    # define costs
    # could've been done with a matrix as well, but meh
    TP_costs = {l: -1 for l in command_labels}
    FN_costs = {l: 0.5 for l in command_labels}
    FP_costs = {
        'Alarm an': 4,
        'Alarm aus': 4,
        'Ofen an': 4,
        'Ofen aus': 4,
        'Heizung an': 3,
        'Heizung aus': 3,
        'Lüftung an': 3,
        'Lüftung aus': 3,
        'Licht an': 2,
        'Licht aus': 2,
        'Fernseher an': 2,
        'Fernseher aus': 2,
        'Radio an': 2,
        'Radio aus': 2,
        'Staubsauger an': 2,
        'Staubsauger aus': 2
    }

    def CT_cost(k, l):
        if k == l:
            # not a cross-trigger
            return 0
        elif k.split(' ')[0] == l.split(' ')[0]:
            # semi-bad cross-trigger
            return 0.1
        # bad cross-trigger
        return 1

    # compute costs
    total_cost = 0
    for k in command_labels:
        total_cost += TP_costs[k] * TP[k]
        total_cost += FP_costs[k] * FP[k]
        total_cost += FN_costs[k] * FN[k]
        for l in command_labels:
            total_cost += CT_cost(k, l) * CT[k][l]

    return {"cost": total_cost, 'TP': TP, 'FP': FP, 'FN': FN, 'CT': CT}


def load_ground_truth(annotations_file_path: str):
    # read ground truth
    actual_commands_per_file = {}
    with open(annotations_file_path, 'r') as file:
        for i, r in enumerate(csv.reader(file)):
            if i == 0:
                # skip header
                assert len(r) == 4 and r[0] == 'filename' and r[1] == 'command' and r[
                    2] == 'start' and r[3] == 'end', f"Header of predictions.csv is not correct. Should be filename,command,start,end but got {r}"
                continue
            r = list(r)
            event_list = actual_commands_per_file.get(r[0], [])
            # check filename, label and timestamp
            assert r[1] in VALID_LABELS, f"Unexpected label '{r[1]}'"
            assert float(r[2]) >= 0, f"Unexpected timestamp '{r[2]}'"
            assert float(r[3]) >= 0, f"Unexpected timestamp '{r[3]}'"
            assert len(r[0].split(os.sep)) == 1, f"Specify filename without folder and file ending. Yours: {r[0]}"
            assert len(r[0].split(".")) == 1, f"Specify filename without folder and file ending. Yours: {r[0]}"
            # append annotation
            event_list.append(
                (r[1], float(r[2]), float(r[3]))
            )
            actual_commands_per_file[r[0]] = event_list
    return actual_commands_per_file


def load_predictions(predictions_file_path: str):
    # read predictions
    predicted_commands_per_file = {}
    with open(predictions_file_path, 'r') as file:
        for i, r in enumerate(csv.reader(file)):
            if i == 0:
                # skip header
                assert len(r) == 3 and r[0] == 'filename' and r[1] == 'command' and r[
                    2] == 'timestamp', f"Header of predictions.csv is not correct. Should be filename,command,timestamp but got {r}"
                continue
            # get event list for current file
            event_list = predicted_commands_per_file.get(r[0], [])
            # check filename, label and timestamp
            assert r[1] in VALID_LABELS, f"Unexpected label '{r[1]}'"
            assert float(r[2]) >= 0, f"Unexpected timestamp '{r[2]}'"
            assert len(r[0].split(os.sep)) == 1, f"Specify filename without folder and file ending. Yours: {r[0]}"
            assert len(r[0].split(".")) == 1, f"Specify filename without folder and file ending. Yours: {r[0]}"
            # append to list of detected events
            event_list.append(
                (r[1], float(r[2]))
            )
            # set list
            predicted_commands_per_file[r[0]] = event_list

    return predicted_commands_per_file


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # load predictions
    predicted_commands_per_file = load_predictions(options.predictions)

    if options.check_format:
        print("The structure of your prediction file is OK.")
        return

    # load ground truth
    actual_commands_per_file = load_ground_truth(options.annotations)

    # compute cost
    result = compute_cost(actual_commands_per_file, predicted_commands_per_file)

    # dump output
    if options.dump_result:
        with open("results_" + options.predictions.split(".")[0] + '.json', 'w') as f:
            json.dump(result, f, indent=4)

    # print results
    print("Classification Report")
    print("The total cost is:", result['cost'])
    print("TP", result['TP'])
    print("FP", result['FP'])
    print("CT: ", result['CT'])
    print("FN: ", result['FN'])


if __name__ == "__main__":
    main()