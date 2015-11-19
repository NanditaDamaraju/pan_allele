from paths import *
import sys
sys.path.append(HOME_PATH)

from pan_allele.helpers.peptide_trim import make_prediction
from pan_allele.helpers.generate_pseudo_sequences import create_fasta_file
from pan_allele.helpers.pan_allele_data_helpers import load_allele_sequence_data
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters

from keras.models import Graph
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import numpy as np
import collections
import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pred",
    default='ffn_mult',
    help="neural network type, `ffn_concat`, `ffn_mult` or `conv_mult`")

parser.add_argument(
    "--epoch",
    default="25,26",
    help="model at which epoch to choose")


#file0 negative prediction and file 1 positive predicitons
def read_tcell_predictions(file0, file1):
    predictions = collections.defaultdict(dict)
    with open(file0, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
            peptide = row[2]
            allele = row[1]
            predictions[allele][peptide] = 0
    with open(file1, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
            peptide = row[2]
            allele = row[1]
            predictions[allele][peptide] = 1
    return predictions


def scores(Y_true_binary, Y_pred_log):

    Y_true_binary = np.array(Y_true_binary)
    Y_pred_log = np.array(Y_pred_log)

    Y_pred_binary = max_ic50**(1 - Y_pred_log) <=500

    AUC = 0

    ACC = accuracy_score(Y_true_binary, Y_pred_binary)
    F1 = f1_score(Y_true_binary, Y_pred_binary)
    recall = recall_score(Y_true_binary, Y_pred_binary)
    precision = precision_score(Y_true_binary, Y_pred_binary)
    length = len(Y_true_binary)

    if(Y_true_binary.all() or not Y_true_binary.any()):
        pass
    else:
        AUC = roc_auc_score(Y_true_binary, Y_pred_log)

    return length, AUC, ACC, F1, precision, recall


def main():
    args = parser.parse_args()

    graph = get_graph_from_hyperparameters(args.pred)
    batch_size = 32
    max_ic50 = 50000
    ##Load graph
    epoch_range = map(int, args.epoch.split(','))
    #graph.set_weights(initial_weights)
    for epoch in range(epoch_range[0], epoch_range[1]) :
        graph.load_weights(HOME_PATH + '/weights'+str(max_ic50)+'/' + args.pred + '/weights' + str(batch_size)+ '_'  + str(epoch) )
        allele_sequence_data, max_allele_length = load_allele_sequence_data('pan_allele/files/pseudo/pseudo_sequences.fasta')

        predictions = read_tcell_predictions('paper_data/iedb-tcell-2009-negative.csv','paper_data/iedb-tcell-2009-positive.csv')


        allele_list = sorted(predictions.keys())
        allele_list[:] = [x for x in allele_list if not x.startswith('C')]
        Y_true = []
        Y_pred = []
        for allele in allele_list:

            peptides = predictions[allele].keys()
            for peptide in peptides:
                if(len(peptide)>7 and len(peptide)<12):
                    #print allele, peptide, predictions[allele][peptide], 20000**(1-make_prediction(peptide, allele_sequence_data[allele], graph))
                    Y_true.append( predictions[allele][peptide])
                    Y_pred.append(make_prediction(peptide, allele_sequence_data[allele], graph))
            #print "=====", allele, sum(Y_true), len(Y_true), "===="
        score = scores(Y_true, Y_pred)
        print epoch, ','.join(map(str,score[1:]))

if __name__ == "__main__":
    main()
