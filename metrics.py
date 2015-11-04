
'''
Takes the model type as input and provides the
AUC, Accuracy, F1, precision and recall
'''

import sys
import os
path = os.getcwd()
sys.path.append(path)


from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters
from pan_allele.helpers.peptide_trim import make_prediction
from pan_allele.helpers.pan_allele_data_helpers import *

from keras.models import Graph
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import numpy as np
import collections
import pandas as pd
import csv
import argparse

np.set_printoptions(precision=3,suppress=True)

max_ic50 = 20000
ic50_cutoff = 500



parser = argparse.ArgumentParser()

parser.add_argument(
    "--pred",
    default='ffn_mult',
    help="neural network type, `ffn_concat`, `ffn_mult` or `conv_mult`")

parser.add_argument(
    "--epoch",
    default="25,26",
    help="model at which epoch to choose")

parser.add_argument(
    "--allele_info",
    default=False,
    type=bool,
    help="display allele information or not"

)

def scores(Y_true, Y_pred):
    Y_true_binary = Y_true <=ic50_cutoff
    Y_pred_binary = Y_pred <= ic50_cutoff

    Y_pred_log = 1 - np.log(Y_pred)/np.log(max_ic50)

    ACC = accuracy_score(Y_true_binary, Y_pred_binary)
    F1 = f1_score(Y_true_binary, Y_pred_binary)
    recall = recall_score(Y_true_binary, Y_pred_binary)
    precision = precision_score(Y_true_binary, Y_pred_binary)
    length = len(Y_true)
    AUC = 0

    if(Y_true_binary.all() or not Y_true_binary.any()):
        print "Skipping as all labels are the same"
    else:
        AUC = roc_auc_score(Y_true_binary, Y_pred_log)
    return np.array((length, AUC, ACC, F1, precision, recall))

def read_blind_predictions(filename):
    predictions = collections.defaultdict(dict)
    with open(filename, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
                for i,val in enumerate(header):
                    try:
                        predictions[row[0]][val] = float(row[i])
                    except:
                        pass
    return predictions

def main():

    #prediction input either "conv", "ffn_concat", "ffn_mult"
    args = parser.parse_args()

    graph = get_graph_from_hyperparameters(args.pred)

    predictors = ['mhcflurry', 'netmhcpan', 'netmhc', 'smmpmbec_cpp']
    #allele_list

    allele_list = ['A0101',	    'A0201',	'A0202',    'A0203',	'A0206',	'A0301',
                   'A1101',	    'A2301',	'A2402',	'A2501',	'A2601',    'A2602',
                   'A2603',	    'A2902',	'A3001',	'A3002',	'A3101',	'A3201',
                   'A3301',	    'A6801',	'A6802',	'A6901',    'A8001',	'B0702',
                   'B0801',	    'B0802',	'B0803',	'B1501',	'B1503',    'B1509',
                   'B1517',	    'B1801',	'B2703',    'B2705',    'B3501',	'B3801',
                   'B3901',	    'B4001',	'B4002',	'B4402',	'B4403',	'B4501',
                   'B4601',	    'B5101',    'B5301',	'B5401',	'B5701',	'B5801'	]


    #Load graph
    batch_size = 32
    epoch_range = map(int, args.epoch.split(','))

    for epoch in range(epoch_range[0],epoch_range[1]):

        graph.load_weights('weights/' + args.pred + '/weights' + str(batch_size) + '_'  + str(epoch) )

        #Initializing
        data_len = sum(len(read_blind_predictions('combined-test-data/'+ allele + '.csv').keys()) for allele in allele_list)
        Y_true_all = np.zeros(data_len)
        total_metrics = collections.defaultdict(list)
        for val in predictors:
                total_metrics[val] =  np.zeros(data_len)

        pos  = 0
        calculated_metrics =collections.defaultdict(tuple)
        for val in predictors:
            calculated_metrics[val] = np.zeros(6)



        #calculating metrics per allele
        for allele in allele_list:

            filename = 'combined-test-data/'+ allele + '.csv'
            predictions = read_blind_predictions(filename)

            peptides = predictions.keys()
            allele_sequence_data, max_allele_length = load_allele_sequence_data('pan_allele/files/pseudo/pseudo_sequences.fasta')
            for peptide in peptides:
                predictions[peptide]['mhcflurry'] = 20000**(1-make_prediction(peptide, allele_sequence_data[allele], graph))
            df_pred = pd.DataFrame(predictions)


            Y_true_allele = np.array(df_pred.loc['meas'])
            Y_true_all[pos:pos+len(peptides)] =  Y_true_allele

            if (args.allele_info == True):
                print "\n=====", allele, sum(Y_true_allele <= 500), len(Y_true_allele), "===="

            for val in predictors:
                Y_pred_allele = np.array(df_pred.loc[val])
                calculated_metrics[val]  += len(peptides)*scores(Y_true_allele, Y_pred_allele)
                if (args.allele_info == True):
                    print val, scores(Y_true_allele, Y_pred_allele)
                total_metrics[val][pos:pos+len(peptides)] = (Y_pred_allele)

            pos +=len(peptides)

        print "\n",epoch
        print "AUC\tACC\tF1\tPre\tRecall"

        for val in predictors:
            calculated_metrics[val] = calculated_metrics[val]/data_len
            print "\n",val
            scores_val = scores(Y_true_all, total_metrics[val])
            print scores_val[1:]
            print calculated_metrics[val][1:]


if __name__ == "__main__":
    main()
