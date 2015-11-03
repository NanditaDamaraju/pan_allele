
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

from keras.models import Graph
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import numpy as np
import collections
import pandas as pd
import csv
import argparse




max_ic50 = 20000
ic50_cutoff = 500
log_transformed_ic50_cutoff = 1 - np.log(ic50_cutoff)/np.log(max_ic50)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--pred",
    default='ffn_mult',
    help="neural network type, `ffn_concat`, `ffn_mult` or `conv_mult`")

parser.add_argument(
    "--epoch",
    default=25,
    type=int,
    help="model at which epoch to choose")


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
    return length, AUC, ACC, F1, precision, recall

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
    lr = 0.001
    batch_size = 32
    graph.load_weights('weights/' + pred + '/weights' + str(batch_size)+ '_' + str(lr) + '_'  + str(args.epoch) )

    #Initializing
    data_len = sum(len(read_blind_predictions('combined-test-data/'+ allele + '.csv').keys()) for allele in allele_list)
    Y_true_all = np.zeros(data_len)
    total_metrics = collections.defaultdict(list)
    for val in predictors:
            total_metrics[val] =  np.zeros(data_len)

    pos  = 0
    calculated_metrics =collections.defaultdict(tuple)
    for val in predictors:
        calculated_metrics[val] = (0,0,0,0,0,0)



    #calculating metrics per allele
    for allele in allele_list:
        filename = 'combined-test-data/'+ allele + '.csv'
        predictions = read_blind_predictions(filename)
        peptides = predictions.keys()
        for peptide in peptides:
            predictions[peptide]['mhcflurry'] = 20000**(1-make_prediction(peptide, allele_sequence_data[allele], graph))
        df_pred = pd.DataFrame(predictions)


        Y_true_allele = np.array(df_pred.loc['meas'])
        Y_true_all[pos:pos+len(peptides)] =  Y_true_allele

        #print "\n=====", allele, sum(Y_true_allele <= 500), len(Y_true_allele), "===="
        for val in predictors:
            Y_pred_allele = np.array(df_pred.loc[val])
            calculated_metrics[val]  = map(sum, zip(scores(Y_true_allele, Y_pred_allele), calculated_metrics[val]))
            print val, scores(Y_true_allele, Y_pred_allele)
            total_metrics[val][pos:pos+len(peptides)] = (Y_pred_allele)

        pos +=len(peptides)
    print calculated_metrics
    print "\n",num

    for val in predictors:
        print "\n",val
        scores_val = scores(Y_true_all, total_metrics[val])
        print scores_val

if __name__ == "__main__":
    main()
