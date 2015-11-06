import os
import sys
from sys import argv

path = os.getcwd()
sys.path.append(path)

import pandas as pd
import numpy as np
from collections import defaultdict

from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters
from blind_dataset_metrics import scores, read_blind_predictions
from pan_allele.helpers.peptide_trim import make_prediction

def split_train_test(arr, k_fold):
    train = [x for i, x in enumerate(arr) if i%k_fold != 1]
    test = [x for i, x in enumerate(arr) if i%k_fold == 1]
    return np.array(train), np.array(test)



allele_groups, df = load_binding_data('pan_allele/files/bdata.2009.mhci.public.1.txt')
graph = get_graph_from_hyperparameters('ffn_mult')
allele_sequence_data, max_allele_length = load_allele_sequence_data('pan_allele/files/pseudo/pseudo_sequences.fasta')
allele_list = sorted(create_allele_list(allele_groups, allele_sequence_data))



blind_allele_groups, blind_df = load_binding_data('blind_data.txt')
blind_allele_list = sorted(create_allele_list(blind_allele_groups, allele_sequence_data))


print blind_allele_list
nb_iter = 1
preds_allele = defaultdict(list)
actual_allele = defaultdict(list)



for i in range(0,nb_iter):

    peptides, mhc, Y = get_model_data(  allele_list,
                                        allele_sequence_data,
                                        allele_groups,
                                        dense_mhc_model=None,
                                        peptide_length = 9,
                                        mhc_length=max_allele_length,
                                        mhc_dense = None
                                     )

    peptides_train, peptides_test = split_train_test(peptides,5)
    mhc_train, mhc_test = split_train_test(mhc,5)
    Y_train, Y_test = split_train_test(Y,5)

    graph = get_graph_from_hyperparameters('ffn_mult')
    graph.fit({'peptide':peptides, 'mhc':mhc, 'output': Y},
                batch_size=32,
                nb_epoch=10,
                verbose = 1,
                )
    for allele in blind_allele_list:
        predictions = read_blind_predictions('combined-test-data/'+ allele + '.csv')
        peptides = predictions.keys()
        preds = []
        meas = []
        for peptide in peptides:
            preds.append(make_prediction(peptide, allele_sequence_data[allele],graph))
            meas.append(predictions[peptide]['meas'])
        preds = np.array(preds)
        meas = np.array(meas)
        try:
            preds_allele[allele]+=(20000**(1-preds))/nb_iter
        except:
            preds_allele[allele]=(20000**(1-preds))/nb_iter
        actual_allele[allele] = meas


#sum_scores = np.zeros(6)
calculated_metrics = np.zeros(6)
for allele in blind_allele_list:
    Y_pred_allele = preds_allele[allele]
    Y_true_allele = actual_allele[allele]
    score_allele = scores(Y_true_allele, Y_pred_allele)
    calculated_metrics  += score_allele[0] * score_allele

data_len = sum(len(read_blind_predictions('combined-test-data/'+ allele + '.csv').keys()) for allele in allele_list)
print calculated_metrics/data_len
