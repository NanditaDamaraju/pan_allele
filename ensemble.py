import os
import sys
from sys import argv

path = os.getcwd()
sys.path.append(path)

import pandas as pd
import numpy as np

from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters
from blind_dataset_metrics import scores, blind_predict,  read_blind_predictions

def split_train_test(arr, k_fold):
    indices = np.arange(len(arr))
    np.random.shuffle(indices)
    arr = arr[indices]
    train = [x for i, x in enumerate(arr) if i%k_fold != 1]
    test = [x for i, x in enumerate(arr) if i%k_fold == 1]
    return np.array(train), np.array(test)



allele_groups, df = load_binding_data('pan_allele/files/bdata.2009.mhci.public.1.txt')
allele_sequence_data, max_allele_length = load_allele_sequence_data('pan_allele/files/pseudo/pseudo_sequences.fasta')
allele_list = sorted(create_allele_list(allele_groups, allele_sequence_data))

peptides, mhc, Y = get_model_data(  allele_list,
                                    allele_sequence_data,
                                    allele_groups,
                                    dense_mhc_model=None,
                                    peptide_length = 9,
                                    mhc_length=max_allele_length,
                                    mhc_dense = None
                                 )

blind_allele_groups, blind_df = load_binding_data('blind_data.txt')
blind_allele_list = sorted(create_allele_list(blind_allele_groups, allele_sequence_data))


print blind_allele_list
nb_iter = 10
preds = np.zeros((len(blind_peptides),1))

for i in range(0,nb_iter):
    peptides_train, peptides_test = split_train_test(peptides,5)
    mhc_train, mhc_test = split_train_test(mhc,5)
    Y_train, Y_test = split_train_test(Y,5)
    graph = get_graph_from_hyperparameters('ffn_mult')
    graph.fit({'peptide':peptides_train, 'mhc':mhc_train, 'output': Y_train},
                batch_size=32,
                nb_epoch=19,
                verbose = 1,
                )

    preds += graph.predict({'peptide':blind_peptides, 'mhc':blind_mhc})['output']
    print preds
preds = preds.reshape(preds.shape[0])
preds = (20000**(1-preds))/nb_iter


print scores(blind_Y, preds )
