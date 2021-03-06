from paths import *
import sys
sys.path.append(HOME_PATH)

import pandas as pd
import numpy as np
from collections import defaultdict

from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters
from pan_allele.helpers.peptide_trim import make_prediction

from blind_dataset_metrics import scores, read_blind_predictions


import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pred",
    default='ffn_mult',
    help="neural network type, `ffn_concat`, `ffn_mult` or `conv_mult`")

parser.add_argument(
    "--max_ic50",
    default=20000,
    type=int,
    help="maximum ic50 value")


#Splitting data into training and test sets
def split_train_test(arr, k_fold):
    train = [x for i, x in enumerate(arr) if i%k_fold != 1]
    test = [x for i, x in enumerate(arr) if i%k_fold == 1]
    return np.array(train), np.array(test)


def main():
    args = parser.parse_args()
    max_ic50 = args.max_ic50

    #IEDB data
    allele_groups, df = load_binding_data(BINDING_DATA_PATH, max_ic50=max_ic50, peptide_length=9)

    #graph initialized here so that pseudo sequences are made accordingly
    graph = get_graph_from_hyperparameters(args.pred)

    #allele sequence data
    allele_sequence_data, max_allele_length = load_allele_sequence_data(SEQUENCE_DATA_PATH)
    allele_list = sorted(create_allele_list(allele_groups, allele_sequence_data))

    #reading blind data from txt file that contains aggregated data for all alleles
    blind_allele_groups, blind_df = load_binding_data('blind_data.txt', max_ic50=max_ic50, peptide_length=None)
    blind_allele_list = sorted(create_allele_list(blind_allele_groups, allele_sequence_data))

    nb_iter = 50 #number of networks to include in the ensemble

    preds_allele = defaultdict(list)
    actual_allele = defaultdict(list)

    for i in range(0,nb_iter):

        #get_model_data shuffles the data so theres no need for further shuffling
        peptides, mhc, Y = get_model_data(  allele_list,
                                            allele_sequence_data,
                                            allele_groups,
                                            peptide_length = 9,
                                            mhc_length=max_allele_length
                                         )

        #splitting peptides, mhcs and binding into training and test
        peptides_train, peptides_test = split_train_test(peptides,5)
        mhc_train, mhc_test = split_train_test(mhc,5)
        Y_train, Y_test = split_train_test(Y,5)

        #fit graph model
        graph = get_graph_from_hyperparameters(args.pred)
        graph.fit({'peptide':peptides_train, 'mhc':mhc_train, 'output': Y_train},
                    batch_size=32,
                    nb_epoch=12,
                    verbose = 0,
                    )

        #calculate metrics for each allele
        for allele in blind_allele_list:
            print i, allele

            predictions = read_blind_predictions(HOME_PATH + '/combined-test-data/'+ allele + '.csv')
            peptides = predictions.keys()

            preds = []
            meas = []

            for peptide in peptides:
                preds.append(make_prediction(peptide, allele_sequence_data[allele],graph))
                meas.append(predictions[peptide]['meas'])
            preds = np.array(preds)
            meas = np.array(meas)

            try:
                preds_allele[allele]+=preds/nb_iter
            except:
                preds_allele[allele]=preds/nb_iter

            actual_allele[allele] = meas


    #calculate average for all the alleles

    calculated_metrics = np.zeros(6)

    for allele in blind_allele_list:
        Y_pred_allele = max_ic50**(1-preds_allele[allele])
        Y_true_allele = actual_allele[allele]
        score_allele = scores(Y_true_allele, Y_pred_allele)
        calculated_metrics  += score_allele ##sum metrics for all alleles

    print calculated_metrics/len(blind_allele_list) #divide sum by number of alleles

if __name__ == "__main__":
    main()
