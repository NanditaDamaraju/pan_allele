import os
import sys
from sys import argv

path = os.getcwd()
sys.path.append(path)

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters

import keras
import collections
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    default=32,
    help="minibatch size for training model (int)")

parser.add_argument(
    "--pred",
    default='ffn_mult',
    help="neural network type, `ffn_concat`, `ffn_mult` or `conv_mult`")

parser.add_argument(
    "--epochs",
    default=200,
    help="number of epochs to train upto")


class LossHistory(keras.callbacks.Callback):

    def metrics(self, batch_size):
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        model_save = self.model
        model_save.save_weights('weights/ffn_concat/weights' + str(self.batch_size)+ '_' + str(epoch),overwrite=True)

def save_model(graph, batch_size,nb_epoch):

    allele_groups, df = load_binding_data('pan_allele/files/bdata.2009.mhci.public.1.txt')
    allele_sequence_data, max_allele_length = load_allele_sequence_data('pan_allele/files/pseudo/pseudo_sequences.fasta')
    allele_list = sorted(create_allele_list(allele_groups, allele_sequence_data))

    peptide_train, mhc_train, Y_train = get_model_data(allele_list,
                                                        allele_sequence_data,
                                                        allele_groups,
                                                        dense_mhc_model=None,
                                                        peptide_length = 9,
                                                        mhc_length=max_allele_length,
                                                        mhc_dense = None
                                                        )


    history = LossHistory()
    history.metrics(batch_size)
    graph.fit(
                        {'peptide':peptide_train, 'mhc':mhc_train, 'output': Y_train},
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose = 1,
                        callbacks=[history]
        )

def main():
    args = parser.parse_args()
    print args.pred
    graph = get_graph_from_hyperparameters(args.pred)
    print graph
    batch_sizes = [32]
    learning_rates = [0.001]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            save_model(graph, args.batch_size, args.epochs)


if __name__ == "__main__":

    main()
