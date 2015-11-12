import pandas as pd
import numpy as np

from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters

from paths import *

import keras
import collections
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="minibatch size for training model (int)")

parser.add_argument(
    "--pred",
    default='ffn_mult',
    help="neural network type, `ffn_concat`, `ffn_mult` or `conv_mult`")

parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="number of epochs to train upto")

parser.add_argument(
    "--max_ic50",
    default=20000,
    type=int,
    help="maximum ic50 value")


class LossHistory(keras.callbacks.Callback):

    def metrics(self, batch_size, pred):
        self.batch_size = batch_size
        self.pred = pred

    def on_epoch_end(self, epoch, logs={}):
        model_save = self.model
        model_save.save_weights(HOME_PATH + '/weights50000/' + self.pred +  '/weights' + str(self.batch_size)+ '_' + str(epoch),overwrite=True)

def save_model(graph, pred, batch_size,nb_epoch, max_ic50 = 20000):

    allele_groups, df = load_binding_data(BINDING_DATA_PATH, max_ic50 = max_ic50)
    allele_sequence_data, max_allele_length = load_allele_sequence_data(SEQUENCE_DATA_PATH)
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
    history.metrics(batch_size, pred)
    graph.fit(
                        {'peptide':peptide_train, 'mhc':mhc_train, 'output': Y_train},
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose = 1,
                        callbacks=[history]
        )

def main():
    args = parser.parse_args()
    graph = get_graph_from_hyperparameters(args.pred)
    batch_sizes = [32]
    learning_rates = [0.001]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            save_model(graph, args.pred, args.batch_size, args.epochs, max_ic50=args.max_ic50)


if __name__ == "__main__":

    main()
