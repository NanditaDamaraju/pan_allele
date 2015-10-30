import os
import sys
path = os.getcwd()
sys.path.append(path)

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(5000)
from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.feedforward_models import ffn_matrix, build_graph_native_sequence_model
from pan_allele.helpers.convolution_model import convolution_graph_matrix
from pan_allele.helpers.sequence_encoding import padded_indices
from pan_allele.helpers.amino_acid import amino_acid_letter_indices
from pan_allele.helpers.generate_pseudo_sequences import create_fasta_file
from metrics import read_tcell_predictions, make_prediction
import keras
import collections
from sys import argv




class LossHistory(keras.callbacks.Callback):

    def metrics(self, batch_size, lr):
        self.batch_size = batch_size
        self.lr = lr

    def on_epoch_end(self, epoch, logs={}):
        model_save = self.model
        model_save.save_weights('weights/weights' + str(self.batch_size)+ '_' + str(self.lr) + '_'  + str(epoch),overwrite=True)



def save_ffn(hyperparameters, batch_size=32, lr=0.001):

    allele_groups, df = load_binding_data('pan_allele/files/bdataed')

    create_fasta_file(path, remove_residues = True, consensus_cutoff = hyperparameters['cutoff'][0])
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

    optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-6)
    #graph = build_graph_native_sequence_model(hyperparameters=hyperparameters, maxlen_mhc = max_allele_length,optimizer = optimizer)

    graph = ffn_matrix(hyperparameters= hyperparameters, maxlen_mhc=max_allele_length, optimizer = optimizer)
    history = LossHistory()
    history.metrics(batch_size, lr)
    graph.fit(
                        {'peptide':peptide_train, 'mhc':mhc_train, 'output': Y_train},
                        batch_size=batch_size,
                        nb_epoch=200,
                        verbose = 1,
                        callbacks=[history]
        )

def save_cnn(hyperparameters, batch_size=32, lr=0.001):

    allele_groups, df = load_binding_data('pan_allele/files/bdata.2009.mhci.public.1.txt')

    create_fasta_file(path, remove_residues = False, consensus_cutoff =0)
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


    optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-6)
    graph = convolution_graph_matrix(hyperparameters, maxlen_mhc=max_allele_length, optimizer=optimizer)
    history = LossHistory()
    history.metrics(batch_size, lr)
    graph.fit(
                        {'peptide':peptide_train, 'mhc':mhc_train, 'output': Y_train},
                        batch_size=batch_size,
                        nb_epoch=64,
                        verbose = 1,
                        callbacks = [history]
    )


def main():
    hyperparameters =  {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0.        ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}
    #hyperparameters = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}
    #hyperparameters = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}

    batch_sizes = [32]
    learning_rates = [ 0.001]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            save_ffn(hyperparameters, batch_size, lr)


if __name__ == "__main__":

    main()
