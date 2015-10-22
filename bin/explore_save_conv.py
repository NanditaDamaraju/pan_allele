
from pan_allele_data_helpers import *
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(5000)
from convolution_model import convolution_graph, convolution_graph_matrix
from feedforward_models import build_graph_native_sequence_model
from sequence_encoding import padded_indices
from amino_acid import amino_acid_letter_indices
from generate_pseudo_sequences import create_fasta_file
from sys import argv
import keras
path="/home/ubuntu"

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.training_error = []
        self.test_error =[]

    def on_epoch_end(self, epoch, logs={}):
        model_save = self.model
        model_save.save_weights(path + "/py/pan_allele/weights_conv" + str(epoch), overwrite=True)
#path = "/Users/NanditaD/Intern/mhclearn"


def normalize_allele_name(allele_name):
    allele_name = allele_name.upper()
    # old school HLA-C serotypes look like "Cw"
    allele_name = allele_name.replace("CW", "C")
    patterns = [
        "HLA-",
        "-",
        "*",
        ":"
    ]
    for pattern in patterns:
        allele_name = allele_name.replace(pattern, "")
    return allele_name
#create_fasta_file(path, remove_residues = True, consensus_cutoff = 0)
allele_groups, df = load_binding_data(path +'/py/pan_allele/files/bdata.2009.mhci.public.1.txt')
allele_sequence_data, max_allele_length = load_allele_sequence_data(path +'/py/pan_allele/files/trimmed-human-class1.fasta')
allele_list = sorted(create_allele_list(allele_groups, allele_sequence_data))
peptide_train, mhc_train, Y_train = get_model_data(allele_list,
                                                    allele_sequence_data,
                                                    allele_groups,
                                                    dense_mhc_model=None,
                                                    peptide_length = 9,
                                                    mhc_length=max_allele_length,
                                                    mhc_dense = None
                                                    )

nb_epoch = 40
hyperparameters = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}
graph = build_graph_native_sequence_model(hyperparameters=hyperparameters, maxlen_mhc = max_sequence_length)

graph = convolution_graph_matrix(hyperparameters, maxlen_mhc=max_allele_length)
history = LossHistory()
graph.fit(
                    {'peptide':peptide_train, 'mhc':mhc_train, 'output': Y_train},
                    batch_size=32,
                    nb_epoch=nb_epoch,
                    verbose = 1,
                    callbacks = [history]
)
