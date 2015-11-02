import os
import sys
path = os.getcwd()
sys.path.append(path)

from pan_allele.helpers.feedforward_models import ffn_matrix, build_graph_native_sequence_model
from pan_allele.helpers.convolution_model import convolution_graph_matrix
from pan_allele.helpers.generate_pseudo_sequences import create_fasta_file
from pan_allele.helpers.pan_allele_data_helpers import load_allele_sequence_data
from pan_allele.helpers.sequence_encoding import padded_indices
from pan_allele.helpers.amino_acid import amino_acid_letter_indices, amino_acid_letters
from keras.models import Graph
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import collections
import pandas as pd
import csv

max_ic50 = 2000
ic50_cutoff = 500
log_transformed_ic50_cutoff = 1 - np.log(ic50_cutoff)/np.log(max_ic50)

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
