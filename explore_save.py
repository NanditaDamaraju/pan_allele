
from pan_allele_data_helpers import *
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(5000)
from feedforward_models import ffn_matrix, build_graph_native_sequence_model
from sequence_encoding import padded_indices
from amino_acid import amino_acid_letter_indices
from generate_pseudo_sequences import create_fasta_file
import keras
from sys import argv

import os
path = os.getcwd()

class LossHistory(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        model_save = self.model
        model_save.save_weights('pan_allele/weights/weights_ffn/weights' + str(epoch),overwrite=True)

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
create_fasta_file(path, remove_residues = True, consensus_cutoff = 0)
allele_groups, df = load_binding_data('pan_allele/files/bdata.2009.mhci.public.1.txt')
allele_sequence_data, max_allele_length = load_allele_sequence_data(path +'pan_allele/files/pseudo/pseudo_sequences.fasta')
allele_list = sorted(create_allele_list(allele_groups, allele_sequence_data))
peptide_train, mhc_train, Y_train = get_model_data(allele_list,
                                                    allele_sequence_data,
                                                    allele_groups,
                                                    dense_mhc_model=None,
                                                    peptide_length = 9,
                                                    mhc_length=max_allele_length,
                                                    mhc_dense = None
                                                    )
#hyperparameters =  {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0.        ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}
hyperparameters = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}
graph = build_graph_native_sequence_model(hyperparameters=hyperparameters, maxlen_mhc=max_allele_length)
#graph = ffn_matrix(hyperparameters= hyperparameters, maxlen_mhc=max_allele_length)
history = LossHistory()
graph.fit(
                    {'peptide':peptide_train, 'mhc':mhc_train, 'output': Y_train},
                    batch_size=32,
                    nb_epoch=36,
                    verbose = 1,
                    callbacks=[history]

    )


# In[48]

# In[ ]:




# In[ ]:
