import os
import sys
from sys import argv

path = os.getcwd()
sys.path.append(path)

import pandas as pd
import numpy as np

from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters

graph = get_graph_from_hyperparameters('ffn_mult')

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
