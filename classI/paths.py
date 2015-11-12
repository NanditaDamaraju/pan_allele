import os
import sys
HOME_PATH = os.path.split(os.getcwd())[0]
sys.path.append(HOME_PATH)

BINDING_DATA_PATH = HOME_PATH + '/pan_allele/files/bdata.2009.mhci.public.1.txt'
SEQUENCE_DATA_PATH = HOME_PATH +'/pan_allele/files/pseudo/pseudo_sequences.fasta'
