import os
import sys
path = os.path.split(os.getcwd())[0]
sys.path.append(path)

BINDING_DATA_PATH = path + 'pan_allele/files/bdata.2009.mhci.public.1.txt'
SEQUENCE_DATA_PATH = path +'pan_allele/files/pseudo/pseudo_sequences.fasta'
