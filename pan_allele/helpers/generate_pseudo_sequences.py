import os
import sys
HOME_PATH = os.path.split(os.getcwd())[0]

import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter

def create_fasta_file(remove_residues = False, consensus_cutoff = 0):
    sequence_array =[]
    allele_list = []
    if(remove_residues):
        with open(HOME_PATH + '/files/trimmed-human-class1-IEDB.fasta','rU') as f:
            for record in SeqIO.parse(f, 'fasta'):
                name, sequence = record.description, str(record.seq)
                name = name.split(' ')[0]
                sequence_array.append(sequence)
                allele_list.append(name)



        sequence_mat = np.array([list(seq) for seq in sequence_array])
        delete_columns = []

        for columns in range(0,sequence_mat.shape[1]):
            char_occurence_dict =  Counter(sequence_mat[:,columns])
            if (char_occurence_dict.most_common(1)[0][1] >= len(allele_list)* (1 - consensus_cutoff )) :
                delete_columns.append(columns)

        all_sequences = []

        with open(HOME_PATH + '/files/trimmed-human-class1.fasta', 'rU') as f:
            for record in SeqIO.parse(f, 'fasta'):
                name, sequence = record.description, str(record.seq)
                name = name.split(' ')[0]
                if(len(sequence) == 181):
                    all_sequences.append(sequence)
                    allele_list.append(name)

        sequence_mat = np.array([list(seq) for seq in all_sequences])
        pseudo_sequences = np.delete(sequence_mat,delete_columns,axis=1)
        pseudo_sequences = [''.join(chars) for chars in pseudo_sequences ]

        with open(HOME_PATH+ '/files/pseudo/pseudo_sequences.fasta','w') as f:
             for index in range(0,len(pseudo_sequences)):
                 f.write("\n>"+allele_list[index]+"\n"+pseudo_sequences[index])

    else:

        with open(HOME_PATH+ '/files/trimmed-human-class1.fasta','rU') as f:
            with open(HOME_PATH + '/files/pseudo/pseudo_sequences.fasta', "w") as f1:
                for line in f:
                    f1.write(line)
