import numpy as np
import pandas as pd
import csv
from Bio import SeqIO
from sequence_encoding import padded_indices
from collections import namedtuple
from amino_acid import amino_acid_letter_indices

def load_allele_sequence_data(file_fasta):

    '''
    Loads a fasta file into a dictionary with
    fasta identifiers as keys mapping to the
    sequence
    '''

    fasta_sequences = SeqIO.parse(open(file_fasta), 'fasta')
    allele_sequence_data = {}

    max_sequence_length = 0

    for sequence in fasta_sequences:
        name, sequence = sequence.id, str(sequence.seq)
        allele_sequence_data[name] = sequence
        if(len(sequence) > max_sequence_length):
            max_sequence_length = len(sequence)

    return allele_sequence_data, max_sequence_length

def load_binding_data(
        filename,
        peptide_length=9,
        max_ic50=5000.0,
    ):
    """
    Loads an IEDB dataset and returns a dictionary with alleles as keys
    mapping to the log-transformed ic50 values (Y), original ic50 values(ic50)
    and the peptide sequence
    """
    AlleleData = namedtuple("AlleleData", "Y peptides ic50")

    df = pd.read_csv(filename, sep="\t")
    human_mask = df["species"] == "human"
    length_mask = df["peptide_length"] == peptide_length
    df = df[human_mask & length_mask]
    allele_groups = {}

    for allele, group in df.groupby("mhc"):
        bad_hla_name_elements = [
            "HLA-",
            "-",
            "*",
            ":"
        ]
        for substring in bad_hla_name_elements:
            allele = allele.replace(substring, "")
        ic50 = np.array(group["meas"])
        Y = np.maximum(0.0, 1.0 - np.log(ic50) / np.log(max_ic50))
        peptides = list(group["sequence"])

        allele_groups[allele] = AlleleData(
            Y=Y,
            ic50=ic50,
            peptides=peptides)
    return allele_groups, df

def load_similarity_data(allele_sequence_data, file_similarity):

    X = []
    Y = []

    with  open(file_similarity,'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:

            allele_x = row['allele1']
            allele_y = row['allele2']
            similarity = row['similarity']
            weight = row['overlap']
            try:
                allele_sequence_data[allele_x]
                allele_sequence_data[allele_y]
                if( (allele_x!=allele_y) and (similarity!='nan') and (float(weight) >= 25)):
                    X.append(allele_sequence_data[allele_x])
                    X.append(allele_sequence_data[allele_y])
                    Y.append(float(similarity))
                    Y.append(float(similarity))
            except:
                pass

    Y = np.array(Y)
    X_encoded = padded_indices(X, index_dict=amino_acid_letter_indices,
                            add_end_symbol=False,add_start_symbol=False)
    return X_encoded, Y


def create_allele_list(allele_binding_data, allele_sequence_data):
    '''
    List alleles present in both allele binding and sequence data.
    '''
    allele_list = []
    for allele in allele_binding_data.keys():
        try:
            allele_sequence_data[allele]
            allele_list.append(allele)
        except:
            pass

    return sorted(allele_list)


def get_model_data(allele_list,
                allele_sequence_data,
                allele_binding_data,
                dense_mhc_model,
                peptide_length =9,
                mhc_length=None,
                mhc_dense = True
                ):

    '''
    generate training data for a list of alleles where
    the output is index_encoded peptide, dense MHC sequence
    and the log-transformed ic50 values
    '''

    data_len =0

    for allele in allele_list:
        data_len  += len(allele_binding_data[allele][1])

    X_p = np.zeros((data_len,peptide_length))
    X_mhc = np.zeros((data_len,mhc_length))
    Y_combined = np.zeros(data_len)

    index=0
    for allele in allele_list:
        peptides = allele_binding_data[allele][1]
        end_index = index+len(peptides)


        #index encoding for peptides
        X_p[index:end_index] = padded_indices(peptides,
                                            add_start_symbol=False,
                                            add_end_symbol=False,
                                            index_dict=amino_acid_letter_indices)


        #dense vector for mhc
        mhc_seq = padded_indices([allele_sequence_data[allele]],
                                add_start_symbol=False,
                                add_end_symbol=False,
                                index_dict=amino_acid_letter_indices)

        if(mhc_dense):
            mhc_seq = dense_mhc_model.predict([mhc_seq])
        X_mhc[index:end_index] =  np.tile(mhc_seq,(len(peptides),1))


        #log-transformed binding values
        Y_combined[index:end_index] = allele_binding_data[allele][0]
        index+=len(peptides)

    return X_p, X_mhc, Y_combined
