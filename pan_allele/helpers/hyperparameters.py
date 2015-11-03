from generate_pseudo_sequences import create_fasta_file
from pan_allele_data_helpers import load_allele_sequence_data
from feedforward_models import ffn_matrix, build_graph_native_sequence_model
from convolution_model import convolution_graph_matrix

hyperparameters_ffn_concat = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}
hyperparameters_ffn_mult  = {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0. ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}
hyperparameters_conv_mult = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}
mhc_sequence_fasta_file = 'pan_allele/files/pseudo/pseudo_sequences.fasta'

def get_graph_from_hyperparameters(pred):
    if (pred =='ffn_concat'):

        create_fasta_file(remove_residues = True, consensus_cutoff =hyperparameters_ffn_concat['cutoff'][0])
        allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
        graph = build_graph_native_sequence_model( hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)

        return graph

    elif (pred =='ffn_mult'):

        create_fasta_file(remove_residues = True, consensus_cutoff =hyperparameters_ffn_mult['cutoff'][0])
        allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
        graph = ffn_matrix( hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)

        return graph

    elif (pred =='conv_mult'):

        create_fasta_file(remove_residues = False, consensus_cutoff =0)
        allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
        graph = convolution_graph_matrix( hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)

        return graph
