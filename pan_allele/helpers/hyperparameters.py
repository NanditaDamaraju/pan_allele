from generate_pseudo_sequences import create_fasta_file


hyperparameters_ffn_concat = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}
hyperparameters_ffn_mult  = {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0. ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}
hyperparameters_conv_mult = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}

def get_hyperparameters(pred):
    if (pred =='ffn_concat'):
        create_fasta_file(remove_residues = True, consensus_cutoff =hyperparameters_ffn_concat['cutoff'][0])
        return hyperparameters_ffn_concat

    elif (pred =='ffn_mult'):
        create_fasta_file(remove_residues = True, consensus_cutoff =hyperparameters_ffn_mult['cutoff'][0])
        return hyperparameters_ffn_mult

    elif (pred =='conv_mult'):
        create_fasta_file(remove_residues = False, consensus_cutoff =0)
        return hyperparameters_conv_mult
