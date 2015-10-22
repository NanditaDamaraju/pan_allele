from __future__ import print_function
from pan_allele_data_helpers import *
from generate_pseudo_sequences import create_fasta_file
from feedforward_models import build_graph_native_sequence_model, build_graph_siamese_embedding_model, build_sequential_model
from sklearn.metrics import roc_auc_score, accuracy_score
from convolution_model import convolution_graph, convolution_graph_reshape,convolution_graph_matrix
from test_params import test_params
import keras
#path = "/Users/NanditaD/Intern/mhclearn"
path="/home/ubuntu"

def split_train_test(arr, k, val):
    train = [x for i, x in enumerate(arr) if i%k != val]
    test = [x for i, x in enumerate(arr) if i%k == val]
    return train, test

def five_fold_validation(
            path,
            hyperparameters,
            iedb_data_file,
            mhc_sequence_fasta_file=None,
            peptide_length = 9,
            max_sequence_length=None,
            nb_epoch=30, max_ic50 = 5000.0,):

    AUC = 0
    log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(max_ic50)

    create_fasta_file(path, remove_residues = True, consensus_cutoff = 0)

    ##Load files
    allele_binding_data, df = load_binding_data(iedb_data_file)
    allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
    max_sequence_length= max_allele_length

    learn_rate = 0.01
    peptide_activation = 'tanh'
    mhc_activation = 'tanh'

    optimizer = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-6)

    print (peptide_activation, mhc_activation, learn_rate)

    graph = build_graph_native_sequence_model(  maxlen_mhc = max_sequence_length,
                                        optimizer=optimizer,
                                        mhc_activation=mhc_activation,
                                        peptide_activation= peptide_activation )
    initial_weights = graph.get_weights()

    allele_list =  create_allele_list(allele_binding_data, allele_sequence_data)

    peptides, mhc, Y = get_model_data(  allele_list,
                                        allele_sequence_data,
                                        allele_binding_data,
                                        dense_mhc_model=None,
                                        peptide_length = peptide_length,
                                        mhc_length=max_sequence_length,
                                        mhc_dense = None
                                        )

    k =5


    for val in range(0,k):

        peptides_train, peptides_test = split_train_test(peptides, k, val)
        mhc_train, mhc_test = split_train_test(mhc, k, val)
        Y_train, Y_test = split_train_test(Y, k , val)

        graph.fit({'peptides':peptides_train, 'mhc':mhc_train, 'output':Y_train},
        batch_size=32,
        nb_epoch=10,
        verbose = 0)

        Y_pred = graph.predict({'peptides':peptides_test, 'mhc':mhc_test})['output']

        Y_true = 1 * np.greater(Y_test,log_transformed_ic50_cutoff)
        Y_pred = Y_pred.reshape(Y_pred.shape[0])

        AUC += roc_auc_score(Y_true, Y_pred)

    return AUC
