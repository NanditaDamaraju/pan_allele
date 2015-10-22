import sys
import os
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
from metrics import format_peptide
import numpy as np
import collections
import pandas as pd
import csv
max_ic50 = 20000
ic50_cutoff = 500
log_transformed_ic50_cutoff = 1 - np.log(ic50_cutoff)/np.log(max_ic50)


def scores(Y_true_binary, Y_pred):
    Y_pred_log = 1 - np.log(Y_pred)/np.log(max_ic50)
    AUC = 0
    ACC = 0
    F1 = 0
    precision =0
    recall = 0
    length = 0
    if(Y_true_binary.all() or not Y_true_binary.any()):
        print "Skipping as all labels are the same"
    else:
        AUC = roc_auc_score(Y_true_binary, Y_pred_log)
        Y_pred_binary = Y_pred <= ic50_cutoff
        ACC = accuracy_score(Y_true_binary, Y_pred_binary)
        F1 = f1_score(Y_true_binary, Y_pred_binary)
        recall = recall_score(Y_true_binary, Y_pred_binary)
        precision = precision_score(Y_true_binary, Y_pred_binary)
        length = len(Y_true)
    return length, AUC, ACC, F1, precision, recall

def read_predictions(filename,value):
    predictions = collections.defaultdict(dict)
    with open(filename, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
            peptide = row[2]
            allele = row[3]
            predictions[allele][peptide]['true'] = value
    return predictions

def make_prediction(peptide, allele, model):
    mhc_seq = padded_indices([allele_sequence_data[allele]],
                                    add_start_symbol=False,
                                    add_end_symbol=False,
                                    index_dict=amino_acid_letter_indices)
    X_p = padded_indices(format_peptide(peptide),
                            add_start_symbol=False,
                            add_end_symbol=False,
                            index_dict=amino_acid_letter_indices)
    mhc_seq = np.tile(mhc_seq,(len(X_p),1))
    preds = model.predict({'peptide':X_p,'mhc':mhc_seq})['output']
    preds = np.mean(preds)
    return float(value)







#hyperparameters = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}
##hyperparameters feed forward network concat
hyperparameters  = {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0. ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}
##hyperparameters convolutional network matrix multiply
#hyperparameters = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}
remove_residues = False
pred = sys.argv[1]

cutoff = 0

if (pred[:3] == 'ffn'):
    remove_residues = True
    cutoff = hyperparameters['cutoff'][0]

create_fasta_file(path, remove_residues = remove_residues, consensus_cutoff =cutoff)
mhc_sequence_fasta_file = 'pan_allele/files/pseudo/pseudo_sequences.fasta'
allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)

if (pred == 'ffn_concat'):
    graph = build_graph_native_sequence_model(hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)
elif(pred == 'ffn_mult'):
    graph = ffn_matrix( hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)
elif(pred =='conv'):
    graph = convolution_graph_matrix(hyperparameters = hyperparameters, maxlen_mhc = max_allele_length )
initial_weights = graph.get_weights()



allele_list = predictions.keys()

##Load graph
graph.set_weights(initial_weights)
graph.load_weights('weights/weights_' + pred + '/weights14')
metrics = ['AUC', 'ACC', 'F1', 'precision', 'recall']
total_metrics = collections.defaultdict(dict)
for val in predictors:
    for metric in metrics:
        total_metrics[val][metric] = 0


print "HEY HEY"
total = 0
num = 14
for allele in allele_list:
    print allele
    filename = 'paper_data/iedb-tcell-2009-negative.csv'
    predictions = read_predictions(filename)
    peptides = predictions[allele].keys()
    for peptide in peptides:
        predictions[allele][peptide]['pred'] = make_prediction(peptide, allele, graph)

    df_pred = pd.DataFrame(predictions)
    print df
    #print "\n=====", allele, sum(Y_true <= 500), len(Y_true), "===="

    for val in predictors:
        Y_pred = np.array(df_pred.loc[val])
        calculated_metrics = scores(Y_true, Y_pred)
        #print val, calculated_metrics
        for idx, metric in enumerate(metrics):
            total_metrics[val][metric] += calculated_metrics[idx+1] * calculated_metrics[0]
    total+=calculated_metrics[0]
print "\n",num
for val in predictors:
    print "\n",val
    for metric in metrics:
        print metric, "=", total_metrics[val][metric]/total,
