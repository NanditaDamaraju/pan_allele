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
import numpy as np
import collections
import pandas as pd
import csv


max_ic50 = 20000
ic50_cutoff = 500
log_transformed_ic50_cutoff = 1 - np.log(ic50_cutoff)/np.log(max_ic50)

#
def format_peptide(peptide):
    if(len(peptide) == 9):
        return [peptide]
    elif(len(peptide) == 8):
        peptides = [peptide[:pos] +char +peptide[(pos):] for pos in range(4,9) for char in amino_acid_letters]
    elif(len(peptide) == 10):
        peptides = [peptide[:pos] + peptide[(pos+1):] for pos in range(4,9)]
    elif(len(peptide) == 11):
        peptides = [peptide[:pos] + peptide[(pos+2):] for pos in range(4,9)]
    return peptides


def scores(Y_true, Y_pred):
    Y_true_binary = Y_true <=ic50_cutoff
    Y_pred_log = 1 - np.log(Y_pred)/np.log(max_ic50)
    AUC = 0
    ACC = 0
    F1 = 0
    precision =0
    recall = 0
    length = 0
    Y_pred_binary = Y_pred <= ic50_cutoff
    ACC = accuracy_score(Y_true_binary, Y_pred_binary)
    F1 = f1_score(Y_true_binary, Y_pred_binary)
    recall = recall_score(Y_true_binary, Y_pred_binary)
    precision = precision_score(Y_true_binary, Y_pred_binary)
    length = len(Y_true)


    if(Y_true_binary.all() or not Y_true_binary.any()):
        print "Skipping as all labels are the same"
    else:
        AUC = roc_auc_score(Y_true_binary, Y_pred_log)

    return length, AUC, ACC, F1, precision, recall

def read_blind_predictions(filename):
    predictions = collections.defaultdict(dict)
    with open(filename, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
                for i,val in enumerate(header):
                    try:
                        predictions[row[0]][val] = float(row[i])
                    except:
                        pass
    return predictions

#file0 negative prediction and file 1 positive predicitons
def read_tcell_predictions(file0, file1):
    predictions = collections.defaultdict(dict)
    with open(file0, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
            peptide = row[2]
            allele = row[1]
            predictions[allele][peptide] = 0
    with open(file1, 'rb') as csvfile:
        records = csv.reader(csvfile)
        header = records.next()
        for row in records:
            peptide = row[2]
            allele = row[1]
            predictions[allele][peptide] = 1
    return predictions

'''takes a single peptide and allele sequence as input
    along with the model for  prediction
    and returns the predicted probability of binding'''

def make_prediction(peptide, allele_sequence, model=None):
    mhc_seq = padded_indices([allele_sequence],
                                    add_start_symbol=False,
                                    add_end_symbol=False,
                                    index_dict=amino_acid_letter_indices)


    #returns an array of index encoded peptide/peptides depending on peptide length

    X_p = padded_indices(format_peptide(peptide),
                            add_start_symbol=False,
                            add_end_symbol=False,
                            index_dict=amino_acid_letter_indices)

    #tiling the mhc in case the peptide is more than a length of 9

    mhc_seq = np.tile(mhc_seq,(len(X_p),1))
    preds = 0

    #mean of the predicted outputs in case peptide is more than length of 9

    if(model):
        preds = model.predict({'peptide':X_p,'mhc':mhc_seq})['output']
        preds = np.mean(preds)

    return float(preds)



def main():

    ##hyperparameters feed forward network concat
    #hyperparameters = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}

    ##hyperparameters feed forward network matrix multiply
    hyperparameters  = {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0. ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}

    ##hyperparameters convolutional network matrix multiply
    #hyperparameters = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}

    #prediction input either "conv", "ffn_concat", "ffn_mult"
    pred = sys.argv[1]

    if (pred == 'ffn_concat'):
        hyperparameters = {'cutoff':[ 0.33711265], 'dropouts': [ 0. ,  0.0254818 ,  0.10669398], 'sizes': [ 53,  82, 103,  74, 106, 59]}
        create_fasta_file(path, remove_residues = True, consensus_cutoff =hyperparameters['cutoff'][0])
        mhc_sequence_fasta_file = 'pan_allele/files/pseudo/pseudo_sequences.fasta'
        allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
        graph = build_graph_native_sequence_model(hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)

    elif(pred == 'ffn_mult'):
        hyperparameters  = {'cutoff':[ 0], 'dropouts': [ 0.17621593,  0. ,  0.   ], 'sizes': [ 16, 128,  99, 128, 102], 'mult_size': [32, 15]}
        create_fasta_file(path, remove_residues = True, consensus_cutoff =hyperparameters['cutoff'][0])
        mhc_sequence_fasta_file = 'pan_allele/files/pseudo/pseudo_sequences.fasta'
        allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
        graph = ffn_matrix( hyperparameters=hyperparameters, maxlen_mhc = max_allele_length)
    elif(pred =='conv'):
        hyperparameters = {'filter_length': [3, 4], 'nb_filter': [67, 92], 'mult_size': [32, 10], 'layer_size': [ 128, 92, 65]}
        create_fasta_file(path, remove_residues = False, consensus_cutoff =0)
        mhc_sequence_fasta_file = 'pan_allele/files/pseudo/pseudo_sequences.fasta'
        allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
        graph = convolution_graph_matrix(hyperparameters = hyperparameters, maxlen_mhc = max_allele_length )

    ##Load graph

    for num in range(15,16):

        predictors = ['mhcflurry', 'netmhcpan', 'netmhc', 'smmpmbec_cpp']




        allele_list = ['A0101',	    'A0201',	'A0202',    'A0203',	'A0206',	'A0301',
                       'A1101',	    'A2301',	'A2402',	'A2501',	'A2601',    'A2602',
                       'A2603',	    'A2902',	'A3001',	'A3002',	'A3101',	'A3201',
                       'A3301',	    'A6801',	'A6802',	'A6901',    'A8001',	'B0702',
                       'B0801',	    'B0802',	'B0803',	'B1501',	'B1503',	'B1509',
                       'B1517',	    'B1801',	'B2703',    'B2705',    'B3501',	'B3801',
                       'B3901',	    'B4001',	'B4002',	'B4402',	'B4403',	'B4501',
                       'B4601',	    'B5101',    'B5301',	'B5401',	'B5701',	'B5801'	]

        #allele_list = ['A0201']

        data_len = sum(len(read_blind_predictions('combined-test-data/'+ allele + '.csv').keys()) for allele in allele_list)

        Y_true_all = np.zeros(data_len)

        #Initialize metrics to 0
        total_metrics = collections.defaultdict(list)

        for val in predictors:
                total_metrics[val] =  np.zeros(data_len)

        lr = 0.001
        batch_size = 32
        graph.load_weights('weights/weights' + str(batch_size)+ '_' + str(lr) + '_'  + str(num) )

        counter  = 0
        calculated_metrics =collections.defaultdict(tuple)
        for val in predictors:
            calculated_metrics[val] = (0,0,0,0,0,0)
        for allele in allele_list:
            filename = 'combined-test-data/'+ allele + '.csv'
            predictions = read_blind_predictions(filename)
            peptides = predictions.keys()
            for peptide in peptides:
                predictions[peptide]['mhcflurry'] = 20000**(1-make_prediction(peptide, allele_sequence_data[allele], graph))
            df_pred = pd.DataFrame(predictions)


            Y_true_allele = np.array(df_pred.loc['meas'])
            Y_true_all[counter:counter+len(peptides)] =  Y_true_allele

            print "\n=====", allele, sum(Y_true_allele <= 500), len(Y_true_allele), "===="

            for val in predictors:
                Y_pred_allele = np.array(df_pred.loc[val])
                calculated_metrics[val]  = map(sum, zip(scores(Y_true_allele, Y_pred_allele), calculated_metrics[val]))
                print val, calculated_metrics[val]
                total_metrics[val][counter:counter+len(peptides)] = (Y_pred_allele)

            counter +=len(peptides)

        print "\n",num

        for val in predictors:
            print "\n",val
            scores_val = scores(Y_true_all, total_metrics[val])
            print scores_val

if __name__ == "__main__":
    main()
