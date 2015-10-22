

from pan_allele_data_helpers import *
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(5000)
from convolution_model import convolution_graph, convolution_graph_matrix
from feedforward_models import build_graph_native_sequence_model
from sequence_encoding import padded_indices
from amino_acid import amino_acid_letter_indices
from sys import argv

#path = "/Users/NanditaD/Intern/mhclearn"
path="/home/ubuntu"
net = argv[1]
allele_groups, df = load_binding_data(path +'/py/pan_allele/files/2009_test_20.txt')

if(net == 'conv'):
    allele_sequence_data, max_allele_length = load_allele_sequence_data(path+'/py/pan_allele/files/trimmed-human-class1.fasta')
    hyperparameters = {'filter_length': [5, 5], 'nb_filter': [ 100, 100], 'mult_size':[32,10], 'layer_size': [ 16, 108, 52]}
    graph = convolution_graph_matrix(hyperparameters, maxlen_mhc = max_allele_length)

elif(net == 'ffn'):
    allele_sequence_data, max_allele_length = load_allele_sequence_data(path +'/py/pan_allele/files/pseudo/pseudo_sequences.fasta')
    graph = build_graph_native_sequence_model(maxlen_mhc = max_allele_length)

allele_list = create_allele_list(allele_groups, allele_sequence_data)


for i in range(1,80):
    graph.load_weights(path + '/py/pan_allele/weights_' +net+str(i))
    max_ic50 = 5000
    log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(max_ic50)
    AUC = {'A':0, 'B':0,'C':0}
    ACC = {'A':0, 'B':0,'C':0}
    num = {'A':0, 'B':0,'C':0}
    for allele in sorted(allele_list):
        peptide_test, mhc_test, Y_test = get_model_data([allele],
                                                        allele_sequence_data,
                                                        allele_groups,
                                                        dense_mhc_model=None,
                                                        peptide_length = 9,
                                                        mhc_length=max_allele_length,
                                                        mhc_dense = None
                                                        )

        #print "YAYA"
        output = graph.predict({'peptide':peptide_test,'mhc':mhc_test})['output']
        Y_true = 1 * np.greater(Y_test,log_transformed_ic50_cutoff)
        Y_pred = output.reshape(output.shape[0])
        try:
            AUC_score = roc_auc_score(Y_true, Y_pred)
            Y_pred = 1 * np.greater(Y_pred,log_transformed_ic50_cutoff)
            ACC_score = accuracy_score(Y_true, Y_pred)
            #print allele, AUC, len(peptide_test)
            AUC[allele[0]]+=AUC_score
            ACC[allele[0]]+=ACC_score
            num[allele[0]]+=1
        except:
            pass
            #print "THIS"
            #print allele, "NA", len(peptide_test)
    print i
    print "Average ALL:\tAUC:", (AUC['A']+AUC['B'])/(num['A']+num['B']), "\tACC\t",(ACC['A']+ACC['B'])/(num['A']+num['B'])
    print "Average A:\tAUC:", AUC['A']/num['A'],"\tACC\t",ACC['A']/num['A']
    print "Average B:\tAUC:", AUC['B']/num['B'],"\tACC\t",ACC['B']/num['B']
    #print "Average C:", AUC['C']/num['C'],"\tACC\t",ACC['C']/num['C']

# In[48]:



# In[ ]:




# In[ ]:
