from __future__ import print_function
import numpy as np
import keras
import os
import time
from pan_allele_data_helpers import *
from convolution_model import convolution_graph_matrix, convolution_graph_reshape
from generate_pseudo_sequences import create_fasta_file
from feedforward_models import build_graph_native_sequence_model, build_graph_siamese_embedding_model, build_sequential_model
from sklearn.metrics import roc_auc_score, accuracy_score
class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.training_error = []
        self.test_error =[]

    def get_data(self,peptides_train, mhc_train, Y_train, peptides_test, mhc_test, Y_test):
        self.peptides_train = peptides_train
        self.mhc_train = mhc_train
        self.Y_train = Y_train
        self.peptides_test = peptides_test
        self.mhc_test = mhc_test
        self.Y_test = Y_test

    def on_epoch_end(self, epoch, logs={}):
        #self.training_error.append(get_similarity_error(self.Y_train,self.model.predict(self.X_train)))
        Y_pred_train = self.model.predict({'peptide':self.peptides_train,'mhc':self.mhc_train})['output']
        Y_pred_test = self.model.predict({'peptide':self.peptides_test,'mhc':self.mhc_test})['output']
        Y_pred_test = Y_pred_test.reshape(Y_pred_test.shape[0])
        Y_pred_train = Y_pred_train.reshape(Y_pred_train.shape[0])
        AUC_test = roc_auc_score(self.Y_test, Y_pred_test)
        AUC_train = roc_auc_score(self.Y_train, Y_pred_train)
        self.training_error.append(AUC_train)
        self.test_error.append(AUC_test)
        print(epoch, AUC_test, AUC_train)

def leave_one_out(
            newpath,
            path,
            hyperparameters,
            iedb_data_file,
            mhc_sequence_fasta_file=None,
            peptide_length = 9,
            max_sequence_length=None,
            nb_epoch=10, max_ic50 = 5000.0,):




    log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(max_ic50)

    create_fasta_file(path, remove_residues = True, consensus_cutoff = 0)

    ##Load files
    allele_binding_data, df = load_binding_data(iedb_data_file)
    allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
    max_sequence_length= max_allele_length

    #optimizer = keras.optimizers.Adagrad(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
    ##build models


    for learn_rate in [ 0.001]:
        for peptide_activation in [ 'tanh']:
            for mhc_activation in [ 'tanh']:
                total_AUC_score = []
                print (peptide_activation, mhc_activation, learn_rate)
                optimizer = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-6)
                graph = build_graph_native_sequence_model(
                                                    maxlen_mhc = max_sequence_length,
                                                    optimizer=optimizer,
                                                    mhc_activation=mhc_activation,
                                                    peptide_activation= peptide_activation )
                initial_weights = graph.get_weights()

                allele_list =  create_allele_list(allele_binding_data, allele_sequence_data)
                formatted_allele_list = ['A0101','A0201','A2902','A3101','A6801','B0801','B1501','B1801','B2705','B3901']
                #formatted_allele_list = ['B2705']
                # formatted_allele_list = ['A0101','A0201','A0202','A0203',
                #                         'A0205','A0206','A0207','A0211',
                #                         'A0212','A0216','A0219','A0301',
                #                         'A1101','A2301','A2402','A2403',
                #                         'A2501','A2601','A2602','A2603',
                #                         'A2902','A3001','A3002','A3101',
                #                         'A3201','A3301','A6801','A6802',
                #                         'A6901','A8001','B0702','B0801',
                #                         'B0802','B0803','B1501','B1502',
                #                         'B1503','B1509','B1517','B1801',
                #                         'B2705','B3501','B3801','B3901',
                #                         'B4001','B4002','B4402','B4403',
                #                         'B4501','B4601','B4801','B5101',
                #                         'B5301','B5401','B5701','B5801',
                #                         'B5802','B7301']
                formatted_allele_list = allele_list
                ##leave one out validaiton
                for allele in allele_list:
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    f = open(newpath+ '/' + allele,'wb')
                    #split into training_list and test_list, where the test_list consists of a single allele
                    training_list = create_allele_list(allele_binding_data, allele_sequence_data)
                    training_list.remove(allele)
                    peptides_train, mhc_train, Y_train = get_model_data(training_list,
                                                                        allele_sequence_data,
                                                                        allele_binding_data,
                                                                        dense_mhc_model=None,
                                                                        peptide_length = peptide_length,
                                                                        mhc_length=max_sequence_length,
                                                                        mhc_dense = None
                                                                        )
                    peptides_test, mhc_test, Y_test = get_model_data([allele],
                                                                    allele_sequence_data,
                                                                    allele_binding_data,
                                                                    dense_mhc_model=None,
                                                                    peptide_length = peptide_length,
                                                                    mhc_length=max_sequence_length,
                                                                    mhc_dense=None)

                    #convert ic50 values into binary binders(1) or non-binders(0)
                    Y_true = 1 * np.greater(Y_test,log_transformed_ic50_cutoff)
                    Y_train = 1 * np.greater(Y_train,log_transformed_ic50_cutoff)

                    history = LossHistory()
                    history.get_data(peptides_train, mhc_train, Y_train, peptides_test, mhc_test, Y_true)

                    if(not np.all([Y_true[0]]*len(Y_true) == Y_true) ):  #check if all values in Y_true are not the same

                        graph.set_weights(initial_weights)

                        graph.fit(
                                {'peptide':peptides_train,'mhc':mhc_train, 'output': Y_train},
                                batch_size=32,
                                nb_epoch=nb_epoch,
                                verbose = 1,
                                callbacks = [history]
                            )

                        Y_true = 1 * np.greater(Y_test,log_transformed_ic50_cutoff)



                        Y_pred = graph.predict({'peptide':peptides_test,'mhc':mhc_test})['output']
                        Y_pred = Y_pred.reshape(Y_pred.shape[0])

                        AUC = roc_auc_score(Y_true, Y_pred)
                        #convert Y_pred to binary
                        Y_pred_binary = 1 * np.greater(Y_pred,log_transformed_ic50_cutoff)
                        ACC = accuracy_score(Y_true, Y_pred_binary)

                        print ("Allele: ",allele, "\t#entries :", len(peptides_test) ,"\tAUC: ", AUC, "\tACC:", ACC)
                        total_AUC_score.append(AUC)
                        print(history.test_error,'\n',history.training_error, file=f)
                        print ("\nAllele: ",allele, "\t#entries :", len(peptides_test) ,"\tAUC: ", AUC, "\tACC:", ACC, file=f)
                        f.close()

        print (1 - sum(total_AUC_score)/len(total_AUC_score))
