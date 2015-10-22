from __future__ import print_function
from pan_allele_data_helpers import *
from generate_pseudo_sequences import create_fasta_file
from feedforward_models import build_graph_native_sequence_model, ffn_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
from convolution_model import convolution_graph, convolution_graph_reshape,convolution_graph_matrix
from test_params import test_params
import keras
#path = "/Users/NanditaD/Intern/mhclearn"
path="/home/ubuntu"
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
        print(epoch, AUC_test, AUC_train)



def leave_one_out(
            path,
            hyperparameters,
            iedb_data_file,
            mhc_sequence_fasta_file=None,
            peptide_length = 9,
            max_sequence_length=None,
            nb_epoch=30, max_ic50 = 5000.0,):

    log_transformed_ic50_cutoff = 1 - np.log(500)/np.log(max_ic50)

    create_fasta_file(path, remove_residues = True, consensus_cutoff = 0)

    ##Load files
    allele_binding_data, df = load_binding_data(iedb_data_file)
    allele_sequence_data, max_allele_length = load_allele_sequence_data(mhc_sequence_fasta_file)
    max_sequence_length= max_allele_length

    for learn_rate in [ 0.001]:
        for peptide_activation in [ 'tanh']:
            for mhc_activation in [ 'tanh']:
                optimizer = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-6)

                print (peptide_activation, mhc_activation, learn_rate)

                graph = ffn_matrix(  maxlen_mhc = max_sequence_length,
                                                    optimizer=optimizer,
                                                    mhc_activation=mhc_activation,
                                                    peptide_activation= peptide_activation )
                initial_weights = graph.get_weights()

                allele_list =  create_allele_list(allele_binding_data, allele_sequence_data)
                formatted_allele_list = ['A0101', 'A0201']

                ##leave one out validaiton
                for allele in formatted_allele_list:
                    print(allele)
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

                    if(not np.all([Y_true[0]]*len(Y_true) == Y_true) ):  #check if all values in Y_true are not the same

                        graph.set_weights(initial_weights)
                        history = LossHistory()
                        history.get_data(peptides_train, mhc_train, Y_train, peptides_test, mhc_test, Y_true)
                        graph.fit(
                                {'peptide':peptides_train,'mhc':mhc_train, 'output': Y_train},
                                batch_size=32,
                                nb_epoch=nb_epoch,
                                verbose = 1,
                                callbacks=[history]
                            )



def main(job_id, params):
        mhc_sequence_fasta_file=path+"/py/pan_allele/files/trimmed-human-class1.fasta"
        iedb_data_file= path + "/py/pan_allele/files/bdata.2009.mhci.public.1.edited.txt"

        embedding_model=None
        max_sequence_length=None

        result = leave_one_out(
                    path=path,
                    hyperparameters=params,
                    mhc_sequence_fasta_file=mhc_sequence_fasta_file,
                    iedb_data_file=iedb_data_file,
                    peptide_length = 9,
                    max_sequence_length=None,
                    nb_epoch=64, max_ic50 = 5000.0)

        return result

if __name__ == "__main__":
    main(23,  test_params)
