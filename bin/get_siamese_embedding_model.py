from __future__ import print_function
import keras
from feedforward_models import build_sequential_model
from pan_allele_data_helpers import load_similarity_data, load_allele_sequence_data
import numpy as np
import sys

###callbacks to visualize test and training_error ###

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']

    # def get_data(self,X_train, Y_train, X_test, Y_test):
    #     self.X_train = X_train
    #     self.Y_train = Y_train
    #     self.X_test = X_test
    #     self.Y_test = Y_test

    def on_epoch_begin(self, epoch, logs={}):
        if(epoch%25 ==0 ):
            print("Building siamese_embedding model: ", epoch*100/150, "%")


def create_embedding_file(mhc_sequence_fasta_file,embedding_size):
    allele_sequence_data, max_sequence_length = load_allele_sequence_data(mhc_sequence_fasta_file)
    X_encoded, Y = load_similarity_data(allele_sequence_data,"files/allele_similarity_2009.csv")
    print (X_encoded.shape, Y.shape)
    history=LossHistory()
    model = build_sequential_model(input_embedding_dim=20,last_layer_size= embedding_size, max_sequence_length=max_sequence_length)
    model.fit(X_encoded, Y, batch_size = 32, nb_epoch = 150, callbacks=[history], verbose=0)
    return model


### cross_validation ###

# def shuffle_data(X_encoded,Y):
#     shuffle_indices = np.random.permutation(len(X)/2)
#     #print(type(shuffle_indices))
#     X_shuffled = np.zeros((len(X),max_len))
#     X_shuffled[0::2] = X_encoded[shuffle_indices*2 - 2]
#     X_shuffled[1::2] = X_encoded[shuffle_indices*2 - 1]
#     Y_shuffled = Y[0::2]
#     Y_shuffled = Y_shuffled[shuffle_indices]
#     return X_shuffled, Y_shuffled

# def split_train_test(X_shuffled, Y_shuffled,test, train):
#
#     '''
#     Splits data into Training and Test sets
#     '''
#
#     X_test = np.zeros((len(test)*2,max_len))
#     X_train = np.zeros((len(train)*2,max_len))
#     Y_test= np.zeros((len(test)*2))
#     Y_train = np.zeros((len(train)*2))
#
#     X_test[0::2] = X_shuffled[np.multiply(2,test) ]
#     X_test[1::2] = X_shuffled[np.multiply(2,test) +1]
#
#     X_train[0::2] = X_shuffled[np.multiply(2,train)]
#     X_train[1::2] = X_shuffled[np.multiply(2,train) + 1]
#
#     Y_test[0::2] = Y_shuffled[test]
#     Y_test[1::2] = Y_shuffled[test]
#
#     Y_train[0::2] = Y_shuffled[train]
#     Y_train[1::2] = Y_shuffled[train]
#
#     return X_train, X_test, Y_train, Y_test
