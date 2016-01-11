import numpy as np
from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Merge
from keras.optimizers import RMSprop
import theano.tensor as T


#keras 0.3.0
def ffn(
        hyperparameters  = None,
        optimizer='rmsprop',
        peptide_length = 9,
        maxlen_mhc = 181,
        mhc_activation = 'tanh',
        peptide_activation = 'tanh'
        ):

    size_names = [
            'peptide_embedding_size',
            'mhc_embedding_size',
            'peptide_hidden_size',
            'mhc_hidden_size',
            'combined_hidden_size',
            'combined_hidden_final'
            ]

    dropout_names = [
            'dropout_merged',
            'dropout_peptide',
            'dropout_mhc'
            ]

    size_dict = {}
    dropout_dict = {}

    for idx, name in enumerate(size_names):
        # print idx,name
        size_dict[name] = hyperparameters['sizes'][idx]

    for idx, name in enumerate(dropout_names):
        dropout_dict[name] = hyperparameters['dropouts'][idx]

    # print size_dict, dropout_dict

    # print("Building Graph Model")
    graph = Graph()

    graph.add_input(name='peptide',input_shape=(peptide_length,), dtype=int)

    graph.add_input(name='mhc', input_shape=(maxlen_mhc,), dtype=int)


    ##PEPTIDE

    graph.add_node(
                Embedding(  input_dim=20,
                            output_dim=size_dict['peptide_embedding_size'],
                            input_length=peptide_length ),

                name='peptide_embedding',
                input='peptide')

    graph.add_node(
                Flatten(),
                name = 'peptide_flatten',
                input = 'peptide_embedding')

    graph.add_node(
                Dense(  input_dim = size_dict['peptide_embedding_size'] * peptide_length,
                        output_dim = size_dict['peptide_hidden_size'],
                        activation = peptide_activation ),

                name='peptide_dense',
                input='peptide_flatten'   )

    dropout_output_peptide = 'peptide_dense'
    if(dropout_dict['dropout_peptide']):
        dropout_output_peptide = 'peptide_dropout'
        graph.add_node(Dropout(dropout_dict['dropout_peptide']), name = 'peptide_dropout', input='peptide_dense')


    ##MHC

    graph.add_node(
                Embedding(  input_dim=20,
                            output_dim=size_dict['mhc_embedding_size'],
                            input_length=maxlen_mhc ),
                name='mhc_embedding',
                input='mhc')

    graph.add_node(
                Flatten(),
                name = 'mhc_flatten',
                input = 'mhc_embedding')

    graph.add_node(
                Dense(  input_dim = size_dict['mhc_embedding_size'] * maxlen_mhc,
                        output_dim = size_dict['mhc_hidden_size'],
                        activation = mhc_activation),
                name='mhc_dense',
                input='mhc_flatten')

    dropout_output_mhc = 'mhc_dense'
    if(dropout_dict['dropout_mhc']):
        dropout_output_mhc = 'mhc_dropout'
        graph.add_node(Dropout(dropout_dict['dropout_mhc']), name = dropout_output_mhc, input='mhc_dense')


    ##MERGE

    graph.add_node(
            Dense(  input_dim = size_dict['mhc_hidden_size'] + size_dict['peptide_hidden_size'],
                    output_dim = size_dict['combined_hidden_size'],
                    activation = 'relu'),
            name='dense_merged_1',
            inputs=[dropout_output_peptide,dropout_output_mhc],
            merge_mode='concat')


    graph.add_node(
            Dropout(dropout_dict['dropout_merged']),
            name = 'dense_dropout_1',
            input='dense_merged_1')


    graph.add_node(
            Dense(  input_dim = size_dict['combined_hidden_size'],
                    output_dim = size_dict['combined_hidden_final'],
                    activation = 'relu'),
            name = 'dense_merged_2',
            input = 'dense_dropout_1')


    graph.add_node(
            Dropout(dropout_dict['dropout_merged']),
            name = 'dense_dropout_2',
            input='dense_merged_2')


    graph.add_node(
            Dense(  input_dim = size_dict['combined_hidden_final'],
                    output_dim = 1,
                    activation = 'sigmoid'  ),

            name = 'dense_output',
            input = 'dense_dropout_2')

    graph.add_output(
            name='output',
            input='dense_output')

    graph.compile(optimizer,{'output':'mse'})
    return graph

def ffn_matrix(
        hyperparameters  = None,
        optimizer='rmsprop',
        peptide_length = 9,
        maxlen_mhc = 181,
        mhc_activation = 'tanh',
        peptide_activation = 'tanh'
        ):

    #hyperparameters = {'sizes':[128,64,128,128,128,64],'dropouts':[0.2,0.2,0.2], 'mult_size':[16,16]}
    size_names = [
            'peptide_embedding_size',
            'mhc_embedding_size',
            'mhc_hidden_size',
            'combined_hidden_size',
            'combined_hidden_final'
            ]

    dropout_names = [
            'dropout_merged',
            'dropout_peptide',
            'dropout_mhc'
            ]
    mult_size = {
            'mhc_m':hyperparameters['mult_size'][0],
            'mhc_n':hyperparameters['mult_size'][1]
    }

    size_dict = {}
    dropout_dict = {}

    for idx, name in enumerate(size_names):
        size_dict[name] = hyperparameters['sizes'][idx]

    for idx, name in enumerate(dropout_names):
        dropout_dict[name] = hyperparameters['dropouts'][idx]


    # print size_dict, dropout_dict
    #
    # print("Building Graph Model")
    graph = Graph()

    graph.add_input(name='peptide',input_shape=(peptide_length,), dtype=int)
    graph.add_input(name='mhc', input_shape=(maxlen_mhc,), dtype=int)

    ##PEPTIDE

    graph.add_node(
                Embedding(  input_dim=20,
                            output_dim=size_dict['peptide_embedding_size'],
                            input_length=peptide_length ),

                name='peptide_embedding',
                input='peptide')

    graph.add_node(
                Flatten(),
                name = 'peptide_flatten',
                input = 'peptide_embedding')

    graph.add_node(
                Dense(  input_dim = size_dict['peptide_embedding_size'] * peptide_length,
                        output_dim = mult_size['mhc_m'],
                        activation = peptide_activation ),

                name='peptide_dense',
                input='peptide_flatten'   )


    dropout_output_peptide = 'peptide_dense'
    if(dropout_dict['dropout_peptide']):
        dropout_output_peptide = 'peptide_dropout'
        graph.add_node(Dropout(dropout_dict['dropout_peptide']), name = 'peptide_dropout', input='peptide_dense')



    ##MHC

    graph.add_node(
                Embedding(  input_dim = 20,
                            output_dim = size_dict['mhc_embedding_size'],
                            input_length = maxlen_mhc),
                name='mhc_embedding',
                input='mhc')

    graph.add_node(
                Flatten(),
                name = 'mhc_flatten',
                input = 'mhc_embedding')

    graph.add_node(
                Dense(  input_dim = size_dict['mhc_embedding_size'] * maxlen_mhc,
                        output_dim = size_dict['mhc_hidden_size'],
                        activation = mhc_activation),
                name='mhc_dense',
                input='mhc_flatten')


    dropout_output_mhc = 'mhc_dense'
    if(dropout_dict['dropout_mhc']):
        dropout_output_mhc = 'mhc_dropout'
        graph.add_node(Dropout(dropout_dict['dropout_mhc']), name = dropout_output_mhc, input='mhc_dense')

    graph.add_node(
                Dense(  input_dim = size_dict['mhc_hidden_size'],
                        output_dim = mult_size['mhc_m'] * mult_size['mhc_n'],
                        activation = mhc_activation),
                    name = 'mhc_dense_2',
                    input = dropout_output_mhc)

    graph.add_node(
                Reshape(dims =(mult_size['mhc_n'],mult_size['mhc_m'])),
                name = 'mhc_final',
                input = 'mhc_dense_2')
    ##MERGE

    graph.add_node(
            Dense(  input_dim = mult_size['mhc_n'],
                    output_dim = size_dict['combined_hidden_size'],
                    activation = "relu"),
            name='dense_merged_1',
            inputs=[dropout_output_peptide,'mhc_final'],
            merge_mode='dot')


    graph.add_node(
            Dropout(dropout_dict['dropout_merged']),
            name = 'dense_dropout_1',
            input='dense_merged_1')

    graph.add_node(
            Dense(  input_dim = size_dict['combined_hidden_size'],
                    output_dim = size_dict['combined_hidden_final'],
                    activation = "relu"),
            name = 'dense_merged_2',
            input = 'dense_dropout_1')


    graph.add_node(
            Dropout(dropout_dict['dropout_merged']),
            name = 'dense_dropout_2',
            input='dense_merged_2')


    graph.add_node(
            Dense(  input_dim = size_dict['combined_hidden_final'],
                    output_dim = 1,
                    activation = "sigmoid"),
            name = 'dense_output',
            input = 'dense_dropout_2')

    graph.add_output(
            name='output',
            input='dense_output')

    graph.compile(optimizer,{'output':'mse'})
    return graph

# #keras 0.1.2
# def build_graph_native_sequence_model(
#         hyperparameters  = None,
#         optimizer='rmsprop',
#         peptide_length = 9,
#         maxlen_mhc = 181,
#         mhc_activation = 'tanh',
#         peptide_activation = 'tanh'
#         ):
#
#     #hyperparameters = {'sizes':[128,64,128,128,128,64],'dropouts':[0.2,0.2,0.2]}
#     size_names = [
#             'peptide_embedding_size',
#             'mhc_embedding_size',
#             'peptide_hidden_size',
#             'mhc_hidden_size',
#             'combined_hidden_size',
#             'combined_hidden_final'
#             ]
#
#     dropout_names = [
#             'dropout_merged',
#             'dropout_peptide',
#             'dropout_mhc'
#             ]
#
#     size_dict = {}
#     dropout_dict = {}
#
#     for idx, name in enumerate(size_names):
#         # print idx,name
#         size_dict[name] = hyperparameters['sizes'][idx]
#
#     for idx, name in enumerate(dropout_names):
#         dropout_dict[name] = hyperparameters['dropouts'][idx]
#
#     # print size_dict, dropout_dict
#
#     # print("Building Graph Model")
#     graph = Graph()
#
#     graph.add_input(name='peptide', ndim=2)
#
#     graph.add_input(name='mhc', ndim=2)
#
#     graph.inputs['peptide'].input = T.imatrix()
#     graph.inputs['mhc'].input = T.imatrix()
#
#     ##PEPTIDE
#
#     graph.add_node(
#                 Embedding(20,size_dict['peptide_embedding_size']),
#                 name='peptide_embedding',
#                 input='peptide')
#
#     graph.add_node(
#                 Flatten(),
#                 name = 'peptide_flatten',
#                 input = 'peptide_embedding')
#
#     graph.add_node(
#                 Dense(size_dict['peptide_embedding_size'] * peptide_length, size_dict['peptide_hidden_size'],
#                     activation = peptide_activation ),
#                 name='peptide_dense',
#                 input='peptide_flatten'   )
#
#
#     dropout_output_peptide = 'peptide_dense'
#     if(dropout_dict['dropout_peptide']):
#         dropout_output_peptide = 'peptide_dropout'
#         graph.add_node(Dropout(dropout_dict['dropout_peptide']), name = 'peptide_dropout', input='peptide_dense')
#
#
#
#     ##MHC
#
#     graph.add_node(
#                 Embedding(20,size_dict['mhc_embedding_size']),
#                 name='mhc_embedding',
#                 input='mhc')
#
#     graph.add_node(
#             Flatten(),
#             name = 'mhc_flatten',
#             input = 'mhc_embedding')
#
#     graph.add_node(
#             Dense(size_dict['mhc_embedding_size'] * maxlen_mhc, size_dict['mhc_hidden_size'],
#                 activation = mhc_activation),
#             name='mhc_dense',
#             input='mhc_flatten')
#
#     dropout_output_mhc = 'mhc_dense'
#     if(dropout_dict['dropout_mhc']):
#         dropout_output_mhc = 'mhc_dropout'
#         graph.add_node(Dropout(dropout_dict['dropout_mhc']), name = dropout_output_mhc, input='mhc_dense')
#
#
#     ##MERGE
#
#     graph.add_node(
#             Dense(size_dict['mhc_hidden_size'] + size_dict['peptide_hidden_size'], size_dict['combined_hidden_size'],
#                 activation = "relu"),
#             name='dense_merged_1',
#             inputs=[dropout_output_peptide,dropout_output_mhc],
#             merge_mode='concat')
#
#
#     graph.add_node(
#             Dropout(dropout_dict['dropout_merged']),
#             name = 'dense_dropout_1',
#             input='dense_merged_1')
#
#
#     graph.add_node(
#             Dense(size_dict['combined_hidden_size'],size_dict['combined_hidden_final'],activation = "relu"),
#             name = 'dense_merged_2',
#             input = 'dense_dropout_1')
#
#
#     graph.add_node(
#             Dropout(dropout_dict['dropout_merged']),
#             name = 'dense_dropout_2',
#             input='dense_merged_2')
#
#
#     graph.add_node(
#             Dense(size_dict['combined_hidden_final'],1,activation = "sigmoid"),
#             name = 'dense_output',
#             input = 'dense_dropout_2')
#
#     graph.add_output(
#             name='output',
#             input='dense_output')
#
#     graph.compile(optimizer,{'output':'mse'})
#     return graph

# def ffn_matrix(
#         hyperparameters  = None,
#         optimizer='rmsprop',
#         peptide_length = 9,
#         maxlen_mhc = 181,
#         mhc_activation = 'tanh',
#         peptide_activation = 'tanh'
#         ):
#
#     #hyperparameters = {'sizes':[128,64,128,128,128,64],'dropouts':[0.2,0.2,0.2], 'mult_size':[16,16]}
#     size_names = [
#             'peptide_embedding_size',
#             'mhc_embedding_size',
#             'mhc_hidden_size',
#             'combined_hidden_size',
#             'combined_hidden_final'
#             ]
#
#     dropout_names = [
#             'dropout_merged',
#             'dropout_peptide',
#             'dropout_mhc'
#             ]
#     mult_size = {
#             'mhc_m':hyperparameters['mult_size'][0],
#             'mhc_n':hyperparameters['mult_size'][1]
#     }
#
#     size_dict = {}
#     dropout_dict = {}
#
#     for idx, name in enumerate(size_names):
#         size_dict[name] = hyperparameters['sizes'][idx]
#
#     for idx, name in enumerate(dropout_names):
#         dropout_dict[name] = hyperparameters['dropouts'][idx]
#
#
#     # print size_dict, dropout_dict
#     #
#     # print("Building Graph Model")
#     graph = Graph()
#
#     graph.add_input(name='peptide', ndim=2)
#
#     graph.add_input(name='mhc', ndim=2)
#
#     graph.inputs['peptide'].input = T.imatrix()
#     graph.inputs['mhc'].input = T.imatrix()
#
#     ##PEPTIDE
#
#     graph.add_node(
#                 Embedding(20,size_dict['peptide_embedding_size']),
#                 name='peptide_embedding',
#                 input='peptide')
#
#     graph.add_node(
#                 Flatten(),
#                 name = 'peptide_flatten',
#                 input = 'peptide_embedding')
#
#     graph.add_node(
#                     Dense(size_dict['peptide_embedding_size'] * peptide_length, mult_size['mhc_m'],
#                     activation = peptide_activation ),
#                 name='peptide_dense',
#                 input='peptide_flatten'   )
#
#
#     dropout_output_peptide = 'peptide_dense'
#     if(dropout_dict['dropout_peptide']):
#         dropout_output_peptide = 'peptide_dropout'
#         graph.add_node(Dropout(dropout_dict['dropout_peptide']), name = 'peptide_dropout', input='peptide_dense')
#
#
#
#     ##MHC
#
#     graph.add_node(
#                 Embedding(20,size_dict['mhc_embedding_size']),
#                 name='mhc_embedding',
#                 input='mhc')
#
#     graph.add_node(
#             Flatten(),
#             name = 'mhc_flatten',
#             input = 'mhc_embedding')
#
#     graph.add_node(
#             Dense(size_dict['mhc_embedding_size'] * maxlen_mhc, size_dict['mhc_hidden_size'],
#                 activation = mhc_activation),
#             name='mhc_dense',
#             input='mhc_flatten')
#
#     dropout_output_mhc = 'mhc_dense'
#     if(dropout_dict['dropout_mhc']):
#         dropout_output_mhc = 'mhc_dropout'
#         graph.add_node(Dropout(dropout_dict['dropout_mhc']), name = dropout_output_mhc, input='mhc_dense')
#
#     graph.add_node(Dense(size_dict['mhc_hidden_size'], mult_size['mhc_m'] * mult_size['mhc_n'], activation = mhc_activation),
#                     name = 'mhc_dense_2', input = dropout_output_mhc)
#     graph.add_node(Reshape(mult_size['mhc_m'],mult_size['mhc_n']), name = 'mhc_final', input = 'mhc_dense_2')
#     ##MERGE
#
#     graph.add_node(
#             Dense(mult_size['mhc_n'], size_dict['combined_hidden_size'],
#                 activation = "relu"),
#             name='dense_merged_1',
#             inputs=[dropout_output_peptide,'mhc_final'],
#             merge_mode='matmul')
#
#
#     graph.add_node(
#             Dropout(dropout_dict['dropout_merged']),
#             name = 'dense_dropout_1',
#             input='dense_merged_1')
#
#
#     graph.add_node(
#             Dense(size_dict['combined_hidden_size'],size_dict['combined_hidden_final'],activation = "relu"),
#             name = 'dense_merged_2',
#             input = 'dense_dropout_1')
#
#
#     graph.add_node(
#             Dropout(dropout_dict['dropout_merged']),
#             name = 'dense_dropout_2',
#             input='dense_merged_2')
#
#
#     graph.add_node(
#             Dense(size_dict['combined_hidden_final'],1,activation = "sigmoid"),
#             name = 'dense_output',
#             input = 'dense_dropout_2')
#
#     graph.add_output(
#             name='output',
#             input='dense_output')
#
#     graph.compile(optimizer,{'output':'mse'})
#     return graph
