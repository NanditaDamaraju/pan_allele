from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Merge, Permute
from keras.optimizers import RMSprop
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import theano.tensor as T

def convolution_graph_1(maxlen_mhc=34):
    max_features = 20
    maxlen_peptide = 9
    maxlen_mhc = maxlen_mhc
    batch_size = 32
    embedding_dims = 64
    nb_filters = 20
    filter_length = 4
    hidden_dims = 25


    graph = Graph()
    graph.add_input(name='peptide', ndim=2)
    graph.add_input(name='mhc',ndim=2)
    graph.inputs['mhc'].input = T.imatrix()
    graph.inputs['peptide'].input = T.imatrix()

    ##PEPTIDE
    graph.add_node(Embedding(max_features, embedding_dims),
                   name = 'peptide_embedding',
                   input = 'peptide')

    graph.add_node( Convolution1D(
                                input_dim=embedding_dims,
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                border_mode="valid",
                                activation="relu",
                                subsample_length=1
                                ),

                    name = 'peptide_conv',
                    input = 'peptide_embedding')
    graph.add_node( Flatten(), name = 'peptide_flatten', input = 'peptide_conv')

    output_size_peptide = nb_filters * (((maxlen_peptide - filter_length) / 1) + 1)
    graph.add_node( Dense(output_size_peptide, hidden_dims,activation = "relu" ),
                    name = 'peptide_dense',
                    input = 'peptide_flatten')

    ##MHC
    graph.add_node( Embedding(max_features, embedding_dims),
                    name = 'mhc_embedding',
                    input = 'mhc')

    graph.add_node( Convolution1D(
                            input_dim=embedding_dims,
                            nb_filter=nb_filters,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1
                            ),

                    name = 'mhc_conv',
                    input = 'mhc_embedding')

    graph.add_node( MaxPooling1D(pool_length=2), name = 'mhc_pool', input = 'mhc_conv')
    graph.add_node( Flatten(), name = 'mhc_flatten', input = 'mhc_pool')
    output_size_mhc = nb_filters * (((maxlen_mhc - filter_length) / 1) + 1) / 2
    graph.add_node( Dense( output_size_mhc, hidden_dims,activation = "relu" ),
                    name = 'mhc_dense',
                    input = 'mhc_flatten')


    ##MERGE
    last_peptide = 'peptide_dense'
    last_mhc     = 'mhc_dense'
    graph.add_node( Dense( hidden_dims*2, 32, activation = "relu"),
                    name='merged_output',
                    inputs=[last_peptide,last_mhc],
                    merge_mode='concat'
                )
    graph.add_node( Dense(32,1, activation='sigmoid'),
                    name= 'merged_final',
                    input = 'merged_output'
                )
    graph.add_output( name='output', input='merged_final')
    graph.compile('rmsprop',{'output':'mse'})
    return graph

def convolution_graph(maxlen_mhc=34):
    max_features = 20
    maxlen_peptide = 9
    batch_size = 32
    embedding_dims = 64
    nb_filters = 10
    filter_length = 3
    hidden_dims = 25

    graph = Graph()
    graph.add_input(name='peptide', ndim=2)
    graph.add_input(name='mhc',ndim=2)
    graph.inputs['mhc'].input = T.imatrix()
    graph.inputs['peptide'].input = T.imatrix()

    ##PEPTIDE
    graph.add_node( Embedding(max_features, embedding_dims),
                    name = 'peptide_embedding',
                    input = 'peptide'
                   )
    graph.add_node( Convolution1D(
                        input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1),

                    name = 'peptide_conv',
                    input = 'peptide_embedding'
                    )


    graph.add_node( Flatten(), name = 'peptide_flatten', input = 'peptide_conv')
    output_size_peptide = nb_filters * (((maxlen_peptide - filter_length) / 1) + 1)
    graph.add_node( Dense(output_size_peptide,  hidden_dims, activation = "relu" ),
                    name = 'peptide_dense',
                    input = 'peptide_flatten'
                    )

    ##MHC
    graph.add_node( Embedding(max_features, embedding_dims),
                    name = 'mhc_embedding',
                    input = 'mhc'
                    )
    graph.add_node( Convolution1D(
                        input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1),

                    name = 'mhc_conv_1',
                    input = 'mhc_embedding'
                    )

    graph.add_node( MaxPooling1D(pool_length=2),    name = 'mhc_pool_1',    nput = 'mhc_conv_1')
    output_size_mhc = nb_filters * (((maxlen_mhc - filter_length) / 1) + 1) / 2
    graph.add_node( Convolution1D(
                            input_dim = nb_filters/2,
                             nb_filter = nb_filters,
                             filter_length= filter_length,
                             border_mode="valid",
                             activation="relu",
                             subsample_length=1),

                    name= 'mhc_conv_2',
                    input = 'mhc_pool_1'
                    )

    graph.add_node( MaxPooling1D(pool_length=2),    name = 'mhc_pool_2',    input = 'mhc_conv_2')
    graph.add_node( Flatten(), name = 'mhc_flatten_2', input = 'mhc_pool_2')
    output_size_mhc = nb_filters * (maxlen_mhc/2.0  - filter_length + 1)
    print "output_size_mhc", output_size_mhc
    graph.add_node( Dense(output_size_mhc,  hidden_dims, activation = "relu" ),
                    name = 'mhc_dense',
                    input = 'mhc_flatten_2'
                    )




    ##MERGE
    last_peptide = 'peptide_dense'
    last_mhc = 'mhc_dense'
    graph.add_node( Dense(hidden_dims*2, 32, activation = "relu"),
                    name='merged_output',
                    inputs=[last_peptide,last_mhc],
                    merge_mode='concat'
                    )
    graph.add_node( Dense(32,1, activation='sigmoid'),
                    name= 'merged_final',
                    input = 'merged_output'
                    )
    graph.add_output(   name='output',  input='merged_final')


    graph.compile(  'rmsprop',  {'output':'mse'}  )
    return graph

def convolution_graph_reshape(
                hyperparameters = None,
                maxlen_mhc = 181,
                optimizer='rmsprop',
                mhc_activation='tanh',
                peptide_activation= 'tanh',
                maxlen_peptide = 9):

    max_features = 20
    print ("PARAMS", hyperparameters)

    nb_filters = {
                    'peptide': hyperparameters['nb_filter'][0],
                    'mhc':hyperparameters['nb_filter'][1]
                }
    filter_lengths = {
                        'peptide':hyperparameters['filter_length'][0],
                        'mhc':hyperparameters['filter_length'][1]
                     }
    layer_sizes = {
            'peptide_embedding_size':hyperparameters['layer_size'][0],
            'mhc_embedding_size':hyperparameters['layer_size'][1],
            'peptide_hidden_size':hyperparameters['layer_size'][2],
            'mhc_hidden_size':hyperparameters['layer_size'][3],
            'merged_hidden_size':hyperparameters['layer_size'][4]
            }

    graph = Graph()

    graph.add_input(name='peptide', ndim=2)
    graph.add_input(name='mhc', ndim=2)
    graph.inputs['mhc'].input = T.imatrix()
    graph.inputs['peptide'].input = T.imatrix()



    ##PEPTIDE
    graph.add_node( Embedding(max_features, layer_sizes['peptide_embedding_size']),
                    name = 'peptide_embedding',
                    input = 'peptide'
                    )
    graph.add_node( Convolution1D(
                        input_dim=layer_sizes['peptide_embedding_size'],
                        nb_filter=nb_filters['peptide'],
                        filter_length=filter_lengths['peptide'],
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1),

                    name = 'peptide_conv',
                    input = 'peptide_embedding'
                    )


    graph.add_node( Flatten(),  name = 'peptide_flatten',   input = 'peptide_conv')
    output_size_peptide = nb_filters['peptide'] * (((maxlen_peptide - filter_lengths['peptide']) / 1) + 1)
    graph.add_node( Dense(  output_size_peptide, layer_sizes['peptide_hidden_size'],    activation = peptide_activation ),
                    name = 'peptide_dense',
                    input = 'peptide_flatten'
                    )

    ##MHC
    graph.add_node( Embedding(  max_features,   layer_sizes['mhc_embedding_size']),
                    name = 'mhc_embedding',
                    input = 'mhc'
                    )


    ##1st MHC convolution
    graph.add_node( Convolution1D(
                        input_dim=layer_sizes['mhc_embedding_size'],
                        nb_filter=nb_filters['mhc'],
                        filter_length=filter_lengths['mhc'],
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1),

                    name = 'mhc_conv_1',
                    input = 'mhc_embedding'
                    )

    graph.add_node( Permute((2,1)), name = 'mhc_reshape_1', input = 'mhc_conv_1')
    graph.add_node( MaxPooling1D(pool_length=2),    name = 'mhc_pool_1',    input = 'mhc_reshape_1')
    maxlen_mhc_new = (maxlen_mhc - filter_lengths['mhc'] + 1) / 2

    graph.add_node( Permute((2,1)), name = 'mhc_reshape_2', input = 'mhc_pool_1')


    ##2nd MHC convolution
    graph.add_node(Convolution1D(input_dim = nb_filters['mhc'],
                                 nb_filter = nb_filters['mhc'],
                                 filter_length= filter_lengths['mhc'],
                                 border_mode="valid",
                                 activation="relu",
                                 subsample_length=1),
                                 name= 'mhc_conv_2',
                                 input = 'mhc_reshape_2'
                    )

    graph.add_node( Permute((2,1)), name = 'mhc_reshape_3', input = 'mhc_conv_2')
    graph.add_node(MaxPooling1D(pool_length=2), name = 'mhc_pool_2', input = 'mhc_reshape_3')
    maxlen_mhc_new  = ( maxlen_mhc_new - filter_lengths['mhc']+ 1) / 2
    graph.add_node( Permute((2,1)), name = 'mhc_reshape_4', input = 'mhc_pool_2')



    graph.add_node( Flatten(), name = 'mhc_flatten_2', input = 'mhc_reshape_4')
    output_size_mhc = nb_filters['mhc'] * maxlen_mhc_new
    graph.add_node( Dense(output_size_mhc,  layer_sizes['mhc_hidden_size'],    activation = mhc_activation ),
                    name = 'mhc_dense',
                    input = 'mhc_flatten_2'
                    )




    ##MERGE
    last_peptide = 'peptide_dense'
    last_mhc = 'mhc_dense'

    graph.add_node( Dense(layer_sizes['mhc_hidden_size']+layer_sizes['peptide_hidden_size'], layer_sizes['merged_hidden_size'],    activation = "relu"),
                    name='merged_output',
                    inputs=[last_peptide,last_mhc],
                    merge_mode='concat'
                    )

    graph.add_node( Dense(layer_sizes['merged_hidden_size'],1, activation='sigmoid'),
                    name= 'merged_final',
                    input = 'merged_output'
                    )

    graph.add_output(   name='output',  input='merged_final')
    graph.compile(  optimizer,  {'output':'mse'})
    return graph

def convolution_graph_matrix(hyperparameters = None,
                            maxlen_mhc = 181,
                            optimizer='rmsprop',
                            mhc_activation='tanh',
                            peptide_activation= 'tanh',
                            maxlen_peptide = 9):

    max_features = 20
    print ("PARAMS", hyperparameters)

    nb_filters = {
        'peptide': hyperparameters['nb_filter'][0],
        'mhc':hyperparameters['nb_filter'][1]
    }
    filter_lengths = {
            'peptide':hyperparameters['filter_length'][0],
            'mhc':hyperparameters['filter_length'][1]
         }
    layer_sizes = {
    'peptide_embedding_size':hyperparameters['layer_size'][0],
    'mhc_embedding_size':hyperparameters['layer_size'][1],
    'merged_hidden_size':hyperparameters['layer_size'][2]
    }

    mult_size = {
    'mhc_m': hyperparameters['mult_size'][0],
    'mhc_n': hyperparameters['mult_size'][1]
    }

    max_features = 20


    graph = Graph()
    graph.add_input(name='peptide', ndim=2)
    graph.add_input(name='mhc', ndim=2)
    graph.inputs['mhc'].input = T.imatrix()
    graph.inputs['peptide'].input = T.imatrix()


    ##MHC
    graph.add_node( Embedding(  max_features,  layer_sizes['mhc_embedding_size']),
                name = 'mhc_embedding',
                input = 'mhc'
                )


    ##1st MHC convolution
    graph.add_node( Convolution1D(
                    input_dim=layer_sizes['mhc_embedding_size'],
                    nb_filter=nb_filters['mhc'],
                    filter_length=filter_lengths['mhc'],
                    border_mode="valid",
                    activation="relu",
                    subsample_length=1),

                name = 'mhc_conv_1',
                input = 'mhc_embedding'
                )

    graph.add_node( Permute((2,1)), name = 'mhc_reshape_1', input = 'mhc_conv_1')
    graph.add_node( MaxPooling1D(pool_length=2),    name = 'mhc_pool_1',    input = 'mhc_reshape_1')
    maxlen_mhc_new = (maxlen_mhc - filter_lengths['mhc'] + 1) / 2

    graph.add_node( Permute((2,1)), name = 'mhc_reshape_2', input = 'mhc_pool_1')


    ##2nd MHC convolution
    graph.add_node(Convolution1D(input_dim = nb_filters['mhc'],
                             nb_filter = nb_filters['mhc'],
                             filter_length= filter_lengths['mhc'],
                             border_mode="valid",
                             activation="relu",
                             subsample_length=1),
                             name= 'mhc_conv_2',
                             input = 'mhc_reshape_2'
                )

    graph.add_node( Permute((2,1)), name = 'mhc_reshape_3', input = 'mhc_conv_2')
    graph.add_node(MaxPooling1D(pool_length=2), name = 'mhc_pool_2', input = 'mhc_reshape_3')
    maxlen_mhc_new  = ( maxlen_mhc_new - filter_lengths['mhc'] + 1) / 2
    graph.add_node( Permute((2,1)), name = 'mhc_reshape_4', input = 'mhc_pool_2')
    graph.add_node(Flatten(), name = 'mhc_flatten', input = 'mhc_reshape_4')
    graph.add_node(Dense(maxlen_mhc_new *nb_filters['mhc'], mult_size['mhc_m']*mult_size['mhc_n'], activation = mhc_activation),
                    name = 'mhc_dense', input = 'mhc_flatten')
    graph.add_node(Reshape(mult_size['mhc_m'],mult_size['mhc_n']), name = 'mhc_final', input = 'mhc_dense')

    ##PEPTIDE
    graph.add_node( Embedding(max_features, layer_sizes['peptide_embedding_size']),
                name = 'peptide_embedding',
                input = 'peptide'
                )
    graph.add_node( Convolution1D(
                    input_dim=layer_sizes['peptide_embedding_size'],
                    nb_filter=nb_filters['peptide'],
                    filter_length=filter_lengths['peptide'],
                    border_mode="valid",
                    activation="relu",
                    subsample_length=1),

                name = 'peptide_conv',
                input = 'peptide_embedding'
                )
    graph.add_node( Flatten(),  name = 'peptide_flatten',   input = 'peptide_conv')
    output_size_peptide = nb_filters['peptide'] * (((maxlen_peptide - filter_lengths['peptide']) / 1) + 1)
    graph.add_node( Dense(  output_size_peptide,   mult_size['mhc_m'],    activation = peptide_activation ),
                name = 'peptide_dense',
                input = 'peptide_flatten'
                )




    ##MERGE
    last_peptide = 'peptide_dense'
    last_mhc = 'mhc_final'
    graph.add_node( Dense(mult_size['mhc_n'], layer_sizes['merged_hidden_size'],    activation = "relu"),
                name='merged_output',
                inputs=[last_peptide,last_mhc],
                merge_mode='matmul'
                )

    graph.add_node( Dense(layer_sizes['merged_hidden_size'],1, activation='sigmoid'),
                name= 'merged_final',
                input = 'merged_output'
                )

    graph.add_output(   name='output',  input='merged_final')


    graph.compile( optimizer,  {'output':'mse'})
    return graph
