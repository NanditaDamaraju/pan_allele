
from pan_allele.helpers.pan_allele_data_helpers import *
from pan_allele.helpers.hyperparameters import get_graph_from_hyperparameters

from paths import *

import numpy as np

nb_peptides = 100
peptides = []
for i in range(nb_peptides):
    peptides.append(np.random.randint(20, size=9))

allele_sequence_data, max_allele_length = load_allele_sequence_data(SEQUENCE_DATA_PATH)


allele_list = ['A0101',	    'A0201',	'A0202',    'A0203',	'A0206',	'A0301',
               'A1101',	    'A2301',	'A2402',	'A2501',	'A2601',    'A2602',
               'A2603',	    'A2902',	'A3001',	'A3002',	'A3101',	'A3201',
               'A3301',	    'A6801',	'A6802',	'A6901',    'A8001',	'B0702',
               'B0801',	    'B0802',	'B0803',	'B1501',	'B1503',    'B1509',
               'B1517',	    'B1801',	'B2703',    'B2705',    'B3501',	'B3801',
               'B3901',	    'B4001',	'B4002',	'B4402',	'B4403',	'B4501',
               'B4601',	    'B5101',    'B5301',	'B5401',	'B5701',	'B5801'	]


graph = get_graph_from_hyperparameters('ffn_mult')
for epoch in range(1,99):
    graph.load_weights(HOME_PATH + 'weights/ffn_mult/weights32_' + str(epoch))
    predictions = np.empty(len(allele_list)*nb_peptides)
    counter = 0
    for allele in allele_list:
        allele_sequence = mhc_seq = padded_indices([allele_sequence_data[allele]],
                                        add_start_symbol=False,
                                        add_end_symbol=False,
                                        index_dict=amino_acid_letter_indices)

        for peptide in peptides:
            predictions[counter] = 20000**(1-graph.predict({'peptide':[[peptide]],'mhc':mhc_seq})['output'])
            counter = counter + 1

    print epoch, np.sum(predictions<500)
