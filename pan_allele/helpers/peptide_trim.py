from sequence_encoding import padded_indices
from amino_acid import amino_acid_letter_indices, amino_acid_letters

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

'''takes a single peptide and allele sequence as inputs
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

    return preds
