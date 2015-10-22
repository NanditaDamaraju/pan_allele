from __future__ import print_function
import sys
####APPEND PATH FROM DIRECTORY AFTER INSTALLATION
path = "/home/ubuntu"
#path = "/Users/NanditaD/Intern/mhclearn"
sys.path.append(path + "/py/")


from pan_allele.pan_allele_data_helpers import *
from pan_allele.leave_one_out_validation import *
from pan_allele.feedforward_models import build_graph_native_sequence_model, build_graph_siamese_embedding_model, build_sequential_model
from sklearn.metrics import roc_auc_score, accuracy_score


def main(job_id, params):
        mhc_sequence_fasta_file=path+"/py/pan_allele/files/pseudo/pseudo_sequences.fasta"
        print(mhc_sequence_fasta_file)
        iedb_data_file= path + "/py/pan_allele/files/bdata.2009.mhci.public.1.txt"

        embedding_model=None
        max_sequence_length=None
        print(params)
        result = leave_one_out(
                    path,
                    hyperparameters=params,
                    mhc_sequence_fasta_file=mhc_sequence_fasta_file,
                    iedb_data_file=iedb_data_file,
                    peptide_length = 9,
                    max_sequence_length=None,
                    embedding_model=None,
                    siamese_embedding=False,
                    nb_epoch=10, max_ic50 = 5000.0)

        return result

if __name__ == "__main__":
    #main(23,  {'cutoff':[ 0.48023558], 'dropouts': [ 0.33062075,  0.21962448,  0.2374656 ], 'sizes': [193, 149, 134, 102, 128]})
    main (23, {'cutoff': [ 0.31173147], 'dropouts': [ 0.,  0.,  0.], 'sizes': [   8, 126, 123, 128, 128,  35]})



if __name__ == "__main__":
    #main(23,  {'cutoff':[ 0.48023558], 'dropouts': [ 0.33062075,  0.21962448,  0.2374656 ], 'sizes': [193, 149, 134, 102, 128]})
    #main (23, {'cutoff': [ 0.31173147], 'dropouts': [ 0.,  0.,  0.], 'sizes': [   8, 126, 123, 128, 128,  35]})
    #main(23, {'cutoff': [ 0.22145585], 'dropouts': [ 0.24845259,  0.07000219,  0.08056138], 'sizes': [119,  63, 121, 110,  97, 100,  86]})
    main(42, {'cutoff': [ 0.31780098], 'dropouts': [ 0.00467912,  0.00682629,  0.00852883],'sizes': [119,  57, 114, 111, 108, 106,  83]})
