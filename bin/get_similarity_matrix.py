import pandas as pd
from pan_allele_data_helpers import load_binding_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set()

def allelename(allele):
    bad_hla_name_elements = [
        "HLA-",
        "-",
        "*",
        ":"
    ]
    for substring in bad_hla_name_elements:
            allele = allele.replace(substring, "")
    return allele

def binary(val):
    if(val > 500):
        return 0
    else:
        return 1


data, df = load_binding_data('files/bdata.2009.mhci.public.1.txt')
df['mhc'] = df['mhc'].apply(allelename)

allele_list = []
for alleles in data:
    allele_list.append(alleles)

data = np.zeros((len(allele_list),len(allele_list)))

with open("files/allele_similarity_2009.csv",'wb') as f:
    f.write("allele1,allele2,similarity,overlap\n")
    for ind_x, allele_x in enumerate(allele_list):
        print(allele_x)
        for ind_y, allele_y in enumerate(allele_list):
            new_df = df[    (df['mhc'] == allele_x)  |    (df['mhc'] == allele_y)   ]

            pepetide_x = new_df[new_df['mhc'] == allele_x]['sequence']
            pepetide_y = new_df[new_df['mhc'] == allele_y]['sequence']

            peptides = pd.Series(list(set(pepetide_x).intersection(set(pepetide_y))))
            new_df  = new_df[new_df.sequence.isin(peptides)]

            peptide_vector_x = new_df[new_df['mhc'] == allele_x]['meas'].apply(binary)
            peptide_vector_y = new_df[new_df['mhc'] == allele_y]['meas'].apply(binary)

            similarity = np.mean(peptide_vector_x == peptide_vector_y)

            f.write(allele_x + ',' + allele_y + ',' + str(float(similarity)) + ',' + str(len(peptides))+'\n')
