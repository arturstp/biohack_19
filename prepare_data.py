import pandas as pd
import numpy as np

char_to_num = {
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
    'X': 21
}

mhc_to_num = {
    'H2-Dd': 101, #HOW TO INTERPRET THIS?
    'H2-Db': 102,
    'H2-Kd': 103,
    'H2-Kb': 104,
    'H2-Kk': 105,
    'H2-Ld': 106
}


def convertToNum(row):
    peptide_in_num = np.zeros(len(row.loc['Peptide']))
    text = row.loc['Peptide']
    i = 0
    for char in text:
        peptide_in_num[i] = char_to_num[char]
        i += 1
    return np.concatenate((peptide_in_num, mhc_to_num[row.loc['MHC']], row.loc['Immunogenicity']), axis=None)


def read_data(path):
    return pd.read_csv(path, sep='\t')


def add_zeros(data):
    for index, row in data.iterrows():
        text = row.loc['Peptide']
        while len(text) < 12:
            text = text + 'X'
        data.at[index, 'Peptide'] = text

def get_inputs():
    all_data = read_data('./data/train.tsv')

    add_zeros(all_data)

    data_new = np.empty([all_data.shape[0], 14])  # get 13 not by constant! - and change
    for index, row in all_data.iterrows():
        numbers = convertToNum(row)
        data_new[index] = numbers

    return data_new

def main():

    data = get_inputs(); #todo: pass filename


if __name__ == "__main__":
    main()
