import pandas as pd
import numpy as np

def main():
    all_data = pd.read_csv('./data/train.tsv', sep='\t')

    for index, row in all_data.iterrows():
        text = row.loc['Peptide']
        while len(text) < 12:
            text = text + "0"
        all_data.at[index, 'Peptide'] = text

    print(all_data)

if __name__ == "__main__":
    main()
