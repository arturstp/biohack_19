import pandas as pd
import numpy as np

def main():
    all_data = pd.read_csv('~/biohack/data/train.tsv', sep='\t')
    print(all_data)


if __name__ == "__main__":
    main()
