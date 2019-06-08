import pandas as pd
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
import matplotlib.pyplot as plt



char_to_num = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19,
    'X': 20
}

mhc_to_num = {
    'H2-Dd': 1, #HOW TO INTERPRET THIS?
    'H2-Db': 2,
    'H2-Kd': 3,
    'H2-Kb': 4,
    'H2-Kk': 5,
    'H2-Ld': 6
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
        while len(text) < 11:
            text = text + 'X' #noli w seredine, a ne na koncah
        data.at[index, 'Peptide'] = text

def get_inputs():
    all_data = read_data('./data/train.tsv')

    add_zeros(all_data)

    data_new = np.empty([all_data.shape[0], 13])  # get 13 not by constant! - and change
    for index, row in all_data.iterrows():
        numbers = convertToNum(row)
        data_new[index] = numbers

    return data_new

def main():

    data = get_inputs()  #todo: pass filename


    train_data = data[:6000, :]
    test_data = data[6000:, :]

    train_X = train_data[:, 0:12]
    train_Y = train_data[:, 12]

    test_X = test_data[:, 0:12]
    test_Y = test_data[:, 12]

    model = Sequential()
    #add embedding
    model.add(Embedding(21, 21, input_length=12))
    # model.add(Dense(22, input_dim=12, activation='relu'))
    model.add(Flatten())
    model.add(Dense(22, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('starting fitting')
    model.fit(train_X, train_Y, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)  #test me..

    scores = model.evaluate(test_X, test_Y)

    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_pred_keras = model.predict(test_X).ravel()

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_Y, y_pred_keras)

    print(y_pred_keras[:10])
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
