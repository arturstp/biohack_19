import pandas as pd
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
import matplotlib.pyplot as plt



char_to_num = {
    'A': amino_acid_to_array('A'),
    'C': amino_acid_to_array('C'),
    'D': amino_acid_to_array('D'),
    'E': amino_acid_to_array('E'),
    'F': amino_acid_to_array('F'),
    'G': amino_acid_to_array('G'),
    'H': amino_acid_to_array('H'),
    'I': amino_acid_to_array('I'),
    'K': amino_acid_to_array('K'),
    'L': amino_acid_to_array('L'),
    'M': amino_acid_to_array('M'),
    'N': amino_acid_to_array('N'),
    'P': amino_acid_to_array('P'),
    'Q': amino_acid_to_array('Q'),
    'R': amino_acid_to_array('R'),
    'S': amino_acid_to_array('S'),
    'T': amino_acid_to_array('T'),
    'V': amino_acid_to_array('V'),
    'W': amino_acid_to_array('W'),
    'Y': amino_acid_to_array('Y'),
    'X': amino_acid_to_array('X')
}

mhc_to_num = {
    'H2-Dd': 1, #HOW TO INTERPRET THIS?
    'H2-Db': 2,
    'H2-Kd': 3,
    'H2-Kb': 4,
    'H2-Kk': 5,
    'H2-Ld': 6
}

def amino_acid_to_array(char):
    arr =  np.array(read_amino_acid_parametrs().loc[arr.Litera == 'A'])[:,1:]
    return arr


def read_amino_acid_parametrs():
    amino_acid_parametrs = pd.read_csv('data/amino_acid.csv')
    arr = pd.DataFrame(df, columns=['Litera', 'Alifatyczne', 'Zasadowe', 'Siarkowe', 'Lminokwasy', 'Kwasowe', 'Amidy', 'Aromatyczne', 'Grupa_OH', 'Universal'])
    return arr



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
            text = text[:len(text)//2] + 'X' + text[len(text)//2:]
        data.at[index, 'Peptide'] = text

def get_inputs():
    all_data = read_data('./data/train.tsv')

    add_zeros(all_data)

    data_new = np.empty([all_data.shape[0], 13])  # get 13 not by constant! - and change
    for index, row in all_data.iterrows():
        numbers = convertToNum(row)
        data_new[index] = numbers

    return data_new


def create_model():
    model = Sequential()
    # add embedding
    model.add(Embedding(21, 42, input_length=12))
    # model.add(Dense(22, input_dim=12, activation='relu'))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    EPOCHS = 10
    SPLITS = 10
    data = get_inputs()  #todo: pass filename

    X = data[:, 0:12]
    Y = data[:, 12]
    train_data = data[:6000, :]
    test_data = data[6000:, :]

    train_X = train_data[:, 0:12]
    train_Y = train_data[:, 12]

    test_X = test_data[:, 0:12]
    test_Y = test_data[:, 12]

    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True)
    auc_best = 0.0
    history_best = None
    fpr_kt = None
    tpr_kt = None

    for index, (train_indices, val_indices) in enumerate(skf.split(X, Y)):
        print("Training on fold " + str(index + 1) + "/10...")

        # Generate batches from indices
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = Y[train_indices], Y[val_indices]

        model = None
        model = create_model()

        # Debug message I guess
        print ("Training new iteration on " + str(xtrain.shape[0])\
              + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while...")

        history = model.fit(xtrain, ytrain, epochs=EPOCHS, batch_size=32, validation_split=0.2)

        ypred = model.predict(xval).ravel()

        fpr_k, tpr_k, threholds_k = roc_curve(yval, ypred)
        auc_k = auc(fpr_k, tpr_k)

        if auc_best < auc_k:
            history_best = history
            fpr_kt = fpr_k
            tpr_kt = tpr_k
            auc_best = auc_k
            print('NEW BEST MODEL HISTORY RECORDED')




    # model = create_model()
    #
    #
    # print('starting fitting')
    # H = model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=32, shuffle=True, validation_split=0.2)  #test me..
    #
    # scores = model.evaluate(test_X, test_Y)
    #
    # # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    #
    # y_pred_keras = model.predict(test_X).ravel()
    #
    # fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_Y, y_pred_keras)
    #
    # print(y_pred_keras[:10])
    # auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_kt, tpr_kt, label='Keras (area = {:.3f})'.format(auc_best))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()

    plt.figure(2)
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history_best.history["loss"], label="train_loss")
    plt.plot(N, history_best.history["val_loss"], label="val_loss")
    plt.plot(N, history_best.history["acc"], label="train_acc")
    plt.plot(N, history_best.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    # plt.savefig(args["plot"])





if __name__ == "__main__":
    main()
