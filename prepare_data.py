import pandas as pd
import numpy as np
import sys
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dropout, concatenate
from keras.callbacks import EarlyStopping
from keras import Input
from keras import Model
import matplotlib.pyplot as plt

# converting amino_acids to set of parameters
def read_amino_acid_parametrs():
    amino_acid_parametrs = pd.read_csv('data/amino_acid.csv')
    arr = pd.DataFrame(amino_acid_parametrs, columns=['Litera', 'Alifatyczne', 'Zasadowe', 'Siarkowe', 'Lminokwasy', 'Kwasowe', 'Amidy', 'Aromatyczne', 'Grupa_OH', 'Universal'])
    return arr


def amino_acid_to_array(char):
    arr =  np.array(read_amino_acid_parametrs().loc[read_amino_acid_parametrs().Litera == char])[:,1:]
    return arr


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


def convertToNum(row):
    peptide_in_num = np.empty((0, 10))
    peptide_in_num.ravel()
    text = row.loc['Peptide']
    i = 0
    for char in text:
        b = np.array(char_to_num[char])
        b.ravel()
        b = np.append(b, mhc_to_num[row.loc['MHC']])
        peptide_in_num = np.vstack((peptide_in_num, b))
        i += 1

    return peptide_in_num, row.loc['Immunogenicity']

def read_data(path):
    return pd.read_csv(path, sep='\t')


def add_zeros(data):
    for index, row in data.iterrows():
        text = row.loc['Peptide']
        while len(text) < 11:
            text = text[:len(text)//2] + 'X' + text[len(text)//2:]
        data.at[index, 'Peptide'] = text

def get_inputs(filename):
    all_data = read_data(filename)

    add_zeros(all_data)

    acids_data = np.empty([all_data.shape[0], 11, 10])
    react_data = np.empty((all_data.shape[0], 1))
    for index, row in all_data.iterrows():
        # acids_data[index], mhc_data[index], react_data[index] = convertToNum(row)
        acids_data[index], react_data[index] = convertToNum(row)
        # print(numbers)

        # data_new[index] = numbers
    return acids_data, react_data
    # return acids_data, mhc_data, react_data


def create_model():

    input_acid = Input(shape=(11,9))
    input_mhc = Input(shape=(1,))

    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_acid)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Model(inputs=input_acid, outputs=x)

    y = Embedding(7, 14, input_length=1)(input_mhc)
    y = Flatten()(y)
    y = Dense(16, activation='relu')(y)
    y = Model(inputs=input_mhc, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dense(10, activation="relu")(combined)
    z = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# pass vector with amino_acids and mhc ids and get them separately
def split_acids_mhc(data):
    x_main = data[:, :, :9]
    x_mhc = data[:, -1, 9:]
    return x_main, x_mhc

def main(argv):
    EPOCHS = 20
    SPLITS = 10
    # data, mhc, react = get_inputs()  #todo: pass filename
    train_set_path = './data/train.tsv'
    data, react = get_inputs(train_set_path)
    X = data
    # X1 = data
    # X2 = mhc
    Y = react

    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True)
    auc_best = 0.0
    history_best = None
    model_best = None
    fpr_keras = None
    tpr_keras = None
    all_auc = []
    for index, (train_indices, val_indices) in enumerate(skf.split(X, Y)):
        print("Training on fold " + str(index + 1) + "/10...")

        # Generate batches from indices
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = Y[train_indices], Y[val_indices]
        print(xval.shape)
        x_main, x_mhc = split_acids_mhc(xtrain)
        x_main_val, x_mhc_val = split_acids_mhc(xval)
        model = None
        model = create_model()

        # Debug message I guess
        print ("Training new iteration on " + str(xtrain.shape[0])\
              + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while...")

        #end when loss func stops improving
        es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)

        history = model.fit([x_main, x_mhc], ytrain, epochs=EPOCHS, batch_size=96, validation_data=([x_main_val, x_mhc_val], yval), callbacks=[es])
        ypred = model.predict([x_main_val, x_mhc_val]).ravel()

        fpr_k, tpr_k, threholds_k = roc_curve(yval, ypred)
        auc_k = auc(fpr_k, tpr_k)

        if auc_best < auc_k:
            history_best = history
            fpr_keras = fpr_k
            tpr_keras = tpr_k
            auc_best = auc_k
            model_best = model
            print('NEW BEST MODEL HISTORY RECORDED')

        all_auc.append(auc_k)


    # model.predict()

    display_data(all_auc, auc_best, fpr_keras, history_best, tpr_keras)


def display_data(all_auc, auc_best, fpr_keras, history_best, tpr_keras):
    print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(all_auc), np.std(all_auc)))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_best))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.figure(1)
    N = np.arange(0, len(history_best.history["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history_best.history["loss"], label="train_loss")
    plt.plot(N, history_best.history["val_loss"], label="val_loss")
    plt.plot(N, history_best.history["acc"], label="train_acc")
    plt.plot(N, history_best.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (best model)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
