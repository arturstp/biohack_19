import pandas as pd
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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




#param count = 9


def convertToNum(row):
    peptide_in_num = np.empty((1, 0))
    peptide_in_num.ravel()
    # print(peptide_in_num.shape)
    text = row.loc['Peptide']
    i = 0
    for char in text:
        a = char_to_num[char]
        b = np.array(char_to_num[char])
        b.ravel()
        peptide_in_num = np.hstack((peptide_in_num, b))
        i += 1
    # print(peptide_in_num)
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

    data_new = np.empty([all_data.shape[0], 101])  # get 13 not by constant! - and change/// 102 = 11x9 + mhc+react
    for index, row in all_data.iterrows():
        numbers = convertToNum(row)
        # print(numbers)

        data_new[index] = numbers

    return data_new


def create_model():
    model = Sequential()
    # add embedding
    # model.add(Embedding(21, 10, input_length=100))
    print('MODEL OUTPUT = ')
    # print(model.output_shape)
    # model.add(Dense(22, input_dim=12, activation='relu'))
    # model.add(Flatten())
    # model.add(Dropout(0.99))
    print('now after flatten: ')
    print('MODEL OUTPUT = ')
    # print(model.output_shape)
    model.add(Dense(32, activation='relu', input_dim=100))
    # model.add(Dense(8, activation='relu'))
    print('now after dense:')
    print('MODEL OUTPUT = ')
    print(model.output_shape)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.get_config())
    return model


def main():
    EPOCHS = 12
    SPLITS = 10
    data = get_inputs()  #todo: pass filename

    X = data[:, 0:100]
    Y = data[:, 100]

    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True)
    auc_best = 0.0
    history_best = None
    fpr_kt = None
    tpr_kt = None
    all_auc = []
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

        es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
        history = model.fit(xtrain, ytrain, epochs=EPOCHS, batch_size=32, validation_data=(xval, yval), callbacks=[es])

        ypred = model.predict(xval).ravel()

        fpr_k, tpr_k, threholds_k = roc_curve(yval, ypred)
        auc_k = auc(fpr_k, tpr_k)

        if auc_best < auc_k:
            history_best = history
            fpr_kt = fpr_k
            tpr_kt = tpr_k
            auc_best = auc_k
            print('NEW BEST MODEL HISTORY RECORDED')

        all_auc.append(auc_k)

    print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(all_auc), np.std(all_auc)))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_kt, tpr_kt, label='Keras (area = {:.3f})'.format(auc_best))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.figure(2)
    N = np.arange(0, len(history_best.history["loss"]))
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


if __name__ == "__main__":
    main()
