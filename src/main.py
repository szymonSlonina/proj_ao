import cv2
import os
import pandas as pd
import string
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import src.im_prep.functions as fun
import src.resources.train_dataset.process_pictures as proc_train

if __name__ == "__main__":

    RUN_MODELING = True

    REPROCESS_TEST_LETTERS = False
    REPROCESS_TRAIN_LETTERS = False
    REPROCESS_TEST_DATA = False
    REPROCESS_TRAIN_DATA = False

    TEST_DATASET_PATH = 'resources' + os.sep + 'datasets' + os.sep + 'test_dataset.csv'
    TRAIN_DATASET_PATH = 'resources' + os.sep + 'datasets' + os.sep + 'train_dataset.csv'

    TEST_PICTURES_PATH = 'resources' + os.sep + 'test_pictures' + os.sep
    TRAIN_PICTURES_PATH = 'resources' + os.sep + 'train_dataset' + os.sep + 'Fnt_preproc'
    TEST_LETTERS_PATH = 'resources' + os.sep + 'test_letters' + os.sep
    TEST_IMAGE = 'test1.bmp'


    #
    #JEŚLI CHCESZ URUCHOMIĆ OD NOWA PREPROCESSING WSZYSTKICH ZDJĘĆ Z KATALOGU resources/train_dataset
    # ZMIEŃ FLAGE REROCESS_TRAIN_LETTERS
    #
    #to spowoduje uzupełnienie katalogu resources/train_dataset/Fnt_preproc zdjęciami przerobionymi
    if REPROCESS_TRAIN_LETTERS:
        proc_train.preprocess()


    #
    #JEŚLI CHCESZ URUCHOMIĆ PROCES PRZERABIANIA SKANOWANEGO DOKUMENTU (TESTOWYCH PRÓBEK)
    # TAK ŻEBY LITERY BYŁY WYEKSTRACHOWANE I PREPROCESSOWANE TO ZMIEŃ FLAGĘ REPROCESS_TEST_LETTERS
    #
    #to spowoduje wypełnienie katalogu test_letters zdjęciami liter z danego skanu
    if REPROCESS_TEST_LETTERS:
        im = cv2.imread(TEST_PICTURES_PATH + TEST_IMAGE, 0)
        letters = fun.preproces_scan(im)
        for ind1, row in enumerate(letters):
            for ind2, letter in enumerate(row):
                cv2.imwrite(TEST_LETTERS_PATH + 'img_' + str(ind1) + '_' + str(ind2) + '.png', letter)
            print(ind1)


    #
    #JEŚLI CHCESZ URUCHOMIĆ PROCES GENEROWANIA ZBIORU DANYCH (WEKTORÓW CECH ETC.) ZE ZBIORU ZDJĘĆ TRENINGOWYCH
    # ZMIEŃ FLAGĘ REPROCES_TRAIN_DATA
    #
    # spowoduje to wygenerowanie się pliku train_dataset.cvs w katalogu datasets w resources
    #
    if REPROCESS_TRAIN_DATA:
        train_indexes = list(range(10))
        train_indexes.extend(list(string.ascii_uppercase))
        train_indexes.extend(list(string.ascii_lowercase))

        file = open(TRAIN_DATASET_PATH, 'w')

        letter_dirs = sorted(os.listdir(TRAIN_PICTURES_PATH))
        for ind, letter_dir in enumerate(letter_dirs):
            if letter_dir == 'file_for_structure.txt':
                continue

            photos = sorted(os.listdir(TRAIN_PICTURES_PATH + os.sep + letter_dir))
            for ind2, photo in enumerate(photos):
                im = cv2.imread(TRAIN_PICTURES_PATH + os.sep + letter_dir + os.sep + photo, 0)
                im = cv2.bitwise_not(im)

                im = np.reshape(im, 32*32)
                for data in im:
                    file.write(str(data))
                    file.write(',')
                file.write(str(train_indexes[ind]))
                file.write('\n')

            print(ind)
        file.close()


    #
    #JEŚLI CHCESZ URUCHOMIĆ PROCES GENEROWANIA ZBIORU DANYCH (WEKTORÓW CECH ETC.) ZE ZBIORU ZDJĘĆ TESTOWYCH
    # ZMIEŃ FLAGĘ REPROCES_TEST_DATA
    #
    # spowoduje to wygenerowanie się pliku test_dataset.csv w katalogu datasets w resources
    #
    if REPROCESS_TEST_DATA:
        file = open(TEST_DATASET_PATH, 'w')
        letters_dir = sorted(os.listdir(TEST_LETTERS_PATH))
        for ind, letter_dir in enumerate(letters_dir):
            im = cv2.imread(TEST_LETTERS_PATH + os.sep +letter_dir, 0)
            im = cv2.bitwise_not(im)

            im = np.reshape(im, 32*32)
            for data in im:
                file.write(str(data))
                file.write(',')
            file.write('\n')

            print(ind)
        file.close()

    #model będzie się odpalał po zmianie flagi RUN_MODELING
    # tutaj jest właściwe uczenie i model...
    # wszystko klasyfikuje jako "i" podejrzewam dla tego że jest najmniejsze i zajmuje najmniej czarnych pikseli...
    if RUN_MODELING:
        dataset_train = pd.read_csv(TRAIN_DATASET_PATH, header=None, low_memory=False)
        X_train = dataset_train[dataset_train.columns[0:-1]]
        y_train = dataset_train[dataset_train.columns[-1]]

        dataset_test = pd.read_csv(TEST_DATASET_PATH, header=None, low_memory=False)
        X_test = dataset_test[dataset_test.columns[0:-1]]


        #now we try to build model of knn
        model = KNeighborsClassifier(n_neighbors=250)
        model.fit(X_train, y_train)

        y_test = model.predict(X_test)
        print(y_test)