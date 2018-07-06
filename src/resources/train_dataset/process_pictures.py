import os
import cv2
import src.im_prep.functions as fun

#this is script for im_prep photos in this folder
# we have to segment each picture, resize it and skeletonize it
# than we write it to Fnt_preproc folder

def preprocess():
    CWD = os.getcwd()
    RAW_LETTERS_DIR = CWD + os.sep + 'resources' + os.sep + 'train_dataset' + os.sep + 'Fnt'
    PREPROCESSED_LETTERS_DIR = CWD + os.sep + 'resources' + os.sep + 'train_dataset' + os.sep + 'Fnt_preproc'

    #get all catalogues in raw letters dir
    raw_letters_dir_dirs = os.listdir(RAW_LETTERS_DIR)
    for ind1, letter_dir in enumerate(raw_letters_dir_dirs):

        #if this dir is not in Fnt_preproc, we have to create it
        if not os.path.exists(PREPROCESSED_LETTERS_DIR + os.sep + letter_dir):
            os.makedirs(PREPROCESSED_LETTERS_DIR + os.sep + letter_dir)

        #we get every picture in letter folder, preprocess it, and save to
        # preprocessed folder
        letter_pics = os.listdir(RAW_LETTERS_DIR + os.sep + letter_dir)
        for letter in letter_pics:

            current_pic = RAW_LETTERS_DIR + os.sep + letter_dir + os.sep + letter
            im = cv2.imread(current_pic, 0)
            letters = fun.segmentation_train(im)
            letters = fun.resize_binarize(letters)
            letters = fun.skeletonise(letters)

            cv2.imwrite(PREPROCESSED_LETTERS_DIR + os.sep + letter_dir + os.sep + letter, letters[0][0])

        print(ind1)