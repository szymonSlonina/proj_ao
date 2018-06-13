import cv2
import os
import numpy as np


def generate_title(prefix, row_index, letter_index):
    """
    Title generator.
    :param prefix:
    :param row_index:
    :param letter_index:
    :return:
    """
    row_index = str(row_index) if row_index > 9 else '0' + str(row_index)
    letter_index = str(letter_index) if letter_index > 9 else '0' + str(letter_index)
    return "{}{}_{}.bmp".format(prefix, row_index, letter_index)


def save_letters(letters):
    """
    Saves every letter to src/processed/
    :param letters: 2 dim table of letters [row][letter]
    :return: <list> of paths to saved letters <string>
    """
    saved_letters = list()
    for row_index, row in enumerate(letters):
        for letter_index, letter in enumerate(row):
            path = os.path.join('processed', generate_title('l', row_index, letter_index))
            saved_letters.append(path)
            cv2.imwrite(path, letter)
    return saved_letters


def reshape_letters(letters_paths):
    """
    Reshape letters and saves them to reshaped/
    :param letters_paths: <list> of paths to letters <string>
    :return: None
    """
    for letter_path in letters_paths:
        letter = reshape(letter_path)
        path = os.path.join('reshaped', letter_path.split('/')[-1])
        print(path)
        cv2.imwrite(path, letter)


def remove_redundant_bottom(im):
    """
    Removes white pixels from bottom if all row is white.
    :param im: Image
    :return: Image without bottom white rows.
    """
    arr = np.array(im)
    reversed_arr = arr[::-1]
    reversed_arr = remove_redundant_top(reversed_arr)
    arr = reversed_arr[::-1]
    return arr


def remove_redundant_top(im):
    """
    Removes white pixels from top if all row is white.
    :param im: Image
    :return: Image without top white rows.
    """
    arr = np.array(im)
    to_delete = list()
    for i, row in enumerate(arr):
        row = row == 255
        if all(row):
            to_delete.append(i)
        else:
            break

    arr = np.delete(arr, to_delete, axis=0)
    return arr


def reshape(letter_path):
    """
    Reshape letter to 28x28 pixels.
    Before reshaping removes redundant top and bottom white rows.
    :param letter_path: <string>
    :return: reshaped letter
    """
    im = cv2.imread(letter_path, 0)
    im = remove_redundant_top(im)
    im = remove_redundant_bottom(im)
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    return im


if __name__ == "__main__":
    reshape('processed/l00_02.bmp')
