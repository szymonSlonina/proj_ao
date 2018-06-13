import cv2
import os


def generate_title(row_index, letter_index):
    row_index = str(row_index) if row_index > 9 else '0' + str(row_index)
    letter_index = str(letter_index) if letter_index > 9 else '0' + str(letter_index)
    return 'l' + row_index + '_' + letter_index + '.bmp'


def save_letters(letters):
    """
    Saves every letter to src/processed/
    :param letters: 2 dim table of letters [row][letter]
    :return: None
    """
    for row_index, row in enumerate(letters):
        for letter_index, letter in enumerate(row):
            cv2.imwrite(
                os.path.join('processed',
                             generate_title(row_index, letter_index)),
                letter)
