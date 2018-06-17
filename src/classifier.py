from keras.models import model_from_json
import cv2
import numpy as np
from preprocessing import reshape

# Download from

# https://www.kaggle.com/crawford/emnist/data

# emnist-balanced-mapping.txt
# emnist-balanced-test.csv
# emnist-balanced-train.csv


def load_network():
    """
    Loads trained network
    :return: Keras.models.Sequential
    """
    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    return loaded_model


def evaluate_letter(letter, network):
    """
    Evaluating letters from text.
    :param letter:
    :return:
    """
    input_data = np.array(letter).reshape((1, 28*28)) / 255
    result = network.predict_classes(input_data)
    return result


def map_label_to_name(label):
    """
    Maps labels to real names. E.g. 32 -> W (87 is ASCII)
    :param label: <int>
    :return: <char> label
    """
    with open("resources/emnist-balanced-mapping.txt") as f:
        lines = f.readlines()

    label_name = dict()
    for entry in lines:
        key = int(entry.split(' ')[0])
        value = chr(int(entry.split(' ')[1].strip()))
        label_name[key] = value

    return label_name[label]



if __name__ == "__main__":
    # network = load_network()
    # letter = reshape('processed/l00_21.bmp')
    # print(evaluate_letter(letter, network))
    print(map_label_to_name(11))
