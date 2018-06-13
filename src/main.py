import cv2
import numpy as np
from preprocessing import save_letters

if __name__ == "__main__":

    # czytanie zdjecia
    im = cv2.imread("./test_pictures/test1.bmp", 0)

    # NOTE: zwiększenie kontrastu powoduje że histogram jest wyrównany, ale w strone
    #   szumów tła, co powoduje że binaryzacja Otsu się wywala.
    # zwiększenie kontrastu wyrównaniem histogramu
    # im = cv2.equalizeHist(im)

    # liczenie oraz wyswietlanie histogramu
    # hist = cv2.calcHist([im], [0], None, [256], [0, 256])
    # plt.bar(range(0, 256), [x[0] for x in hist])
    # plt.show()

    # tresholding globalny metodą otsu (bo w histogramie są 2 widoczne piki)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)

    # erozja
    im_negative = cv2.bitwise_not(im)
    kernel = np.ones((2, 1), np.uint8)
    im_negative = cv2.erode(im_negative, kernel, iterations=1)

    # deskewing
    coords = np.column_stack(np.where(im_negative > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = im_negative.shape[:2]
    center = (h // 2, w // 2)
    rotM = cv2.getRotationMatrix2D(center, angle, 1.)
    im_negative = cv2.warpAffine(im_negative, rotM, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ret, im_negative = cv2.threshold(im_negative, 0, 255, cv2.THRESH_OTSU)
    im = cv2.bitwise_not(im_negative)

    # segmentation

    # extracting rows of text
    text_row = []
    mask = (np.sum(im, axis=1) / im.shape[1]) != 255
    for ind, row_flag in enumerate(mask):
        if row_flag == True and mask[ind - 1] == False:
            text_row.append([])
            text_row[-1].append(im[ind])
        elif row_flag == True and mask[ind - 1] == True:
            text_row[-1].append(im[ind])

    for ind, row in enumerate(text_row):
        row = np.array(row)
        text_row[ind] = row

    # extracting letters from rows of text
    letters = []
    for row in text_row:
        letter_row = []
        mask = (np.sum(row, axis=0) / row.shape[0]) != 255
        for ind, col_flag in enumerate(mask):
            if col_flag == True and mask[ind - 1] == False:
                letter_row.append(np.array(row[:, ind])[:, np.newaxis])
                print(len(letter_row))
            if col_flag == True and mask[ind - 1] == True:
                letter_row[-1] = np.insert(letter_row[-1], -1, np.array(row[:, ind]), axis=1)

        letters.append(letter_row)

    # variable letters has all letters... this is 2 dim table of format letters[row][letter].
    save_letters(letters)
