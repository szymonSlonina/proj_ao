import numpy as np
import cv2


#letters są w formacie listy list z literami posegmentowanymi
def skeletonise(letters):
    for ind1, row in enumerate(letters):
        for ind2, letter in enumerate(row):
            letter = cv2.resize(letter, (32, 32), cv2.INTER_CUBIC)
            letters[ind1][ind2] = skeletonize(letter)

    return letters


def segmentation(im):
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
            if col_flag == True and mask[ind - 1] == True:
                letter_row[-1] = np.insert(letter_row[-1], -1, np.array(row[:, ind]), axis=1)

        letters.append(letter_row)

    return letters


def preproces_scan(im):
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
    letters = segmentation(im)

    #resize and binarize
    letters = resize_binarize(letters)

    # skeletonize all letters
    letters = skeletonise(letters)

    return letters


def resize_binarize(letters):
    for ind1, row in enumerate(letters):
        for ind2, im in enumerate(row):
            letters[ind1][ind2] = cv2.resize(im, (32, 32), cv2.INTER_CUBIC)
            th, letters[ind1][ind2] = cv2.threshold(letters[ind1][ind2], 254, 255, cv2.THRESH_BINARY)

    return letters


def skeletonize(image):
    sprawdzarka = [[128, 64, 32], [1, 0, 16], [2, 4, 8]]
    czworki = [3, 6, 7, 12, 14, 15, 24, 28, 30, 48, 56, 60, 96, 112, 120, 129, 131, 135, 192, 193, 195, 224, 225, 240]
    wyciecia = [3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48, 52, 53, 54, 55, 56, 60, 61, 62, 63, 65, 67,
                69, 71, 77, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101, 103, 109, 111, 112,
                113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 131, 133, 135, 141, 143, 149, 151, 157,
                159, 181, 183, 189, 191, 192, 193, 195, 197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217,
                219, 220, 221, 222, 223, 224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249,
                251, 252, 253, 254, 255]

    flag = True
    while True:
        img_mask = cv2.bitwise_not(image) // 255

        if flag == True:
            img_mask = np.pad(img_mask, 1, 'constant', constant_values=0)
            flag = False

        #zamieniamy 1 na 2
        for ind1, row in enumerate(img_mask):
            for ind2, cell in enumerate(row):
                if cell == 1:
                    try:
                        if (img_mask[ind1-1, ind2] == 0 or img_mask[ind1+1, ind2] == 0
                        or img_mask[ind1, ind2-1] == 0 or img_mask[ind1, ind2+1] == 0):
                            img_mask[ind1, ind2] = 2
                    except IndexError as e:
                        pass

        #pozostałe 1 zamieniamy na 3
        for ind1, row in enumerate(img_mask):
            for ind2, cell in enumerate(row):
                if cell == 1:
                    try:
                        if (img_mask[ind1-1, ind2-1] == 0 or img_mask[ind1+1, ind2+1] == 0
                        or img_mask[ind1-1, ind2+1] == 0 or img_mask[ind1+1, ind2-1] == 0):
                            img_mask[ind1, ind2] = 3
                    except IndexError as e:
                        pass

        #zamieniamy 2 na 4 jeśli się zgadza
        for ind1, row in enumerate(img_mask):
            for ind2, cell in enumerate(row):
                if cell == 2:
                    try:
                        suma = 0
                        for y in range(-1, 2):
                            for x in range(-1, 2):
                                if not(y == 0 and x == 0):
                                    suma += sprawdzarka[1+y][1+x] if img_mask[ind1+y, ind2+x] != 0 else 0

                        if suma in czworki:
                            img_mask[ind1, ind2] = 4
                    except IndexError as e:
                        pass

        #zamieniamy 4 na 0 lub 1
        for ind1, row in enumerate(img_mask):
            for ind2, cell in enumerate(row):
                if cell == 4:
                    try:
                        suma = 0
                        for y in range(-1, 2):
                            for x in range(-1, 2):
                                if not(y == 0 and x == 0):
                                    suma += sprawdzarka[1+y][1+x] if img_mask[ind1+y, ind2+x] != 0 else 0

                        if suma in wyciecia:
                            img_mask[ind1, ind2] = 0
                        else:
                            img_mask[ind1, ind2] = 1
                    except IndexError as e:
                        pass

        #zamieniamy 2 na 0 lub 1
        for ind1, row in enumerate(img_mask):
            for ind2, cell in enumerate(row):
                if cell == 2:
                    try:
                        suma = 0
                        for y in range(-1, 2):
                            for x in range(-1, 2):
                                if not(y == 0 and x == 0):
                                    suma += sprawdzarka[1+y][1+x] if img_mask[ind1+y, ind2+x] != 0 else 0

                        if suma in wyciecia:
                            img_mask[ind1, ind2] = 0
                        else:
                            img_mask[ind1, ind2] = 1
                    except IndexError as e:
                        pass

        #zamieniamy 3 na 0 lub 1
        for ind1, row in enumerate(img_mask):
            for ind2, cell in enumerate(row):
                if cell == 3:
                    try:
                        suma = 0
                        for y in range(-1, 2):
                            for x in range(-1, 2):
                                if not(y == 0 and x == 0):
                                    suma += sprawdzarka[1+y][1+x] if img_mask[ind1+y, ind2+x] != 0 else 0

                        if suma in wyciecia:
                            img_mask[ind1, ind2] = 0
                        else:
                            img_mask[ind1, ind2] = 1
                    except IndexError as e:
                        pass

        if np.all(image == cv2.bitwise_not(img_mask * 255)):
            image = cv2.bitwise_not(img_mask * 255)
            break
        else:
            image = cv2.bitwise_not(img_mask * 255)

    image = image[1:-1, 1:-1]
    return image