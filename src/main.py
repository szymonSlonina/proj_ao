import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    #czytanie zdjecia
    im = cv2.imread("./test_pictures/ocr4.png", 0)

    #NOTE: zwiększenie kontrastu powoduje że histogram jest wyrównany, ale w strone
    #   szumów tła, co powoduje że binaryzacja Otsu się wywala.
    #zwiększenie kontrastu wyrównaniem histogramu
    #im = cv2.equalizeHist(im)

    #liczenie oraz wyswietlanie histogramu
    hist = cv2.calcHist([im], [0], None, [256], [0, 256])
    plt.bar(range(0, 256), [x[0] for x in hist])
    plt.show()

    #tresholding globalny metodą otsu (bo w histogramie są 2 widoczne piki)
    ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)

    #erozja
    im_negative = cv2.bitwise_not(im)
    kernel = np.ones((2, 1), np.uint8)
    im_negative = cv2.erode(im_negative, kernel, iterations=1)

    #deskewing
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

    cv2.imshow('processed', im)
    cv2.waitKey()