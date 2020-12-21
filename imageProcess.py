import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt


def imPreProcess(imPath):
    im = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
    # im = cv2.resize(
    #     im,
    #     (256, 256),
    #     interpolation=cv2.INTER_CUBIC
    # )
    # gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(im, (5, 5), 0) # Gaussian smoothing

    thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1] # binarization
    thresh = cv2.erode(thresh, None, iterations=2) # erosion, removes minor noise
    thresh = cv2.dilate(thresh, None, iterations=2) # dilation, enlarges features

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # get contour whatever the opencv version
    c = max(cnts, key=cv2.contourArea)  # get contour with biggest area

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    img_cnt = cv2.drawContours(im.copy(), [c], -1, (0, 255, 255), 4)

    # add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

    # crop
    ADD_PIXELS = 0
    new_img = im[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    # plt.figure(figsize=(15, 6))
    # plt.subplot(141)
    # plt.imshow(im)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 1. Get the original image')
    # plt.subplot(142)
    # plt.imshow(img_cnt)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 2. Find the biggest contour')
    # plt.subplot(143)
    # plt.imshow(img_pnt)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 3. Find the extreme points')
    # plt.subplot(144)
    # plt.imshow(new_img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 4. Crop the image')
    # # plt.ion()
    # plt.show()

    return new_img

# imPreProcess('./images/raw\RMI_DEG/0043_STIR_0011.png'.replace('\\','/'))