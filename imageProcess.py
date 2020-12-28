import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt


def imPreProcess(imPath):
    im = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
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

    img_cnt = cv2.drawContours(im.copy(), [c], -1, (199, 0, 129), 5)

    # add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 0, 0), -1)

    # crop
    ADD_PIXELS = 0
    new_img = im[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    return new_img