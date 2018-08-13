'''
***************************************************************************************
*
*                   Yara Cloud Segmentation
*
*
*  Name : Idaly Ali
*
*  Designation : Data Scientist
*
*  Description : Cloud segmentation program for SkyWeather
*
*
***************************************************************************************

'''

########################

import numpy as np
import cv2 as cv
import os

########################

directory = 'images/hdr'
mser = cv.MSER_create()

for filename in os.listdir(directory):
    path = directory + '/' + filename

    '''Read image from directory'''
    img = cv.imread(path)
    img_copy = img.copy()
    img = cv.medianBlur(img, 5)

    '''Convert to HSV and extract V'''
    hsv = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
    # imgV = imgHSV[2]

    cv.imshow(path, hsv[2])
    cv.waitKey(0)
    cv.destroyAllWindows()


    # '''Apply Gaussian filter to remove noise'''
    # imgGaussian = cv.GaussianBlur(img, (5,5), 0)
    #
    # '''Apply Laplacian filter acute edges of the foreground'''
    #
    # # Create second-order kernel
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # imgLaplace = cv.filter2D(imgGaussian, cv.CV_32F, kernel)
    # sharpen = np.float32(img)
    # result = sharpen - imgLaplace
    #
    # # Convert to 8-bit grayscale
    # result = np.clip(result, 0, 255)
    # result = result.astype('uint8')
    #
    # # Convert to 8-bit grayscale
    # imgLaplace = np.clip(imgLaplace, 0, 80)
    # imgLaplace = imgLaplace.astype('uint8')
    #
    # gray = cv.cvtColor(result,cv.COLOR_BGR2GRAY)
    #
    # '''Detect Sun'''
    #
    # # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)
    # # circles = np.uint16(np.around(circles))
    # # sun = img.copy()
    # # for i in circles[0, :]:
    # #     # draw the outer circle
    # #     cv.circle(sun, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # #     # draw the center of the circle
    # #     cv.circle(sun, (i[0], i[1]), 2, (0, 0, 255), 3)
    #
    # '''Segmentation'''
    #
    # # ret, thresh = cv.adaptiveThreshold(gray, 0, 255, cv.THRESH_BINARY)
    #
    # # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
    # #                            cv.THRESH_BINARY, 11, 2)
    # ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # vis = img.copy()
    # cv.drawContours(vis, contours, -1, (0, 255, 0), 3)
    #
    # output = np.hstack((img, cv.cvtColor(gray, cv.COLOR_GRAY2BGR), cv.cvtColor(thresh, cv.COLOR_GRAY2BGR), vis))
    # # output = np.hstack((vis))
    # cv.namedWindow(path, cv.WINDOW_NORMAL)
    # cv.resizeWindow(path, 1000, 1000)
    # cv.imshow(path, output)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #
