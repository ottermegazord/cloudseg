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
import pylab
import mahotas as mh

########################

directory = 'images/hdr'
mser = cv.MSER_create()

'''Define sun'''

lower_blue = np.array([235, 235, 235])
upper_blue = np.array([255, 255, 255])
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([255,255,255])


for filename in os.listdir(directory):
    path = directory + '/' + filename

    '''Read image from directory'''
    img = cv.imread(path)
    img_copy = img.copy()
    img = cv.medianBlur(img, 5)

    '''Convert to HSV and extract V'''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    output = np.hstack((img, cv.cvtColor(gray, cv.COLOR_GRAY2BGR), cv.cvtColor(thresh, cv.COLOR_GRAY2BGR),
                        cv.cvtColor(sure_bg, cv.COLOR_GRAY2BGR), cv.cvtColor(sure_fg, cv.COLOR_GRAY2BGR),
                        ))
    cv.namedWindow(path, cv.WINDOW_NORMAL)
    cv.resizeWindow(path, (1000, 1000))
    cv.imshow(path, output)
    cv.waitKey(0)
    cv.destroyAllWindows()
