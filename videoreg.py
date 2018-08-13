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
import video
import sys

########################

directory = 'images/thick'
mser = cv.MSER_create()

if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    # cam = video.create_capture(video_src)
    cam = cv.VideoCapture('vid3.mp4')
    mser = cv.MSER_create()

    while True:
        ret, img = cam.read()
        if ret == 0:
            break
        vis = img.copy()

        '''Read image from directory'''
        img_copy = img.copy()
        img = cv.medianBlur(img, 5)

        '''Apply Gaussian filter to remove noise'''
        imgGaussian = cv.GaussianBlur(img, (5,5), 0)

        '''Apply Laplacian filter acute edges of the foreground'''

        # Create second-order kernel
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        imgLaplace = cv.filter2D(imgGaussian, cv.CV_32F, kernel)
        sharpen = np.float32(img)
        result = sharpen - imgLaplace

        # Convert to 8-bit grayscale
        result = np.clip(result, 0, 255)
        result = result.astype('uint8')

        # Convert to 8-bit grayscale
        imgLaplace = np.clip(imgLaplace, 0, 80)
        imgLaplace = imgLaplace.astype('uint8')

        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

        # '''MSER Segmentation'''
        #
        # regions, _ = mser.detectRegions(gray)
        # mser_vis = img.copy()
        #
        # hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        # cv.polylines(mser_vis, hulls, 1, (0, 255, 0))
        #
        # mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        # mask = cv.dilate(mask, np.ones((150, 150), np.uint8))
        #
        # for contour in hulls:
        #     #cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        #
        #     text_only = cv.bitwise_and(img, img, mask=mask)
        #
        # for i, contour in enumerate(hulls):
        #     x, y, w, h = cv.boundingRect(contour)

        '''Segmentation'''

        # ret, thresh = cv.adaptiveThreshold(gray, 0, 255, cv.THRESH_BINARY)

        # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                            cv.THRESH_BINARY, 11, 2)
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        vis = img.copy()
        cv.drawContours(vis, contours, -1, (0, 255, 0), 3)

        output = np.hstack((img, cv.cvtColor(gray, cv.COLOR_GRAY2BGR), cv.cvtColor(thresh, cv.COLOR_GRAY2BGR), vis))
        output = cv.resize(output, (1000,250))
        cv.imshow('test', output)
        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()


