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

########################

class Cloud:

    def __init__(self, path):
        self.path = path

    def segmentation(self):

        '''Read image from directory'''
        self.img = cv.imread(self.path)
        img_copy = self.img.copy()
        img = cv.medianBlur(self.img, 5)

        '''Convert to HSV and extract V'''
        imgHSV = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
        imgV = imgHSV[2]

        '''Apply Gaussian filter to remove noise'''
        imgGaussian = cv.GaussianBlur(img, (5, 5), 0)

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

        self.gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(self.gray, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return ret, thresh

    def percent(self):

        ret, thresh = self.segmentation()
        height, width = thresh.shape
        resolution = height * width
        white_pixels = cv.countNonZero(thresh)
        return white_pixels / float(resolution)

    def draw(self):

        ret, thresh = self.segmentation()
        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        vis = self.img.copy()
        cv.drawContours(vis, contours, -1, (0, 255, 0), 3)

        output = np.hstack(
            (self.img, cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR), cv.cvtColor(thresh, cv.COLOR_GRAY2BGR), vis))
        # output = np.hstack((vis))
        cv.namedWindow(self.path, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.path, 1000, 1000)
        cv.imshow(self.path, output)
        cv.waitKey(0)
        cv.destroyAllWindows()



