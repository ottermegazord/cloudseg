
import numpy as np
import skimage
import os
import PIL.Image
from sklearn.model_selection import train_test_split
import cv2
from sklearn import svm
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score

X = np.array((784, 30240, 1))
y = list()
hog_images = []
hog_features=[]

def alpha2int(alphabet):
    # options = {
    #     'A': 0,
    #     'B': 1,
    #     'C': 2,
    #     'D': 3,
    #     'E': 4
    # }
    #
    # return options.get(alphabet, "Invalid alphabet")

    options = {
        'D': 1
    }

    return options.get(alphabet, 0)
# print alpha2int('B')

root = 'images/cloud/swimcat'

categories = [1, 2, 3, 4, 5]
# hog = cv2.HOGDescriptor()
ppc = 28

for path, subdirs, files in os.walk(root):
    for name in files:
        img_path = os.path.join(path, name)
        # print (img_path)
        correct_cat = categories[alpha2int(name[0])]
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        fd, hog_image = hog(im, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2',
                            visualise=True)
        hog_images.append(hog_image)
        hog_features.append(fd)
        y.append(correct_cat)


y = np.array(y)
hog_features = np.array(hog_features)
y = y.reshape(784,1)

print y

print (y.reshape(784,1)).shape
print hog_features.shape

X_train, X_test, y_train, y_test = train_test_split(hog_features, y)

clf = svm.LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))


