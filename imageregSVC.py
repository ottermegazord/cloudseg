
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
from sklearn import svm
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

names = ['Nearest Neighbors', 'Decision Tree', 'Gradient Boost Classifier',
         ' Gaussian Process Classifier', 'MLP Classifier', 'SVC']

params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(random_state=0),
    ensemble.GradientBoostingClassifier(**params),
    GaussianProcessClassifier(),
    MLPClassifier(),
    SVC()
]

X = np.array((784, 30240, 1))
y = list()
hog_images = []
hog_features=[]

def alpha2int(alphabet):
    options = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }

    return options.get(alphabet, "Invalid alphabet")


root = 'images/cloud/swimcat'

# categories = [, 2, 3, 4, 5]
categories = ['Sky', 'Pattern', 'Thick-Dark', 'Thick-White', 'Veil']
# hog = cv2.HOGDescriptor()
ppc = 28

for path, subdirs, files in os.walk(root):
    for name in files:
        img_path = os.path.join(path, name)
        # print (name[0])
        # print(alpha2int(name[0]))
        # print (img_path)
        correct_cat = categories[alpha2int(name[0])]
        # print correct_cat
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

# print y
#
# print (y.reshape(784,1)).shape
# print hog_features.shape

X_train, X_test, y_train, y_test = train_test_split(hog_features, y)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    print name
    print clf.score(X_test, y_test)

print "KNN Classifier"
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
# clf = svm.LinearSVC()
# clf.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

print "Decision Tree Classifier"
DTC = DecisionTreeClassifier(random_state=0)
DTC.fit(X_train, y_train)
# clf = svm.LinearSVC()
# clf.fit(X_train, y_train)
y_pred = DTC.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

print "Gradient Boosting Classifier"
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
GBC = ensemble.GradientBoostingClassifier(**params)
GBC.fit(X_train, y_train)
# clf = svm.LinearSVC()
# clf.fit(X_train, y_train)
y_pred = GBC.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

print GBC.predict(X_test)

print "Gradient Process Classifier"
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
GPC = GaussianProcessClassifier()
GPC.fit(X_train, y_train)
# clf = svm.LinearSVC()
# clf.fit(X_train, y_train)
y_pred = GPC.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

print GPC.predict(X_test)


