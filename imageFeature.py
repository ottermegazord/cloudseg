
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
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import label_binarize
import pickle
from sklearn.naive_bayes import GaussianNB

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
        #im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
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

# for name, clf in zip(names, classifiers):
#     clf.fit(X_train, y_train)
#     print name
#     print clf.score(X_test, y_test)

# print "KNN Classifier"
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X_train, y_train)
# # clf = svm.LinearSVC()
# # clf.fit(X_train, y_train)
# y_pred = neigh.predict(X_test)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))
#
# print "Decision Tree Classifier"
# DTC = DecisionTreeClassifier(random_state=0)
# DTC.fit(X_train, y_train)
# # clf = svm.LinearSVC()
# # clf.fit(X_train, y_train)
# y_pred = DTC.predict(X_test)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))

# print "Gradient Boosting Classifier"
# params = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
# gsearch1 = GridSearchCV(ensemble.GradientBoostingClassifier(n_estimators=1200, max_depth=5, subsample=0.9, max_features=15,
#                                                             learning_rate=0.3, min_samples_leaf=40, random_state=3,
#                                                             min_samples_split=50, verbose=True),
#                                                             param_grid = params,n_jobs=4,iid=False, cv=5)
# c, r = y_train.shape
# y_train = y_train.reshape(c,)
#
# # y_train = label_binarize(y_train, classes=categories)
# # y_train.ravel()
# gsearch1.fit(X_train, y_train)
#
# # print X_train.shape
# # print y_train.shape
# # print X_train.shape
# #
# print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# # filename = 'GBC_Model.sav'
# # y_pred = gsearch1.predict(X_test)
# # print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# # print("Score: %.3f" % gsearch1.score(X_test, y_test))
# # print('\n')
# # print(classification_report(y_test, y_pred))
# # pickle.dump(GBC, open(filename, 'wb'))

# print "Gradient Boosting Classifier"
# # params = {'n_estimators': 1200, 'max_depth': 9, 'subsample': 0.5, 'max_features': 'sqrt',
# #           'learning_rate': 0.2, 'min_samples_leaf': 60, 'random_state': 3, 'verbose':True}
# # GBC = ensemble.GradientBoostingClassifier(**params)
# GBC = ensemble.GradientBoostingClassifier(n_estimators=1200, max_depth=5, subsample=0.9, max_features=15,
#                                                             learning_rate=0.3, min_samples_leaf=40, random_state=3,
#                                                             min_samples_split=50, verbose=True)
# GBC.fit(X_train, y_train)
# filename = '_Model.sav'
# y_pred = GBC.predict(X_test)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print("Score: %.3f" % GBC.score(X_test, y_test))
# print('\n')
# print(classification_report(y_test, y_pred))
# pickle.dump(GBC, open(filename, 'wb'))

print "Gaussian NB Classifier"
# params = {'n_estimators': 1200, 'max_depth': 9, 'subsample': 0.5, 'max_features': 'sqrt',
#           'learning_rate': 0.2, 'min_samples_leaf': 60, 'random_state': 3, 'verbose':True}
# GBC = ensemble.GradientBoostingClassifier(**params)
GNB = GaussianNB()
GNB.fit(X_train, y_train)
filename = 'GNB_Model.sav'
y_pred = GNB.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print("Score: %.3f" % GNB.score(X_test, y_test))
print('\n')
print(classification_report(y_test, y_pred))
pickle.dump(GNB, open(filename, 'wb'))

# print "AdaBoost Classifier"
# params = {'n_estimators': 1200, 'learning_rate': 0.01, 'random_state': 3}
# ADA = ensemble.AdaBoostClassifier(**params)
# ADA.fit(X_train, y_train)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print("Score: %.3f" % ADA.score(X_test, y_test))
# print('\n')
# print(classification_report(y_test, y_pred))

# print "Gradient Process Classifier"
#
# params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
#           'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
# GPC = GaussianProcessClassifier()
# GPC.fit(X_train, y_train)
#
# # clf = svm.LinearSVC()
# # clf.fit(X_train, y_train)
# y_pred = GPC.predict(X_test)
# print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
# print('\n')
# print(classification_report(y_test, y_pred))
#
# print GPC.predict(X_test)
#

