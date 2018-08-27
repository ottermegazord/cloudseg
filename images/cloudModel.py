
import cv2
from skimage.feature import hog
import pickle
from cloudsegmentation import Cloud
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def alpha2int(alphabet):
    options = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4
    }

    return options.get(alphabet, "Invalid alphabet")


filename = '/home/ottermegazord/PycharmProjects/cloudseg/GBC_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
ppc = 28

root = '/home/ottermegazord/PycharmProjects/cloudseg/images/cloud/swimcat'
# root = '/home/ottermegazord/PycharmProjects/cloudseg/images/test'

categories = ['Sky', 'Pattern', 'Thick-Dark', 'Thick-White', 'Veil']


for path, subdirs, files in os.walk(root):
    for name in files:
        img_path = os.path.join(path, name)
        im = cv2.imread(img_path)
        im = cv2.resize(im, (125, 125))
        # correct_cat = categories[alpha2int(name[0])]
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        fd, hog_image = hog(im, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2',
                        visualise=True)

        result = loaded_model.predict(fd.reshape(1,-1))
        cloud = Cloud(img_path)
        cloud.segmentation()
        percent = cloud.percent()
        vis = cv2.imread(img_path)
        # title = 'Actual: %s, Predicted: %s, Cloud Percipitation: %.4f' % (correct_cat, result[0], percent)
        title = 'Predicted: %s, Cloud Percipitation: %.4f' % (result[0], percent)
        print title
        # output = zip(categories,result[0])
        # output = np.transpose(output)

        # print(img_path)
        # for i in range(0, output.shape[1]):
        #     cat, score = output[:, i]
        #     print 'Actual: %s' % correct_cat
        #     print 'Prediction: %s, Score: %.3f' % (cat, float(score))
        #
        # # Plotting of images
        # img_plt = mpimg.imread(img_path)
        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # ax1.imshow(img_plt, interpolation='None', aspect='auto')
        # ax2.imshow(hog_image)
        # ax1.set_title('Original Image')
        # ax2.set_title('HOG Image')
        # plt.figtext(0.2, 0.15, title)
        # plt.show()

        # f, axarr = plt.subplots(2,1)
        # img_plt = mpimg.imread(img_path)
        # axarr[0,0] = plt.imshow(img_plt)
        # axarr[0,1] = plt.imshow(hog)
        # plt.show()

        # Plotting of images
        # img_plt = mpimg.imread(hog)
        # plt.imshow(hog)
        # plt.show()



        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 500, 500)
        cv2.imshow(title, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



