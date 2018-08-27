
from cloudsegmentation import Cloud

path = '/home/ottermegazord/PycharmProjects/cloudseg/images/cloud/swimcat/B-pattern/images/B_9img.png'
cloud = Cloud(path)
cloud.segmentation()
print cloud.percent()
# cloud.draw()