import os

import glob
import random
import shutil
import math as m
from sklearn.model_selection import train_test_split

IMAGES_PER_CLASS = 10000
subsets = ['train', 'val', 'test']
subsets_prop = [0.75, 0.2, 0.05]

ds1_path = '/media/jcneves/DATASETS/NVIDIA_FakeFace/byimg_alignedlib_0.3/'
ds2_path = '/media/jcneves/DATASETS/VGG_FACE_2/byid_alignedlib_0.3_train/'
out_path = '/media/jcneves/DATASETS/real2fake_standard/'

ds1_files = [f for f in glob.glob(ds1_path + "*", recursive=True)]
ds2_files = [f for f in glob.glob(ds2_path + "*", recursive=True)]
print("Found " + str(len(ds1_files))  + " in dataset 1")
print("Found " + str(len(ds2_files))  + " in dataset 2")

imgs_per_id = max(m.ceil(IMAGES_PER_CLASS/len(ds1_files)), m.ceil(IMAGES_PER_CLASS/len(ds2_files)))
ids_to_use = m.floor(IMAGES_PER_CLASS/imgs_per_id)
print(ids_to_use)

#ds1_files = ds1_files[:ids_to_use]
#ds2_files = ds2_files[:ids_to_use]


# DIVIDE INTO SUBSETS BY IDENTITY

ds1_files_subsets = dict()
ds1_files_subsets[subsets[0]], ds1_files_subsets[subsets[2]] = \
    train_test_split(ds1_files, test_size=subsets_prop[2], shuffle=True, random_state=42)
ds1_files_subsets[subsets[0]], ds1_files_subsets[subsets[1]] = \
    train_test_split(ds1_files_subsets[subsets[0]], test_size=subsets_prop[1]/(subsets_prop[0]+subsets_prop[1]), shuffle=True, random_state=42)

ds2_files_subsets = dict()
ds2_files_subsets[subsets[0]], ds2_files_subsets[subsets[2]] = \
    train_test_split(ds2_files, test_size=subsets_prop[2], shuffle=True, random_state=42)
ds2_files_subsets[subsets[0]], ds2_files_subsets[subsets[1]] = \
    train_test_split(ds2_files_subsets[subsets[0]], test_size=subsets_prop[1]/(subsets_prop[0]+subsets_prop[1]), shuffle=True, random_state=42)

n_img = dict()
n_img[subsets[0]] = IMAGES_PER_CLASS*subsets_prop[0]
n_img[subsets[1]] = IMAGES_PER_CLASS*subsets_prop[1]
n_img[subsets[2]] = IMAGES_PER_CLASS*subsets_prop[2]

# CLASS 0

for s in subsets:

    try:
        os.makedirs(out_path + s + '/0')
    except OSError:
        print("Creation of the directory %s failed" % out_path)
    else:
        print("Successfully created the directory %s " % out_path)

    idx = 0
    for i in range(len(ds1_files_subsets[s])):

        img_file = ds1_files_subsets[s][i]
        #img_files = [f for f in glob.glob(id_path + "/*.jpg")]
        #img_files = img_files[:imgs_per_id]

        img_name = 'img_{:06d}.jpg'.format(idx)
        shutil.copyfile(img_file, out_path + s + '/0/' + img_name)
        idx = idx + 1

        if idx >= n_img[s]:
            break





# CLASS 1

for s in subsets:

    try:
        os.makedirs(out_path + s + '/1')
    except OSError:
        print("Creation of the directory %s failed" % out_path)
    else:
        print("Successfully created the directory %s " % out_path)

    idx = 0
    for i in range(len(ds2_files_subsets[s])):

        id_path = ds2_files_subsets[s][i]
        img_files = [f for f in glob.glob(id_path + "/*.jpg")]
        #img_files = img_files[:imgs_per_id]

        for f in img_files:

            img_name = 'img_{:06d}.jpg'.format(idx)
            shutil.copyfile(f, out_path + s + '/1/' + img_name)
            idx = idx + 1

            if idx >= n_img[s]:
                break

        if idx >= n_img[s]:
            break