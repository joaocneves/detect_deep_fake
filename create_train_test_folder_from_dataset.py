import os
import glob
import shutil
import random
from sklearn.model_selection import train_test_split

ds_path = '/media/jcneves/DATASETS/real2real'
out_path = '/media/jcneves/DATASETS/real2real_standard'
classes_path = glob.glob(ds_path + '/*')

classes_names = os.listdir(ds_path)

for i,c_path in enumerate(classes_path):

    img_class_path = glob.glob(c_path + '/*.png')

    img_class_path_train, img_class_path_test = train_test_split(img_class_path, test_size=0.15, shuffle=True, random_state=42)
    img_class_path_train, img_class_path_val = train_test_split(img_class_path_train, test_size=0.15/0.85, shuffle=True,
                        random_state=42)
    print("{:d} train images of class {}".format(len(img_class_path_train), classes_names[i]))
    print("{:d} val images of class {}".format(len(img_class_path_val), classes_names[i]))
    print("{:d} test images of class {}".format(len(img_class_path_test), classes_names[i]))

    os.makedirs(out_path + "/train/" + classes_names[i])
    os.makedirs(out_path + "/val/" + classes_names[i])
    os.makedirs(out_path + "/test/" + classes_names[i])

    for f in img_class_path_train:
        img_name = f.split('/')
        img_name = img_name[-1]
        shutil.copyfile(f, out_path + "/train/" + classes_names[i] + "/" + img_name)

    for f in img_class_path_val:
        img_name = f.split('/')
        img_name = img_name[-1]
        shutil.copyfile(f, out_path + "/val/" + classes_names[i] + "/" + img_name)

    for f in img_class_path_test:
        img_name = f.split('/')
        img_name = img_name[-1]
        shutil.copyfile(f, out_path + "/test/" + classes_names[i] + "/" + img_name)


